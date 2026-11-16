"""Train a text-to-image EDM model on sprite32 data with diffusers + accelerate.

    .venv/bin/python hf_train.py configs/smoke.yaml
    .venv/bin/python hf_train.py configs/sprite32.yaml --set batch_size=64 num_steps=1000
    accelerate launch hf_train.py configs/sprite32.yaml          # same thing, via accelerate's launcher
    .venv/bin/python hf_train.py configs/sprite32.yaml --resume outputs_diffusers/sprite32/checkpoints/step_0050000

Library pieces: UNet2DConditionModel, EDM preconditioning from the diffusers EDM
scheduler (precondition_inputs / _noise / _outputs), EMAModel (one per decay),
diffusers.optimization.get_scheduler, Accelerator (device, mixed precision,
grad clipping, save_state / load_state). Custom: the EDM loss (lognormal sigma,
EDM weighting, optional uncertainty head), ~15 lines.

Outputs in output_dir: unet_raw/, unet_ema<decay>/ (diffusers save_pretrained),
scheduler/, run_config.json (sigma_data etc.), training_log.jsonl,
previews/step_XXXXXXX.png (row blocks: raw, then each EMA), checkpoints/.
Optional: exports/step_XXXXXXX/ (export_steps: kept weights, e.g. weak models for autoguidance),
validation loss per sigma on fixed val-split and training images (val_every; "val" records in the log),
only the newest keep_checkpoints checkpoints, label_cond (quality/category one-hots).

Also: qk_norm, conv_in_fp32, text_adapter (hf_common);
lr_schedule constant_cosine_tail; keep_checkpoint_every (resumable checkpoints that pruning leaves alone);
the largest attention logit per layer in every "val" record; the run ends if a qk_norm layer exceeds its bound
or (logit_stop, for runs without qk_norm) a self-attention layer exceeds that value; benchmark_exports (scripts/benchmark.sh on each export, "bench" records in the log);
page_cache_fraction (trim_page_cache, only in a container with a memory limit).
"""

import argparse
import json
import math
import random
import re
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.models.attention_processor import Attention
from diffusers.training_utils import EMAModel

from hf_common import (N_QUALITY, QKNorm, UncertaintyHead, add_labels, build_lr_scheduler, build_scheduler,
                       build_unet, freeze_conv1_bias, load_config, null_labels, save_run_meta, to_pil_grid)
from hf_data import make_dataset, make_loader, rms_sigma_data, val_paths
from hf_sample import sample_images
from hf_text import build_text_encoder


def export(accelerator, unet, emas, scheduler, out):
    model = accelerator.unwrap_model(unet)
    model.save_pretrained(out / "unet_raw")
    for decay, ema in emas.items():
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        model.save_pretrained(out / f"unet_ema{decay}")
        ema.restore(model.parameters())
    scheduler.save_pretrained(out / "scheduler")


@torch.no_grad()  # without it autograd records every lerp: a graph that grows each step (~1.5 MB/step host RAM)
def ema_step(emas, model):
    """EMAModel.step for every decay, as one foreach lerp each: s += (1 - d) * (p - s), the same update
    without diffusers' temporary difference tensors. Frozen params (constant) are skipped."""
    ps = list(model.parameters())
    grad = [p for p in ps if p.requires_grad]
    for ema in emas.values():
        ema.optimization_step += 1
        decay = ema.get_decay(ema.optimization_step)
        ema.cur_decay_value = decay
        torch._foreach_lerp_([s for s, p in zip(ema.shadow_params, ps) if p.requires_grad], grad, 1 - decay)


def preview(accelerator, unet, emas, text, cfg, out, step):
    model = accelerator.unwrap_model(unet)
    model.eval()
    sampler = build_scheduler(cfg, cfg.get("sampler", "dpmpp_2m"))
    blocks = []
    for name, ema in [("raw", None)] + list(emas.items()):
        if ema is not None:
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
        blocks.append(sample_images(model, text, sampler, cfg["sample_prompts"],
                                    cfg.get("sample_n", 2), cfg.get("sample_steps", 35),
                                    cfg.get("sample_guidance", 3.0), cfg.get("sample_interval"),
                                    seed=cfg.get("sample_seed", 0), channels=cfg.get("channels", 4),
                                    size=cfg.get("image_size", 32),
                                    labels=null_labels(cfg, cfg.get("sample_n", 2) * len(cfg["sample_prompts"]),
                                                       quality=N_QUALITY)))
        if ema is not None:
            ema.restore(model.parameters())
    model.train()
    path = out / "previews" / f"step_{step:07d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_grid(torch.cat(blocks), nrow=len(cfg["sample_prompts"])).save(path)
    return path


VAL_SIGMAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def load_val_sets(cfg, text, device, n):
    """Fixed val-split and training images with long captions, encoded once: {name: [(x, ctx, mask, pooled)]}."""
    sets = {}
    for name, paths in zip(("val", "train"), val_paths(cfg, n)):
        ds = make_dataset(cfg, paths=paths, caption_keys=["description"])
        batches, bs = [], 256
        for i in range(0, len(ds), bs):
            items = [ds[j] for j in range(i, min(i + bs, len(ds)))]
            x = torch.stack([it[0] for it in items]).to(device)
            caps = [it[1] for it in items]
            labels = None
            if cfg.get("label_cond"):
                labels = (torch.tensor([c[1] for c in caps]), torch.tensor([c[2] for c in caps]))
                caps = [c[0] for c in caps]
            ctx, mask, pooled = text(caps)
            batches.append((x, ctx, mask, add_labels(pooled, labels)))
        sets[name] = batches
    return sets


@torch.no_grad()
def validate(model, edm, sets, sd, device):
    """EDM-weighted denoising loss (no uncertainty term) per sigma, fixed noise; model in eval mode."""
    model.eval()
    out = {}
    for name, batches in sets.items():
        gen = torch.Generator(device=device).manual_seed(0)
        for sigma in VAL_SIGMAS:
            tot, n = 0.0, 0
            for x, ctx, mask, pooled in batches:
                s4 = torch.full((x.shape[0], 1, 1, 1), sigma, device=device)
                noise = torch.randn(x.shape, generator=gen, device=device)
                x_noisy = x + s4 * noise
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    F = model(edm.precondition_inputs(x_noisy, s4), edm.precondition_noise(s4.flatten()),
                              encoder_hidden_states=ctx, class_labels=pooled, encoder_attention_mask=mask).sample
                D = edm.precondition_outputs(x_noisy, F.float(), s4)
                weight = (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2
                tot += float((weight * (D - x) ** 2).mean(dim=(1, 2, 3)).sum())
                n += x.shape[0]
            out[f"{name}_{sigma:g}"] = tot / n
        out[f"{name}_mean"] = sum(out[f"{name}_{s:g}"] for s in VAL_SIGMAS) / len(VAL_SIGMAS)
    model.train()
    return out


@torch.no_grad()
def attention_logits(model, edm, batch, device, n=16, sigmas=(0.1, 1.0)):
    """Largest |q.k / sqrt(d)| per attention layer on n validation images: the quantity that ran away in an
    earlier run (one 32x32 self-attention layer: 102 at 100k steps, 826,000 at 400k; the others stayed under 60).
    q and k are taken from the layers' own to_q / to_k (+ norm_q / norm_k), so this follows whatever is trained.
    Returns ({layer: logit}, {layer: bound}); bound = scale * sqrt(head_dim) for layers with hf_common.QKNorm."""
    model.eval()
    layers = {name: m for name, m in model.named_modules() if isinstance(m, Attention)}
    grabbed, hooks, out = {}, [], {name: 0.0 for name in layers}
    bounds = {name: m.norm_q.gain ** 2 * (m.to_q.out_features // m.heads) ** 0.5
              for name, m in layers.items() if isinstance(m.norm_q, QKNorm)}
    for name, m in layers.items():
        hooks.append(m.to_q.register_forward_hook(lambda _m, _i, o, k=name: grabbed.__setitem__((k, "q"), o)))
        hooks.append(m.to_k.register_forward_hook(lambda _m, _i, o, k=name: grabbed.__setitem__((k, "k"), o)))
    x, ctx, mask, pooled = (t[:n] for t in batch)
    gen = torch.Generator(device=device).manual_seed(0)
    try:
        for sigma in sigmas:
            s4 = torch.full((x.shape[0], 1, 1, 1), sigma, device=device)
            x_noisy = x + s4 * torch.randn(x.shape, generator=gen, device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                model(edm.precondition_inputs(x_noisy, s4), edm.precondition_noise(s4.flatten()),
                      encoder_hidden_states=ctx, class_labels=pooled, encoder_attention_mask=mask)
            for name, m in layers.items():
                q, k = (grabbed[name, w].float().unflatten(-1, (m.heads, -1)).transpose(1, 2) for w in "qk")
                if m.norm_q is not None:
                    q, k = m.norm_q(q), m.norm_k(k)
                peak = float((q @ k.transpose(-1, -2)).abs().max()) / q.shape[-1] ** 0.5
                out[name] = max(out[name], peak)
    finally:
        for h in hooks:
            h.remove()
        model.train()
    return out, bounds


def run_benchmark(out, step, cfg):
    """scripts/benchmark.sh (QUICK: prompted FID at guidance 3 with long and short captions, and the prompt
    grid) on exports/step_XXXXXXX while training waits; ~10 min on a 5090. Returns {"long_g3": fid, ...}; a
    failed benchmark only prints a warning."""
    name = f"{out.name}_{step // 1000}k"
    bench = out / "benchmark"
    env = {**os.environ, "QUICK": "1", "R": str(bench.resolve()), "PY": sys.executable,
           "EMA": str(cfg.get("benchmark_ema", max(cfg["ema_decays"])))}
    if cfg.get("benchmark_stats"):  # reference file (scripts/benchmark.sh STATS)
        env["STATS"] = str(cfg["benchmark_stats"])
    if cfg.get("benchmark_extra"):  # scripts/bench_extra.py: text accuracy, seed diversity, near-copy rate
        env["EXTRA"] = "1"
    script = Path(__file__).resolve().parent / "scripts" / "benchmark.sh"
    export_dir = (out / "exports" / f"step_{step:07d}").resolve()
    try:
        subprocess.run(["bash", str(script), f"{name}={export_dir}"], env=env, check=True)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"  WARNING: benchmark of step {step} failed ({e}); see {bench}/eval.log", flush=True)
        return {}
    res = {}
    for line in open(bench / "results.txt"):
        m = re.fullmatch(rf"{re.escape(name)} (\S+) (?:Final FID: )?([\d.]+)\s*", line)
        if m:  # "<name> long_g3 Final FID: 9.87", or a bench_extra.py line "<name> text_exact 0.41"
            res[m.group(1)] = float(m.group(2))
    return res


def trim_page_cache(cfg, out):
    """In a container, cached file pages count against its memory limit, and near the limit the kernel stalls every
    process while it reclaims them: on the 2026-11-11 RunPod pod (117 GB limit, 85 GB text cache) training fell
    from 6.0 to 1.3 steps/s. Past page_cache_fraction of the limit (default 0.7), drop the cached copies of this
    run's large files, the text cache and the saved weights. They stay on disk and are read back as needed.
    Returns the GB that were in use, or None when nothing was dropped (always, when no limit is set)."""
    try:
        used, limit = (int(open(f"/sys/fs/cgroup/memory.{k}").read()) for k in ("current", "max"))
    except (OSError, ValueError):  # no cgroup files, or the limit is "max"
        return None
    if used < cfg.get("page_cache_fraction", 0.7) * limit:
        return None
    files = [*out.rglob("*.safetensors"), *out.rglob("*.bin")]
    if cfg.get("text_cache_dir"):
        files.append(Path(cfg["text_cache_dir"]).expanduser() / "tokens.f16")
    for f in files:
        try:
            fd = os.open(f, os.O_RDONLY)
        except OSError:
            continue
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    return used / 2 ** 30


def prune_checkpoints(ckpt_dir, keep, keep_every=0):
    """Delete all but the newest `keep` checkpoints/step_XXXXXXX dirs (only dirs named like that); steps that are
    multiples of keep_every stay."""
    steps = sorted(p for p in ckpt_dir.glob("step_*") if p.is_dir() and re.fullmatch(r"step_\d{7}", p.name))
    for p in steps[:-keep] if keep else []:
        if not (keep_every and int(p.name[5:]) % keep_every == 0):
            shutil.rmtree(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (YAML values)")
    ap.add_argument("--resume", default=None, help="checkpoints/step_XXXXXXX directory")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out = Path(cfg["output_dir"]).expanduser()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    accelerator = Accelerator(mixed_precision=cfg.get("mixed_precision", "no"))
    device = accelerator.device
    set_seed(cfg.get("seed", 0))

    dataset = make_dataset(cfg)
    loader = make_loader(cfg, dataset)
    if args.resume and not cfg.get("sigma_data"):
        prev = json.load(open(out / "run_config.json"))
        cfg["sigma_data"], cfg["P_mean"] = prev["sigma_data"], prev["P_mean"]
    if not cfg.get("sigma_data"):
        cfg["sigma_data"] = rms_sigma_data(loader)
    if not cfg.get("P_mean"):
        cfg["P_mean"] = math.log(cfg["sigma_data"])
    sd, P_mean, P_std = cfg["sigma_data"], cfg["P_mean"], cfg.get("P_std", 1.2)
    print(f"{len(dataset)} images | sigma_data {sd:.4f} | P_mean {P_mean:.4f} | device {device}")

    text = build_text_encoder(cfg, device)
    if cfg.get("text_adapter"):
        cfg["text_adapter_in_dim"] = text.dim
    unet = build_unet(cfg)
    if cfg.get("text_adapter") and not args.resume:  # a resumed run loads the statistics with the weights
        probe = [dataset._load_caption(p) for p in random.Random(0).sample(dataset.image_paths, 2048)]
        unet.text_adapter.set_stats(*text.feature_stats([c if isinstance(c, str) else c[0] for c in probe]))
    if cfg.get("freeze_conv1_bias"):
        freeze_conv1_bias(unet)  # needed for compile at widths 128/256/512, see hf_common
    if cfg.get("gradient_checkpointing"):
        unet.enable_gradient_checkpointing()
    u_head = UncertaintyHead() if cfg.get("uncertainty_weighting", True) else None
    edm = build_scheduler(cfg, "euler")  # used here only for its EDM preconditioning functions
    n_unet = sum(p.numel() for p in unet.parameters())
    n_text = text.param_count() if hasattr(text, "param_count") else sum(p.numel() for p in text.model.parameters())
    cfg["param_count"] = {"unet": n_unet, "text_encoder_frozen": n_text, "total": n_unet + n_text}
    print(f"UNet {n_unet / 1e6:.2f}M trainable + text encoder {n_text / 1e6:.2f}M frozen "
          f"= {(n_unet + n_text) / 1e6:.2f}M")

    params = list(unet.parameters()) + (list(u_head.parameters()) if u_head else [])
    params = [p for p in params if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"],
                                  betas=(0.9, cfg.get("adam_beta2", 0.999)),
                                  weight_decay=cfg.get("weight_decay", 0.0),
                                  fused=cfg.get("fused_adam", False) or None)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)
    if cfg.get("compile"):
        unet = torch.compile(unet, mode=cfg.get("compile_mode"))
    if cfg.get("compile_text"):
        text.model.compile()  # frozen CLIP text tower
    # The DataLoader is not passed through accelerator.prepare: on one GPU that would only
    # move batches, and leaving it alone keeps hf_data's WeightedRandomSampler untouched.
    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)
    if u_head is not None:
        u_head = accelerator.prepare(u_head)

    # Multi-EMA: one diffusers EMAModel per decay. use_ema_warmup=False gives
    # decay_t = min(decay, (1+t)/(10+t)).
    model = accelerator.unwrap_model(unet)
    val_sets = None
    if cfg.get("val_every"):
        val_sets = load_val_sets(cfg, text, device, cfg.get("val_images", 2048))
        val_model = accelerator.unwrap_model(unet, keep_torch_compile=False)  # same parameters, not compiled
    emas = {d: EMAModel(model.parameters(), decay=d, foreach=True) for d in cfg["ema_decays"]}
    for ema in emas.values():
        ema.to(device)
        accelerator.register_for_checkpointing(ema)

    step = 0
    if args.resume:
        accelerator.load_state(args.resume)
        for ema in emas.values():  # load_state_dict leaves the shadow params on the CPU
            ema.to(device)
        step = int(Path(args.resume).name.split("_")[-1])
        print(f"resumed from {args.resume} at step {step}")
    save_run_meta(out, cfg)
    log = open(out / "training_log.jsonl", "a")

    unet.train()
    trims = 0
    if trim_page_cache(cfg, out):  # before the loader's workers start: they stall too
        print("container memory was near its limit: dropped this run's files from the page cache", flush=True)
        trims = 1
    data_iter = iter(loader)
    loss_acc, n_acc, t_train, t_last = 0.0, 0, 0.0, time.perf_counter()
    while step < cfg["num_steps"]:
        try:
            x, captions = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, captions = next(data_iter)
        x = x.to(device, non_blocking=True)
        labels = None
        if cfg.get("label_cond"):  # each label dropped to "no label" independently of the caption
            captions, q, c = captions
            p = cfg.get("label_dropout", 0.1)
            labels = tuple(torch.where(torch.rand(len(t)) < p, 0, t) for t in (q, c))
        drop = torch.rand(len(captions)) < cfg.get("guidance_dropout", 0.1)
        captions = ["" if d else c for d, c in zip(drop.tolist(), captions)]
        if labels is not None and cfg.get("label_drop_with_caption"):  # train the all-null (CFG) branch
            labels = tuple(torch.where(drop, 0, t) for t in labels)

        with accelerator.autocast():
            ctx, mask, pooled = text(captions)
            pooled = add_labels(pooled, labels)
            sigma = (torch.randn(x.shape[0], device=device) * P_std + P_mean).exp()
            s4 = sigma[:, None, None, None]
            x_noisy = x + s4 * torch.randn_like(x)
            c_noise = edm.precondition_noise(sigma)
            F = unet(edm.precondition_inputs(x_noisy, s4), c_noise, encoder_hidden_states=ctx,
                     class_labels=pooled, encoder_attention_mask=mask).sample
            D = edm.precondition_outputs(x_noisy, F.float(), s4)
            weight = (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2           # EDM lambda(sigma)
            per_sample = (weight[:, None, None, None] * (D - x) ** 2).mean(dim=(1, 2, 3))
            if u_head is not None:
                u = u_head(c_noise)
                loss = (per_sample / u.exp() + u).mean()
            else:
                loss = per_sample.mean()

        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(params, cfg.get("grad_clip", 1.0))
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        ema_step(emas, model)
        step += 1
        loss_acc += loss.detach()
        n_acc += 1

        if step % cfg.get("log_every", 50) == 0 or step == cfg["num_steps"]:
            mean_loss = float(loss_acc) / n_acc  # syncs the device
            now = time.perf_counter()
            t_train += now - t_last
            sps = n_acc / (now - t_last)
            rec = {"step": step, "loss": mean_loss, "lr": lr_scheduler.get_last_lr()[0],
                   "grad_norm": float(grad_norm), "steps_per_sec": sps, "train_time_s": t_train}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"step {step:>7} loss {mean_loss:+.4f} lr {rec['lr']:.2e} "
                  f"gnorm {rec['grad_norm']:.2f} {sps:.2f} steps/s", flush=True)
            loss_acc, n_acc = 0.0, 0
            if not math.isfinite(mean_loss):
                raise RuntimeError(f"non-finite loss at step {step}")

            pauses, stop = [], None
            if val_sets is not None and step % cfg["val_every"] == 0:
                val = validate(val_model, edm, val_sets, sd, device)
                logits, bounds = attention_logits(val_model, edm, val_sets["val"][0], device)
                log.write(json.dumps({"step": step, "val": val, "attn_logit_max": logits}) + "\n")
                log.flush()
                pauses.append(f"val loss mean: val {val['val_mean']:.4f} train {val['train_mean']:.4f}")
                top = max((k for k in logits if k.endswith("attn1")), key=logits.get)
                pauses.append(f"largest self-attention logit {logits[top]:.1f} ({top})")
                if getattr(text, "misses", 0):
                    pauses.append(f"{text.misses} captions so far were not in the text cache (encoded live)")
                over = [k for k, b in bounds.items() if logits[k] > 1.01 * b]
                if over:  # cannot happen while the normalisation is applied
                    stop = f"attention logit {logits[over[0]]:.1f} in {over[0]} exceeds its qk_norm bound {bounds[over[0]]:.1f}"
                elif cfg.get("logit_stop") and logits[top] > cfg["logit_stop"]:
                    stop = f"self-attention logit {logits[top]:.1f} in {top} exceeds logit_stop {cfg['logit_stop']}"
            if step in cfg.get("export_steps", []):
                export(accelerator, unet, emas, edm, out / "exports" / f"step_{step:07d}")
                shutil.copy(out / "run_config.json", out / "exports" / f"step_{step:07d}" / "run_config.json")
                pauses.append(f"exported exports/step_{step:07d}")
                if cfg.get("benchmark_exports"):
                    if hasattr(text, "release"):  # a live LLM beside training leaves the subprocess no memory
                        text.release()
                    torch.cuda.empty_cache()
                    bench = run_benchmark(out, step, cfg)
                    log.write(json.dumps({"step": step, "bench": bench}) + "\n")
                    log.flush()
                    pauses.append("benchmark FID " + " ".join(f"{k} {v:.2f}" for k, v in bench.items()))
            if cfg.get("sample_every") and step % cfg["sample_every"] == 0:
                pauses.append(f"preview {preview(accelerator, unet, emas, text, cfg, out, step)}")
            if step % cfg.get("save_every", 10 ** 9) == 0 or step == cfg["num_steps"] or stop:
                if cfg.get("save_state", True):
                    accelerator.save_state(str(out / "checkpoints" / f"step_{step:07d}"))
                    if cfg.get("keep_checkpoints"):
                        prune_checkpoints(out / "checkpoints", cfg["keep_checkpoints"],
                                          cfg.get("keep_checkpoint_every", 0))
                export(accelerator, unet, emas, edm, out)
                pauses.append("exported")
            dropped = trim_page_cache(cfg, out)
            if dropped:
                log.write(json.dumps({"step": step, "page_cache_dropped_at_gb": round(dropped, 1)}) + "\n")
                if not trims:
                    pauses.append(f"container memory at {dropped:.0f} GB: dropped this run's files from the page "
                                  f"cache (repeats are only in training_log.jsonl)")
                trims += 1
            for p in pauses:
                print(f"  {p}", flush=True)
            if stop:
                raise RuntimeError(f"stopped at step {step}: {stop}")
            t_last = time.perf_counter()  # previews/saves are excluded from steps/s

    log.close()
    print(f"done: {step} steps, {t_train / 60:.1f} min training time, outputs in {out}")


if __name__ == "__main__":
    main()
