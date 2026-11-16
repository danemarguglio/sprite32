"""Guidance distillation: classifier-free guidance folded into the UNet, one pass per step.

Sampling a finished run costs two UNet passes per step: the caption and the empty caption, combined as
F_u + w * (F_c - F_u). The student is a copy of the run's UNet trained to give that combination from the caption
pass alone, at one fixed w, on noised training images. Its exports are sampled with guidance off: run_config.json
carries "distilled_guidance", hf_common.load_unet marks the UNet with it and the samplers in hf_sample.py then
skip the second pass, so hf_sample.py, hf_eval.py, scripts/bench_extra.py and scripts/benchmark.sh take the
export as is and their "guidance 3" rows mean the guidance in the weights.

The text adapter stays frozen, so the student's UNet body also works behind another text side made for the
same run (distill_text.py): --small_export gets a second export with that adapter and text encoder.

Written to --out:
  log.jsonl       "train" and "val" records; rel = error against the guided teacher / the same for the
                  unguided teacher, per sigma on held-out images (1 at the start, 0 = identical)
  export/         unet_raw, unet_ema<decay>, scheduler, run_config.json (teacher's text side)
  export_small/   the same UNet body with the text side of --small_export

  python distill_guidance.py outputs_diffusers/sprite32/exports/step_0500000 --out outputs_diffusers/distill/guide3
"""

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from hf_common import add_labels, build_scheduler, freeze_conv1_bias, load_run_meta
from hf_data import make_dataset, make_loader
from hf_sample import load_unet
from hf_text import build_text_encoder
from hf_train import load_val_sets

VAL_SIGMAS = [0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 60.0]
SERVING_LABELS = (5, 0)  # quality, category: what scripts/sprite_web.py samples with


def teacher_target(teacher, x_in, c_noise, ctx, mask, pooled, uctx, umask, upooled, w):
    """(guided F, caption-only F) of the frozen teacher, fp32."""
    with torch.no_grad(), torch.autocast(x_in.device.type, dtype=torch.bfloat16, enabled=x_in.device.type == "cuda"):
        F = teacher(torch.cat([x_in, x_in]), torch.cat([c_noise, c_noise]),
                    encoder_hidden_states=torch.cat([ctx, uctx]), class_labels=torch.cat([pooled, upooled]),
                    encoder_attention_mask=torch.cat([mask, umask])).sample.float()
    Fc, Fu = F.chunk(2)
    return Fu + w * (Fc - Fu), Fc


class Empty:
    """The empty caption with null labels for a batch of n: the teacher's unconditional branch."""

    def __init__(self, text, cfg):
        ctx, mask, pooled = text([""])
        labels = (torch.zeros(1, dtype=torch.long),) * 2 if cfg.get("label_cond") else None
        self.one = ctx, mask, add_labels(pooled, labels)

    def __call__(self, n):
        return tuple(t.expand(n, *t.shape[1:]) for t in self.one)


@torch.no_grad()
def validate(student, teacher, edm, batches, empty, w, device):
    out, gen = {}, torch.Generator(device=device).manual_seed(0)
    for sigma in VAL_SIGMAS:
        err = ref = 0.0
        for x, ctx, mask, pooled in batches:
            s4 = torch.full((x.shape[0], 1, 1, 1), sigma, device=device)
            x_in = edm.precondition_inputs(x + s4 * torch.randn(x.shape, generator=gen, device=device), s4)
            c_noise = edm.precondition_noise(s4.flatten())
            target, Fc = teacher_target(teacher, x_in, c_noise, ctx, mask, pooled, *empty(x.shape[0]), w)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                Fs = student(x_in, c_noise, encoder_hidden_states=ctx, class_labels=pooled,
                             encoder_attention_mask=mask).sample.float()
            err += float(((Fs - target) ** 2).sum())
            ref += float(((Fc - target) ** 2).sum())
        out[f"rel_{sigma:g}"] = err / ref
    out["rel"] = sum(out.values()) / len(VAL_SIGMAS)
    return out


def export(student, emas, names, teacher_dir, weights, cfg, out, w, small_dir):
    """export/ and, with small_dir, export_small/ (module docstring)."""
    sd = {k: v.detach().float().cpu().contiguous() for k, v in student.state_dict().items()}
    sets = {"raw": sd}
    for decay, shadow in emas.items():
        sets[f"ema{decay}"] = {**sd, **{n: s.detach().float().cpu().contiguous() for n, s in zip(names, shadow)}}
    extra = {"distilled_guidance": w, "guidance_distilled_from": str(teacher_dir)}
    targets = [(out / "export", teacher_dir, cfg, None)]
    if small_dir:
        adapter = {k: v for k, v in load_file(small_dir / f"unet_{weights}" / "diffusion_pytorch_model.safetensors").items()
                   if k.startswith("text_adapter.")}
        targets.append((out / "export_small", small_dir, json.load(open(small_dir / "run_config.json")), adapter))
    for ex, src, meta, adapter in targets:
        for name, s in sets.items():
            (ex / f"unet_{name}").mkdir(parents=True, exist_ok=True)
            save_file({**s, **(adapter or {})}, ex / f"unet_{name}" / "diffusion_pytorch_model.safetensors")
            shutil.copy(src / f"unet_{weights}" / "config.json", ex / f"unet_{name}" / "config.json")
        shutil.copytree(src / "scheduler", ex / "scheduler", dirs_exist_ok=True)
        json.dump({**meta, **extra}, open(ex / "run_config.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teacher", help="run or export dir of the finished run (run_config.json, unet_<weights>/)")
    ap.add_argument("--weights", default="ema0.9999")
    ap.add_argument("--out", required=True)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, nargs="*", default=[0.999, 0.9999])
    ap.add_argument("--empty_prob", type=float, default=0.03, help="share of empty captions")
    ap.add_argument("--serving_prob", type=float, default=0.4,
                    help="share of images given the web UI's labels (quality 5, no category) instead of their own")
    ap.add_argument("--schedule_prob", type=float, default=0.5,
                    help="share of sigmas drawn along the sampling schedule; the rest from the training lognormal")
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--val_images", type=int, default=1024)
    ap.add_argument("--export_every", type=int, default=5000)
    ap.add_argument("--small_export", default="outputs_diffusers/distill/qwen06/export",
                    help="distill_text.py export of the same run; skipped if it does not exist")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_dir, out, w = Path(args.teacher), Path(args.out), args.guidance
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_run_meta(teacher_dir)
    small_dir = Path(args.small_export) if args.small_export and Path(args.small_export).exists() else None

    # The loader keeps the run's own batch size and length: its exposure caps are counted over both.
    dataset = make_dataset(cfg)
    loader = make_loader({**cfg, "batch_size": args.batch,
                          "num_steps": cfg["num_steps"] * cfg["batch_size"] // args.batch}, dataset)
    text = build_text_encoder(cfg, device)
    empty = Empty(text, cfg)
    edm = build_scheduler(cfg, "euler")
    teacher = load_unet(teacher_dir, args.weights, device).requires_grad_(False)
    student = load_unet(teacher_dir, args.weights, device)  # eval mode throughout: no dropout against a fixed target
    student.text_adapter.requires_grad_(False)
    if cfg.get("freeze_conv1_bias"):
        freeze_conv1_bias(student)
    if small_dir:  # its adapter was made for the teacher's UNet body; anything else would not fit this student
        key = "conv_in.weight"
        other = load_file(small_dir / f"unet_{args.weights}" / "diffusion_pytorch_model.safetensors")[key]
        if not torch.equal(other, teacher.state_dict()[key].cpu()):
            raise SystemExit(f"{small_dir} was not made from {teacher_dir}")
    named = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
    names, params = [n for n, _ in named], [p for _, p in named]
    emas = {d: [p.detach().clone() for p in params] for d in args.ema}
    val = load_val_sets(cfg, text, device, args.val_images)["val"]
    print(f"{len(dataset)} images | student {sum(p.numel() for p in params) / 1e6:.1f}M trainable parameters | "
          f"guidance {w:g} | {len(val)} validation batches | small text side: {small_dir}", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, cfg.get("adam_beta2", 0.999)), weight_decay=0.0,
                            fused=device.type == "cuda" or None)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * (
        0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))
    run_student, run_teacher = student, teacher
    if args.compile:
        run_student, run_teacher = torch.compile(student), torch.compile(teacher)
    P_mean, P_std, rho = cfg["P_mean"], cfg.get("P_std", 1.2), cfg.get("rho", 7.0)
    lo, hi = cfg.get("sigma_min", 0.002) ** (1 / rho), cfg.get("sigma_max", 80.0) ** (1 / rho)

    log = open(out / "log.jsonl", "a")
    log.write(json.dumps({"kind": "val", "step": 0, **validate(student, teacher, edm, val, empty, w, device)}) + "\n")
    data_iter, t0, run = iter(loader), time.time(), 0.0
    for step in range(1, args.steps + 1):
        try:
            x, captions = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, captions = next(data_iter)
        x, B, labels = x.to(device, non_blocking=True), x.shape[0], None
        if cfg.get("label_cond"):
            captions, q, c = captions
            p = cfg.get("label_dropout", 0.1)
            serving = torch.rand(B) < args.serving_prob
            labels = tuple(torch.where(serving, s, torch.where(torch.rand(B) < p, 0, t))
                           for t, s in zip((q, c), SERVING_LABELS))
        captions = ["" if d else cap for d, cap in zip((torch.rand(B) < args.empty_prob).tolist(), captions)]
        ctx, mask, pooled = text(captions)
        pooled = add_labels(pooled, labels)
        u = torch.rand(B, device=device)
        sigma = torch.where(torch.rand(B, device=device) < args.schedule_prob, (hi + u * (lo - hi)) ** rho,
                            (torch.randn(B, device=device) * P_std + P_mean).exp())
        s4 = sigma[:, None, None, None]
        x_in = edm.precondition_inputs(x + s4 * torch.randn_like(x), s4)
        c_noise = edm.precondition_noise(sigma)
        target, _ = teacher_target(run_teacher, x_in, c_noise, ctx, mask, pooled, *empty(B), w)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            Fs = run_student(x_in, c_noise, encoder_hidden_states=ctx, class_labels=pooled,
                             encoder_attention_mask=mask).sample
        loss = ((Fs.float() - target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.get("grad_clip", 1.0))
        opt.step()
        sched.step()
        with torch.no_grad():
            for decay, shadow in emas.items():
                torch._foreach_lerp_(shadow, params, 1 - min(decay, (1 + step) / (10 + step)))
        run += float(loss.detach()) if step % 10 == 0 else 0.0
        if step % 100 == 0:
            rec = {"kind": "train", "step": step, "loss": run / 10, "lr": sched.get_last_lr()[0],
                   "it_s": 100 / (time.time() - t0)}
            if not math.isfinite(rec["loss"]):
                raise RuntimeError(f"non-finite loss at step {step}")
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % 500 == 0:
                print(f"step {step}/{args.steps} loss {rec['loss']:.5f} {rec['it_s']:.1f} it/s "
                      f"(~{(args.steps - step) / rec['it_s'] / 60:.0f} min left)", flush=True)
            run, t0 = 0.0, time.time()
        if step % args.val_every == 0 or step == args.steps:
            v = validate(student, teacher, edm, val, empty, w, device)
            log.write(json.dumps({"kind": "val", "step": step, **v}) + "\n")
            log.flush()
            print(f"  val step {step}: rel {v['rel']:.4f} (" +
                  " ".join(f"{s:g}: {v[f'rel_{s:g}']:.3f}" for s in VAL_SIGMAS) + ")", flush=True)
            t0 = time.time()
        if step % args.export_every == 0 or step == args.steps:
            export(student, emas, names, teacher_dir, args.weights, cfg, out, w, small_dir)
            t0 = time.time()
    print(f"done: {out / 'export'}" + (f" and {out / 'export_small'}" if small_dir else ""), flush=True)


if __name__ == "__main__":
    main()
