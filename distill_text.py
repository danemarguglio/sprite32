"""Text-encoder distillation: a small text encoder for a finished run, the UNet untouched.

The teacher is the run's frozen LLM (Qwen3-4B) followed by the run's TextAdapter; its output, the cross-attention
context (B, T + 1, 1024), is all the UNet ever sees of the caption. The student is a smaller causal LM with the
same tokenizer (Qwen3-0.6B), fine-tuned, followed by a copy of the adapter with a new input projection. It is
trained to give the same context for the same caption, token by token (masked MSE). The teacher's LLM states come
from the run's text cache, so neither the 4B model nor the UNet is loaded.

Written to --out:
  captions.json     the training captions (the text cache only stores their hashes)
  log.jsonl         "train" and "val" records; rel = MSE / variance of the teacher context (0 = identical)
  student_lm/       the student, loadable by hf_text.LLMEncoder (fp16)
  export/           a run dir like exports/step_XXXXXXX with the student's adapter in the UNet weights and
                    llm_model / text_adapter_in_dim / text_cache_dir changed in run_config.json, so
                    hf_sample.py, hf_eval.py, scripts/bench_extra.py and scripts/benchmark.sh take it as is

  python distill_text.py outputs_diffusers/sprite32/exports/step_0500000 --out outputs_diffusers/distill/qwen06
"""

import argparse
import json
import math
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from hf_common import TextAdapter
from hf_text import TextCache, caption_hash

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))


def collect_captions(cfg, holdout_list):
    """(training captions, holdout captions): what scripts/build_text_cache.py encoded for this config, with
    the captions of the benchmark holdout images kept apart for validation."""
    from build_text_cache import sidecar_captions
    from hf_data import scan_image_paths
    keys = cfg.get("caption_keys", ["description", "verbose_description"])
    data_dir = Path(cfg["data_dir"]).expanduser()
    paths = scan_image_paths(str(data_dir))
    held = [line.strip() for line in open(holdout_list) if line.strip() and not line.startswith("#")]
    with ThreadPoolExecutor(16) as pool:
        per_image = list(pool.map(lambda p: sidecar_captions(p, keys), list(paths) + held, chunksize=2000))
    caps = {c for cs in per_image[:len(paths)] for c in cs}
    if cfg.get("recolor_list") and cfg.get("recolor_prob", 0) > 0:
        import recolor
        listed, root = recolor.load_list(cfg["recolor_list"]), data_dir.resolve()
        for p, cs in zip(paths, per_image):
            targets = listed.get(Path(p).resolve().relative_to(root).as_posix())
            if targets:
                caps.update(v for c in cs for v in recolor.variants(c, targets))
    val = {c for cs in per_image[len(paths):] for c in cs}
    caps -= val
    caps.discard("")
    val.discard("")
    return sorted(caps), sorted(val)


def cached_only(cache, captions):
    """The captions the text cache holds (by hash; reading their states would read the whole cache)."""
    h = np.array([caption_hash(c) for c in captions], dtype=np.uint64)
    i = np.searchsorted(cache.hash, h).clip(max=len(cache.hash) - 1)
    return [c for c, ok in zip(captions, cache.hash[i] == h) if ok]


class Student(torch.nn.Module):
    """Causal LM (states of hidden_states[layer], prefix position dropped, as hf_text.LLMEncoder) + TextAdapter."""

    def __init__(self, name, adapter, layer, max_tokens, prefix):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.tokenizer.padding_side = "right"
        self.lm = AutoModel.from_pretrained(name, dtype=torch.float32)
        self.adapter = adapter
        self.layer, self.max_tokens, self.prefix = layer, max_tokens, prefix

    def states(self, captions, device):
        tok = self.tokenizer([self.prefix + c for c in captions], padding=True, truncation=True,
                             max_length=self.max_tokens, return_tensors="pt").to(device)
        h = self.lm(**tok, output_hidden_states=True).hidden_states[self.layer]
        return h[:, 1:], tok.attention_mask[:, 1:]

    def forward(self, captions, device):
        h, mask = self.states(captions, device)
        return self.adapter(h.float(), mask)


def teacher_context(adapter, cache, captions, T, device):
    """Teacher adapter output for cached captions, padded to T tokens: (B, T + 1, dim), mask (B, T + 1).
    "" is the empty caption (no tokens), which the cache does not hold."""
    toks = [torch.zeros(0, cache.meta["dim"], dtype=torch.float16) if not c else t
            for c, t in zip(captions, cache.get_many(captions))]
    ctx = torch.zeros(len(toks), T, cache.meta["dim"], dtype=torch.float16)
    mask = torch.zeros(len(toks), T, dtype=torch.long)
    for i, t in enumerate(toks):
        ctx[i, :len(t)] = t[:T]
        mask[i, :len(t)] = 1
    with torch.no_grad():
        return adapter(ctx.to(device).float(), mask.to(device))


def context_loss(out, mask_s, target, mask_t):
    """Mean squared error over the visible positions; the two masks differ only if the tokenizers do."""
    T = min(out.shape[1], target.shape[1])
    same = (mask_s[:, :T] == mask_t[:, :T]).all(1) & (mask_s.sum(1) == mask_t.sum(1))
    m = (mask_t[:, :T] * same[:, None]).float()
    err = ((out[:, :T].float() - target[:, :T]) ** 2).mean(-1)
    return (err * m).sum() / m.sum().clamp(min=1), int((~same).sum())


@torch.no_grad()
def validate(student, teacher, cache, captions, T, device, batch=256):
    student.eval()
    se, var, cos, n = 0.0, 0.0, 0.0, 0
    for i in range(0, len(captions), batch):
        caps = captions[i:i + batch]
        target, mask_t = teacher_context(teacher, cache, caps, T, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out, mask_s = student(caps, device)
        t = min(out.shape[1], target.shape[1])
        m = mask_t[:, :t].bool() & mask_s[:, :t].bool()
        a, b = out[:, :t].float()[m], target[:, :t][m]
        se += float(((a - b) ** 2).sum())
        var += float(((b - b.mean(0)) ** 2).sum())
        cos += float(F.cosine_similarity(a, b, dim=-1).sum())
        n += len(a)
    student.train()
    return {"rel": se / var, "cos": cos / n, "tokens": n}


def export(student, teacher_dir, weights, cfg, out):
    """student_lm/ and export/ (module docstring)."""
    lm_dir, ex = out / "student_lm", out / "export"
    student.lm.half().save_pretrained(lm_dir)
    student.lm.float()
    student.tokenizer.save_pretrained(lm_dir)
    (ex / f"unet_{weights}").mkdir(parents=True, exist_ok=True)
    sd = load_file(teacher_dir / f"unet_{weights}" / "diffusion_pytorch_model.safetensors")
    sd = {k: v for k, v in sd.items() if not k.startswith("text_adapter.")}
    sd.update({f"text_adapter.{k}": v.detach().float().cpu().contiguous() for k, v in student.adapter.state_dict().items()})
    save_file(sd, ex / f"unet_{weights}" / "diffusion_pytorch_model.safetensors")
    shutil.copy(teacher_dir / f"unet_{weights}" / "config.json", ex / f"unet_{weights}" / "config.json")
    shutil.copytree(teacher_dir / "scheduler", ex / "scheduler", dirs_exist_ok=True)
    n_lm = sum(p.numel() for p in student.lm.parameters())
    new = {**cfg, "llm_model": str(lm_dir), "text_adapter_in_dim": student.lm.config.hidden_size,
           "text_cache_dir": None, "distilled_from": str(teacher_dir), "teacher_llm": cfg["llm_model"]}
    new["param_count"] = {**cfg.get("param_count", {}), "unet": sum(v.numel() for v in sd.values()),
                          "student_llm": n_lm}
    json.dump(new, open(ex / "run_config.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teacher", help="run or export dir of the finished run (run_config.json, unet_<weights>/)")
    ap.add_argument("--weights", default="ema0.9999")
    ap.add_argument("--student", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr_lm", type=float, default=3e-5)
    ap.add_argument("--lr_adapter", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--empty_prob", type=float, default=0.02, help="share of empty captions (the unconditional context)")
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--val_captions", type=int, default=4000)
    ap.add_argument("--holdout_list", default="curation/holdout.txt")
    ap.add_argument("--max_captions", type=int, default=0, help="smoke test: train on this many captions")
    ap.add_argument("--no_checkpointing", action="store_true", help="faster, much more GPU memory")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_dir, out = Path(args.teacher), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(teacher_dir / "run_config.json"))
    cache = TextCache(Path(cfg["text_cache_dir"]).expanduser())
    layer, max_tokens = cfg.get("llm_layer", -2), cfg.get("llm_max_tokens", 64)
    prefix, T = cfg.get("llm_prefix", "<|im_start|>"), max_tokens - 1

    if (out / "captions.json").exists():
        caps, val = json.load(open(out / "captions.json"))
    else:
        t0 = time.time()
        caps, val = collect_captions(cfg, args.holdout_list)
        n0 = len(caps), len(val)
        caps, val = cached_only(cache, caps), cached_only(cache, val)
        json.dump([caps, val], open(out / "captions.json", "w"))
        print(f"{len(caps)} of {n0[0]} training captions and {len(val)} of {n0[1]} holdout captions are in the text "
              f"cache ({time.time() - t0:.0f}s)", flush=True)
    random.Random(0).shuffle(val)
    val = val[:args.val_captions]
    if args.max_captions:
        caps = caps[:args.max_captions]

    # teacher adapter: the text_adapter.* weights of the run's UNet
    sd = load_file(teacher_dir / f"unet_{args.weights}" / "diffusion_pytorch_model.safetensors")
    sd = {k[len("text_adapter."):]: v for k, v in sd.items() if k.startswith("text_adapter.")}
    dim, n_layers = cfg["cross_attention_dim"], cfg.get("text_adapter_layers", 2)
    teacher = TextAdapter(cfg["text_adapter_in_dim"], dim, n_layers, max_tokens=max_tokens)
    teacher.load_state_dict(sd, strict=True)
    teacher.to(device).eval().requires_grad_(False)

    # student: same adapter with a new input projection, statistics of the student's own states
    from transformers import AutoConfig, AutoTokenizer
    adapter = TextAdapter(AutoConfig.from_pretrained(args.student).hidden_size, dim, n_layers, max_tokens=max_tokens)
    adapter.load_state_dict({k: v for k, v in sd.items() if not k.startswith(("proj.", "feat_"))}, strict=False)
    student = Student(args.student, adapter, layer, max_tokens, prefix).to(device)
    if not args.no_checkpointing:  # without it a batch of 128 needs 21 GiB for the 0.6B model's activations
        student.lm.gradient_checkpointing_enable()
    sample = random.Random(1).sample(caps, min(2000, len(caps)))
    ref = AutoTokenizer.from_pretrained(cfg["llm_model"])
    a = ref([prefix + c for c in sample], truncation=True, max_length=max_tokens)["input_ids"]
    b = student.tokenizer([prefix + c for c in sample], truncation=True, max_length=max_tokens)["input_ids"]
    if a != b:
        sys.exit(f"{args.student} does not tokenize like {cfg['llm_model']}: token-by-token matching needs the same ids")
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        feats = []
        for i in range(0, len(sample), 250):
            h, m = student.states(sample[i:i + 250], device)
            feats.append(h[m.bool()].float())
        feats = torch.cat(feats)
    adapter.set_stats(feats.mean(0), feats.std(0))
    n_lm = sum(p.numel() for p in student.lm.parameters())
    print(f"student {args.student}: {n_lm / 1e6:.0f}M parameters, hidden {feats.shape[1]}; teacher LLM "
          f"{cache.meta.get('llm_params', 0) / 1e6:.0f}M; {len(caps)} captions, {len(val)} for validation", flush=True)

    opt = torch.optim.AdamW([{"params": student.lm.parameters(), "lr": args.lr_lm},
                             {"params": student.adapter.parameters(), "lr": args.lr_adapter}],
                            betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * (
        0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))
    rng = random.Random(args.seed)

    def batch_at(_):
        caps_b = ["" if rng.random() < args.empty_prob else rng.choice(caps) for _ in range(args.batch)]
        return caps_b, teacher_context(teacher, cache, caps_b, T, device)

    log = open(out / "log.jsonl", "a")
    log.write(json.dumps({"kind": "val", "step": 0, **validate(student, teacher, cache, val, T, device)}) + "\n")
    student.train()
    pool = ThreadPoolExecutor(1)  # the next batch's cache reads and teacher pass run during the student's step
    nxt, t0, run, skipped = pool.submit(batch_at, 0), time.time(), 0.0, 0
    for step in range(1, args.steps + 1):
        caps_b, (target, mask_t) = nxt.result()
        nxt = pool.submit(batch_at, step)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out_s, mask_s = student(caps_b, device)
        loss, bad = context_loss(out_s, mask_s, target, mask_t)
        skipped += bad
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
        run += float(loss.detach())
        if step % 100 == 0:
            rec = {"kind": "train", "step": step, "loss": run / 100, "lr_lm": sched.get_last_lr()[0],
                   "skipped": skipped, "it_s": 100 / (time.time() - t0)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % 500 == 0:
                print(f"step {step}/{args.steps} loss {rec['loss']:.4f} {rec['it_s']:.1f} it/s "
                      f"(~{(args.steps - step) / rec['it_s'] / 60:.0f} min left)", flush=True)
            run, t0 = 0.0, time.time()
        if step % args.val_every == 0 or step == args.steps:
            v = validate(student, teacher, cache, val, T, device)
            log.write(json.dumps({"kind": "val", "step": step, **v}) + "\n")
            log.flush()
            print(f"  val step {step}: rel {v['rel']:.4f} cos {v['cos']:.4f}", flush=True)
            t0 = time.time()
    export(student, teacher_dir, args.weights, cfg, out)
    print(f"done: {out / 'export'} (student {n_lm / 1e6:.0f}M parameters in {out / 'student_lm'})", flush=True)


if __name__ == "__main__":
    main()
