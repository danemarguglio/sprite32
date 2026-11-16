#!/usr/bin/env python3
"""Encode every caption of a config's dataset once with its frozen LLM (hf_text.LLMEncoder) and store the token
states for training (hf_text.TextCache).

  python scripts/build_text_cache.py configs/sprite32.yaml                 # all sidecars under data_dir (~90 GB)
  python scripts/build_text_cache.py configs/sprite32.yaml --train_subset --set max_images=20000 \
      text_cache_dir=text_cache/smoke                                    # only what that config trains on

Captions: the config's caption_keys of every image under data_dir (training and val split alike), of the images
in --image_lists (the benchmark holdout, which data_v13/ does not contain, so hf_eval hits the cache too), every
rewritten caption the config's recolor_list can produce (recolor.py), its sample_prompts, and --prompt_files
(one prompt per line).
Written: tokens.f16 (n_tokens x dim, fp16), index.npz (hash, start, len per caption), stats.npz (per-dim mean/std
over all tokens, for the TextAdapter), and meta.json last, so an interrupted build is not mistaken for a cache.
"""
import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hf_common import load_config, pick_device  # noqa: E402
from hf_data import scan_image_paths, train_paths  # noqa: E402
from hf_text import LLMEncoder, caption_hash  # noqa: E402


def sidecar_captions(path, keys):
    try:
        meta = json.load(open(f"{path}.json"))
    except (OSError, ValueError):
        return []
    return [meta[k].strip() for k in keys if isinstance(meta.get(k), str) and meta[k].strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (YAML values)")
    ap.add_argument("--train_subset", action="store_true", help="only the images the config trains on (smoke tests)")
    ap.add_argument("--prompt_files", nargs="*", default=["curation/prompt_grid.txt", "curation/text_prompts.txt"])
    ap.add_argument("--image_lists", nargs="*", default=["curation/holdout.txt"],
                    help="lists of image paths whose sidecar captions are added (skipped with --train_subset)")
    ap.add_argument("--max_batch_tokens", type=int, default=16000, help="padded tokens per LLM batch")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out = Path(cfg["text_cache_dir"]).expanduser()
    if (out / "meta.json").exists():
        sys.exit(f"{out} already holds a finished cache; move it away to rebuild")
    out.mkdir(parents=True, exist_ok=True)

    keys = cfg.get("caption_keys", ["description", "verbose_description"])
    data_dir = Path(cfg["data_dir"]).expanduser()
    paths = train_paths(cfg) if args.train_subset else scan_image_paths(str(data_dir))
    extra = [] if args.train_subset else [line.strip() for f in args.image_lists if Path(f).exists()
                                          for line in open(f) if line.strip() and not line.startswith("#")]
    t0 = time.time()
    with ThreadPoolExecutor(16) as pool:
        per_image = list(pool.map(lambda p: sidecar_captions(p, keys), list(paths) + extra, chunksize=2000))
    caps = {c for cs in per_image for c in cs}
    if cfg.get("recolor_list") and cfg.get("recolor_prob", 0) > 0:
        import recolor
        listed, n0 = recolor.load_list(cfg["recolor_list"]), len(caps)
        root = data_dir.resolve()
        for p, cs in zip(paths, per_image):
            targets = listed.get(Path(p).resolve().relative_to(root).as_posix())
            if targets:
                caps.update(v for c in cs for v in recolor.variants(c, targets))
        print(f"{len(caps) - n0} rewritten captions from {cfg['recolor_list']}", flush=True)
    caps.update(p.strip() for p in cfg.get("sample_prompts", []))
    for f in args.prompt_files:
        if Path(f).exists():
            caps.update(line.strip() for line in open(f))
    caps.discard("")
    caps = sorted(caps)
    print(f"{len(caps)} distinct captions from {len(paths)} + {len(extra)} images ({time.time() - t0:.0f}s)", flush=True)

    spec = {"model": cfg["llm_model"], "layer": cfg.get("llm_layer", -2),
            "max_tokens": cfg.get("llm_max_tokens", 64), "prefix": cfg.get("llm_prefix", "<|im_start|>")}
    enc = LLMEncoder(spec["model"], pick_device(), spec["max_tokens"], spec["layer"], spec["prefix"])
    lens = np.array([n for i in range(0, len(caps), 20000) for n in enc.lengths(caps[i:i + 20000])], dtype=np.int64)
    n_tokens, dim = int(lens.sum()), enc.dim
    need = n_tokens * dim * 2
    free = shutil.disk_usage(out).free
    print(f"{n_tokens} tokens x {dim} dims fp16 = {need / 2**30:.1f} GB (free on that disk: {free / 2**30:.0f} GB); "
          f"tokens per caption: median {int(np.median(lens))}, max {int(lens.max())}", flush=True)
    if need > free - 5 * 2**30:
        sys.exit("not enough free disk for the cache")

    order = np.argsort(-lens, kind="stable")  # longest first: similar lengths share a batch, OOM shows at once
    start = np.zeros(len(caps), dtype=np.int64)
    s1 = torch.zeros(dim, dtype=torch.float64, device=enc.device)
    s2 = torch.zeros(dim, dtype=torch.float64, device=enc.device)
    pos, i, t0, last = 0, 0, time.time(), time.time()
    with open(out / "tokens.f16", "wb") as fh:
        while i < len(order):
            n = int(min(512, max(1, args.max_batch_tokens // (lens[order[i]] + 1)), len(order) - i))
            idx = order[i:i + n]
            h, mask = enc([caps[j] for j in idx])
            got = mask.sum(1).cpu().numpy()
            assert (got == lens[idx]).all(), "tokenizer lengths changed between the two passes"
            flat = h[mask].float()  # row-major: caption by caption, tokens in order
            s1 += flat.sum(0, dtype=torch.float64)
            s2 += flat.pow(2).sum(0, dtype=torch.float64)
            half = flat.to(torch.float16)
            assert torch.isfinite(half).all(), "token state outside the fp16 range"
            fh.write(half.cpu().numpy().tobytes())
            start[idx] = pos + np.concatenate([[0], np.cumsum(got)[:-1]])
            pos += int(got.sum())
            i += n
            if time.time() - last > 30:
                last = time.time()
                rate = pos / (last - t0)
                print(f"  {i}/{len(caps)} captions, {pos / n_tokens:.1%} of tokens, {rate:.0f} tokens/s, "
                      f"~{(n_tokens - pos) / rate / 60:.0f} min left", flush=True)
    assert pos == n_tokens
    mean = s1 / n_tokens
    std = (s2 / n_tokens - mean ** 2).clamp(min=0).sqrt()
    np.savez(out / "stats.npz", mean=mean.float().cpu().numpy(), std=std.float().cpu().numpy())
    hashes = np.array([caption_hash(c) for c in caps], dtype=np.uint64)
    assert len(np.unique(hashes)) == len(caps), "caption hash collision"
    np.savez(out / "index.npz", hash=hashes, start=start, len=lens.astype(np.int32))
    meta = {**spec, "dim": dim, "n_tokens": n_tokens, "n_captions": len(caps), "data_dir": str(cfg["data_dir"]),
            "caption_keys": keys, "llm_params": sum(p.numel() for p in enc.model.parameters()),
            "built": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    print(f"done in {(time.time() - t0) / 60:.1f} min: {out} ({pos} tokens, {len(caps)} captions)")


if __name__ == "__main__":
    main()
