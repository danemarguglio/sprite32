#!/usr/bin/env python3
"""Prompt adherence: mean CLIP similarity between generated images and their prompts.

Reads directories written by `hf_eval.py --keep_dir` (PNGs + prompts.jsonl) and prints
100 * mean cosine(image, prompt). Uses ViT-L/14, not the ViT-B/32 text tower the
models condition on, so no model is scored by its own encoder. Prompts longer than
77 tokens are truncated, as in training. Compare scores only across runs that used
the same prompts.

Usage:
  python scripts/clip_prompt_score.py outputs/benchmark/runs/full_long_g3.0 outputs/benchmark/runs/full_short_g3.0
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    from transformers import CLIPModel, CLIPProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    proc = CLIPProcessor.from_pretrained(args.model)

    for d in map(Path, args.dirs):
        rows = [json.loads(l) for l in open(d / "prompts.jsonl")]
        sims = []
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            imgs = [Image.open(d / r["file"]).convert("RGB").resize((224, 224), Image.BICUBIC)
                    for r in chunk]
            inp = proc(text=[r["prompt"] for r in chunk], images=imgs, return_tensors="pt",
                       padding=True, truncation=True, max_length=77).to(device)
            with torch.no_grad():
                out = model(**inp)
            img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            sims.append((img * txt).sum(-1).cpu())
        s = torch.cat(sims)
        print(f"{d.name:28s} clip_score {100 * s.mean().item():.2f}  (n={len(s)})")


if __name__ == "__main__":
    main()
