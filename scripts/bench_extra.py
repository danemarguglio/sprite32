#!/usr/bin/env python3
"""The text, seed and copy metrics of the benchmark (see BENCHMARK.md): three numbers FID does not give.

  text       exact-match accuracy on strings that are in no training image (data_gen/rendered_text/
             heldout_strings.txt), read by scripts/text_reader.py. Prompts: curation/text_prompts.jsonl.
  diversity  mean RMSE between samples of the same prompt with different seeds (curation/prompt_grid.txt),
             and the share of such pairs within the near-duplicate threshold.
  near-copy  share of samples that have a training image within the near-duplicate threshold, for captions of
             training images and for captions of holdout images. The same share for the real holdout images
             (curation/holdout_near.tsv) is the level expected without copying.

RMSE is over the 32x32x4 premultiplied training view on the 0-255 scale, as in scripts/cluster.py.

  python3 scripts/bench_extra.py prompts                    # write the text prompts (no GPU; done once, then frozen)
  python scripts/bench_extra.py run RUN_DIR --out DIR       # needs outputs/cluster/pixels.npy (cluster.py)
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

TEXT_PROMPTS = REPO / "curation" / "text_prompts.jsonl"
# layouts the reader gets right on at least 98% of real renders (text_reader.py --validate)
TEXT_LAYOUTS = ["plain", "solid", "button", "sign", "badge", "banner"]
NONSENSE = {"QZX", "K7P", "BVM2", "wjf", "Xq9", "x7", "LV9", "1UP"}
NEAR = 10.0


# ------------------------------------------------------------------------------------------- prompts

def string_kind(text):
    if " " in text:
        return "phrase"
    if text in NONSENSE:
        return "random"
    return "word" if any(c.isalpha() for c in text) else "number"


def write_prompts(args):
    import gen_rendered_text as G
    import text_reader
    fonts = text_reader.fonts()
    rng = random.Random(args.seed)
    rows = []
    for text in G.HELDOUT:
        layouts = ["plain"] + rng.sample(TEXT_LAYOUTS[1:], 2)
        for layout in layouts:
            rec = None
            for _ in range(20):
                rec = G.render_one(rng, layout, text, string_kind(text), fonts)
                if rec is not None:
                    break
            if rec is None:
                continue
            short, long_, _ = G.make_captions(rng, rec)
            side = G.sidecar(rec, short, long_, "", args.seed)
            rows.append({"text": text, "layout": layout, "short": short, "long": long_,
                         "quality": side["quality"], "category": side["category"]})
    with open(TEXT_PROMPTS, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(TEXT_PROMPTS.with_suffix(".txt"), "w") as fh:  # for scripts/build_text_cache.py --prompt_files
        for r in rows:
            fh.write(r["short"] + "\n" + r["long"] + "\n")
    print(f"{len(rows)} layouts of {len({r['text'] for r in rows})} strings -> {TEXT_PROMPTS}")


# ------------------------------------------------------------------------------------------- sampling

def to_uint8(x):
    """[-1, 1] premultiplied RGBA tensor [B, 4, 32, 32] -> uint8 [B, 32, 32, 4]."""
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def straight(pm):
    a = pm[..., 3:].astype(np.float32)
    rgb = np.where(a > 0, pm[..., :3].astype(np.float32) * 255.0 / np.maximum(a, 1), 0)
    return np.concatenate([np.clip(rgb, 0, 255), a], -1).round().astype(np.uint8)


class Sampler:
    def __init__(self, run_dir, weights, steps, guidance, batch):
        import torch
        from hf_common import build_scheduler, load_run_meta, pick_device
        from hf_sample import load_unet
        from hf_text import build_text_encoder
        self.cfg = load_run_meta(run_dir)
        self.device = pick_device()
        self.weights = weights or f"ema{max(self.cfg['ema_decays'])}"
        self.unet = load_unet(run_dir, self.weights, self.device)
        self.text = build_text_encoder(self.cfg, self.device)
        self.scheduler = build_scheduler(self.cfg, None)
        self.steps, self.guidance, self.batch, self.torch = steps, guidance, batch, torch

    def __call__(self, prompts, labels, seed):
        """prompts: list of str; labels: list of (quality, category) indices. One sample each -> uint8 [N,32,32,4]."""
        from hf_sample import sample_images
        torch, out = self.torch, []
        for i in range(0, len(prompts), self.batch):
            lab = None
            if self.cfg.get("label_cond"):
                chunk = labels[i:i + self.batch]
                lab = (torch.tensor([q for q, _ in chunk]), torch.tensor([c for _, c in chunk]))
            x = sample_images(self.unet, self.text, self.scheduler, prompts[i:i + self.batch], 1, self.steps,
                              self.guidance, None, seed + i, None, 0.0, self.cfg.get("channels", 4),
                              self.cfg.get("image_size", 32), lab)
            out.append(to_uint8(x))
        return np.concatenate(out)


def sheet(images, cols, path, scale=3):
    from PIL import Image
    n = len(images)
    rows = (n + cols - 1) // cols
    yy, xx = np.mgrid[0:32, 0:32]
    bg = np.where(((yy // 4 + xx // 4) % 2)[..., None] == 0, 200, 240).astype(np.float32)
    canvas = np.zeros((rows * 32, cols * 32, 3), np.uint8)
    for i, im in enumerate(images):
        a = im[..., 3:].astype(np.float32) / 255
        rgb = np.clip(im[..., :3].astype(np.float32) + (1 - a) * bg, 0, 255).astype(np.uint8)  # premultiplied over
        canvas[(i // cols) * 32:(i // cols) * 32 + 32, (i % cols) * 32:(i % cols) * 32 + 32] = rgb
    Image.fromarray(canvas).resize((cols * 32 * scale, rows * 32 * scale), Image.NEAREST).save(path)


# ------------------------------------------------------------------------------------------- the three numbers

def bench_text(sample, out, seeds):
    import text_reader
    from hf_common import CATEGORIES
    specs = [json.loads(line) for line in open(TEXT_PROMPTS)]
    res, images = [], []
    for key in ("short", "long"):
        prompts = [s[key] for s in specs]
        labels = [(s["quality"], CATEGORIES.index(s["category"]) + 1) for s in specs]
        for seed in range(seeds):
            ims = sample(prompts, labels, 1000 * seed)
            for s, im in zip(specs, ims):
                best, sc, st = text_reader.read(straight(im), s["text"])
                ok = best == s["text"] and sc >= text_reader.MIN_SCORE
                res.append({"text": s["text"], "layout": s["layout"], "caption": key, "seed": seed, "read": best,
                            "score": round(sc, 3), "target_score": round(st, 3), "exact": bool(ok)})
            if seed == 0:
                images.extend(ims)
    sheet(images, len(specs) // 2 or 1, out / "text.png")
    with open(out / "text.jsonl", "w") as fh:
        for r in res:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def acc(rows):
        return round(float(np.mean([r["exact"] for r in rows])), 4) if rows else None
    summary = {"text_exact": acc(res), "text_exact_short": acc([r for r in res if r["caption"] == "short"]),
               "text_exact_long": acc([r for r in res if r["caption"] == "long"]),
               "text_any_seed": round(float(np.mean([any(r["exact"] for r in res if r["text"] == t)
                                                     for t in {r["text"] for r in res}])), 4),
               "text_samples": len(res)}
    for lay in sorted({r["layout"] for r in res}):
        summary[f"text_exact_{lay}"] = acc([r for r in res if r["layout"] == lay])
    for n in (1, 2, 3, 4, 5):  # by string length (5 = five or more characters)
        rows = [r for r in res if min(len(r["text"]), 5) == n]
        if rows:
            summary[f"text_exact_len{n}"] = acc(rows)
    return summary


def rmse_matrix(a, b):
    a, b = a.reshape(len(a), -1).astype(np.float32), b.reshape(len(b), -1).astype(np.float32)
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T
    return np.sqrt(np.maximum(d2, 0) / a.shape[1])


def bench_diversity(sample, out, seeds):
    prompts = [p for p in (REPO / "curation" / "prompt_grid.txt").read_text().split("\n") if p.strip()]
    from hf_common import N_QUALITY
    runs = np.stack([sample(prompts, [(N_QUALITY, 0)] * len(prompts), 1000 * s) for s in range(seeds)])  # [S, P, ...]
    per, close = [], []
    iu = np.triu_indices(seeds, 1)
    for p in range(len(prompts)):
        d = rmse_matrix(runs[:, p], runs[:, p])[iu]
        per.append(d.mean())
        close.append((d <= NEAR).mean())
    sheet(list(runs.transpose(1, 0, 2, 3, 4).reshape(-1, 32, 32, 4)), seeds, out / "diversity.png")
    return {"seed_rmse": round(float(np.mean(per)), 2), "seed_rmse_min_prompt": round(float(np.min(per)), 2),
            "seed_pairs_near": round(float(np.mean(close)), 4), "seed_prompts": len(prompts), "seed_seeds": seeds}


def nearest_train(queries, train, torch, device, chunk=32768):
    """(rmse, index) of the nearest row of `train` (uint8 [N, 4096] memmap) for each query."""
    q = torch.from_numpy(queries.reshape(len(queries), -1)).to(device).float()
    q2 = (q * q).sum(1)
    best = torch.full((len(q),), float("inf"), device=device)
    arg = torch.zeros(len(q), dtype=torch.long, device=device)
    for i in range(0, len(train), chunk):
        t = torch.from_numpy(np.ascontiguousarray(train[i:i + chunk])).to(device).float()
        d2 = q2[:, None] + (t * t).sum(1)[None] - 2 * q @ t.T
        m, j = d2.min(1)
        upd = m < best
        best[upd], arg[upd] = m[upd], j[upd] + i
    return torch.sqrt(best.clamp(min=0) / q.shape[1]).cpu().numpy(), arg.cpu().numpy()


def bench_copies(sample, out, args):
    from hf_common import label_index
    torch = sample.torch
    cache = REPO / args.pixels
    paths = (cache / "paths.txt").read_text().split("\n")[:-1]
    pix = np.load(cache / "pixels.npy", mmap_mode="r")
    prefix = args.data_dir.rstrip("/") + "/"
    n_train = sum(p.startswith(prefix) for p in paths)
    assert all(p.startswith(prefix) for p in paths[:n_train]), "training images must come first in the pixel cache"
    train = pix[:n_train].reshape(n_train, -1)
    holdout = [p for p in (REPO / args.holdout).read_text().split("\n") if p.strip()]

    def captions(files, n, seed):
        rng = random.Random(seed)
        picks, caps, labels = [], [], []
        for idx in rng.sample(range(len(files)), len(files)):
            try:
                meta = json.load(open(REPO / f"{files[idx]}.json"))
            except (OSError, ValueError):
                continue
            key = ("description", "verbose_description")[len(picks) % 2]
            if isinstance(meta.get(key), str) and meta[key].strip():
                picks.append(idx)
                caps.append(meta[key].strip())
                labels.append(label_index(meta))
            if len(picks) == n:
                break
        return picks, caps, labels

    summary, worst = {}, []
    for name, files in (("train", paths[:n_train]), ("holdout", holdout)):
        picks, caps, labels = captions(files, args.copies, 0)
        ims = sample(caps, labels, 500000)
        d, j = nearest_train(ims, train, torch, sample.device)
        summary[f"copy_{name}_near"] = round(float((d <= NEAR).mean()), 4)
        summary[f"copy_{name}_near5"] = round(float((d <= NEAR / 2).mean()), 4)
        summary[f"copy_{name}_median_rmse"] = round(float(np.median(d)), 2)
        if name == "train":
            summary["copy_train_own_image"] = round(float(((j == np.array(picks)) & (d <= NEAR)).mean()), 4)
        order = np.argsort(d)[:48]
        for k in order:
            worst.extend([ims[k], np.asarray(pix[j[k]])])
        with open(out / f"copies_{name}.tsv", "w") as fh:
            fh.write("caption_image\tnearest_train\trmse\n")
            for k in np.argsort(d):
                fh.write(f"{files[picks[k]]}\t{paths[j[k]]}\t{d[k]:.2f}\n")
    sheet(worst, 16, out / "copies.png")  # pairs: sample, nearest training image; closest first
    near_file = REPO / "curation" / "holdout_near.tsv"
    if near_file.exists():
        summary["copy_real_holdout_near"] = round((sum(1 for _ in open(near_file)) - 1) / max(len(holdout), 1), 4)
    summary["copy_samples"] = args.copies
    return summary


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sample = Sampler(args.run_dir, args.weights, args.steps, args.guidance, args.batch)
    summary = {"run": str(args.run_dir), "weights": sample.weights, "steps": args.steps, "guidance": args.guidance}
    if "text" in args.only:
        summary.update(bench_text(sample, out, args.text_seeds))
    if "diversity" in args.only:
        summary.update(bench_diversity(sample, out, args.seeds))
    if "copies" in args.only:
        summary.update(bench_copies(sample, out, args))
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    name = args.name or Path(args.run_dir).name
    for k in ("text_exact", "text_exact_short", "text_exact_long", "seed_rmse", "seed_pairs_near", "copy_train_near",
              "copy_train_own_image", "copy_holdout_near", "copy_real_holdout_near"):
        if k in summary:
            print(f"{name} {k} {summary[k]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("run_dir")
    r.add_argument("--out", required=True)
    r.add_argument("--name", default=None)
    r.add_argument("--weights", default=None, help="raw | ema<decay> (default: longest EMA)")
    r.add_argument("--steps", type=int, default=35)
    r.add_argument("--guidance", type=float, default=3.0)
    r.add_argument("--batch", type=int, default=256)
    r.add_argument("--text_seeds", type=int, default=4)
    r.add_argument("--seeds", type=int, default=8, help="samples per prompt for the diversity number")
    r.add_argument("--copies", type=int, default=2000, help="samples per caption set for the near-copy number")
    r.add_argument("--pixels", default="outputs/cluster", help="pixels.npy + paths.txt from cluster.py")
    r.add_argument("--data_dir", default="data_v13")
    r.add_argument("--holdout", default="curation/holdout.txt")
    r.add_argument("--only", nargs="*", default=["text", "diversity", "copies"])
    args = ap.parse_args()
    write_prompts(args) if args.cmd == "prompts" else run(args)


if __name__ == "__main__":
    main()
