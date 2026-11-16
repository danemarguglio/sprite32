#!/usr/bin/env python3
"""List the flat black icons of the training tree that recolor.py may recolour (config recolor_list).

An image qualifies when, in its training view (the pixel cache of scripts/cluster.py), at least 12 pixels
have alpha >= 64, all of those are black (every channel <= 48 after un-premultiplying), at least a quarter of
the image is transparent, and at least one of its captions can be rewritten for one of its target colours
(recolor.swap_caption). rendered_text is skipped: its generator already varies the colour.

Writes curation/recolor.tsv (path relative to data_v13/, target colour names) and prints the counts.

Usage: python scripts/build_recolor_list.py [--pixels outputs/cluster] [--out curation/recolor.tsv]
"""
import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import recolor  # noqa: E402

KEYS = ["description", "verbose_description"]
SKIP = {"rendered_text"}


def captions(path):
    try:
        meta = json.load(open(f"{path}.json"))
    except (OSError, ValueError):
        return []
    return [meta[k].strip() for k in KEYS if isinstance(meta.get(k), str) and meta[k].strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data_v13")
    ap.add_argument("--pixels", default="outputs/cluster")
    ap.add_argument("--out", default="curation/recolor.tsv")
    args = ap.parse_args()
    paths = (Path(args.pixels) / "paths.txt").read_text().split("\n")[:-1]
    X = np.load(Path(args.pixels) / "pixels.npy", mmap_mode="r")
    prefix = f"{args.data}/"
    train = [i for i, p in enumerate(paths) if p.startswith(prefix)]
    assert train == list(range(len(train))), "training images come first in the pixel cache"

    black = np.zeros(len(train), bool)
    for i in range(0, len(train), 50000):
        a = np.asarray(X[i:min(i + 50000, len(train))]).astype(np.float32)
        al = a[..., 3]
        solid = al >= 64
        top = (a[..., :3].max(-1) * 255 / np.maximum(al, 1) * solid).max((1, 2))  # brightest channel, straight alpha
        black[i:i + len(a)] = (solid.sum((1, 2)) >= 12) & (top <= 48) & ((al < 128).mean((1, 2)) >= 0.25)
    cand = [i for i in np.flatnonzero(black) if paths[i][len(prefix):].split("/")[0] not in SKIP]
    print(f"{int(black.sum()):,} flat black images of {len(train):,}; {len(cand):,} outside {sorted(SKIP)}")

    with ThreadPoolExecutor(16) as pool:
        caps = list(pool.map(lambda i: captions(paths[i]), cand, chunksize=2000))
    rows, n_caps, n_var, per_src, per_colour = [], 0, set(), Counter(), Counter()
    for i, cs in zip(cand, caps):
        rel = paths[i][len(prefix):]
        targets = recolor.targets_for(rel)
        vs = [(t, v) for c in cs for t in targets for v in [recolor.swap_caption(c, t)] if v]
        if not vs:
            continue
        rows.append((rel, targets))
        n_caps += sum(1 for c in cs if recolor.variants(c, targets))
        n_var.update(v for _, v in vs)
        per_src[rel.split("/")[0]] += 1
        per_colour.update(targets)
    with open(args.out, "w") as fh:
        fh.write("path\ttargets\n")
        for rel, targets in sorted(rows):
            fh.write(f"{rel}\t{','.join(targets)}\n")
    print(f"{len(rows):,} images listed; {n_caps:,} of their captions name black and can be rewritten "
          f"({n_caps / max(1, sum(len(c) for c in caps)):.0%} of the candidates' captions); "
          f"{len(n_var):,} distinct rewritten captions for the text cache")
    print("by source:", ", ".join(f"{s} {n:,}" for s, n in per_src.most_common(12)))
    print("target colours:", ", ".join(f"{s} {n:,}" for s, n in sorted(per_colour.items())))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
