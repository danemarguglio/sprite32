#!/usr/bin/env python3
"""Near-duplicate clusters of the training tree (data_v13/), and near-duplicates of the benchmark holdout in it.

Every image is compared as the model sees it: premultiplied RGBA, LANCZOS resize of the shorter side to 32,
centre crop (hf_data.ImageDataset). Distance is RMSE over the 32x32x4 values (0-255), as
data_gen/acquire/prefilter/nn.py. Pixel-identical images are merged first; each remaining image is then
linked to its nearest neighbours (up to K) within --rmse, and the connected components are the clusters
(single linkage, so a chain of small steps, e.g. animation frames, ends up in one cluster; a cluster with more
than K members closer than --rmse to each other can in principle split, which would only under-count).

Writes:
  <out>/pixels.npy, paths.txt     the training views (cache; rebuilt when the image list changes)
  curation/clusters.tsv       path (relative to data_v13/) and cluster id, for images in a cluster of 2 or more;
                                  read by the training sampler (config cluster_file)
  curation/holdout_near.tsv   holdout images with a training image within --rmse (path, nearest, rmse)
  <out>/report.md, sheet_*.png    counts and contact sheets of sample clusters

Usage: python scripts/cluster.py [--data data_v13] [--rmse 10] [--device mps|cuda|cpu]
"""
import argparse
import csv
import hashlib
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

K = 32


def train_view(path):
    """32x32x4 uint8 premultiplied, what hf_data.ImageDataset feeds the model."""
    try:
        a = np.asarray(Image.open(path).convert("RGBA"), np.float32)
        a[..., :3] *= a[..., 3:4] / 255.0
        im = Image.fromarray(a.astype(np.uint8), "RGBa")  # as hf_data.premultiply_alpha
        w, h = im.size
        if (w, h) != (32, 32):
            nw, nh = (32, int(32 * h / w)) if w <= h else (int(32 * w / h), 32)
            im = im.resize((nw, nh), Image.LANCZOS)
            left, top = int(round((nw - 32) / 2.0)), int(round((nh - 32) / 2.0))
            im = im.crop((left, top, left + 32, top + 32))
        return np.asarray(im, np.uint8)
    except Exception:
        return None


def load_pixels(paths, out, workers):
    cache, listing = out / "pixels.npy", out / "paths.txt"
    if cache.exists() and listing.exists() and listing.read_text().split("\n")[:-1] == paths:
        return np.load(cache, mmap_mode="r")
    with ProcessPoolExecutor(workers) as ex:
        ims = list(ex.map(train_view, paths, chunksize=4000))
    bad = [p for p, a in zip(paths, ims) if a is None]
    if bad:
        sys.exit(f"{len(bad)} unreadable images, e.g. {bad[0]}")
    X = np.stack(ims)
    np.save(cache, X)
    listing.write_text("\n".join(paths) + "\n")
    return X


def search(X, maxd, dev, qb=2048, rb=65536):
    """idx, rmse (N x K; -1 / inf = none) of the nearest other rows within maxd."""
    n = len(X)
    idx = np.full((n, K), -1, np.int64)
    dist = np.full((n, K), np.inf, np.float32)

    def flat(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(dev).float().reshape(len(a), -1) / 255

    for i in range(0, n, qb):
        q = flat(X[i:i + qb])
        qn = (q * q).sum(1, keepdim=True)
        best_d = torch.full((len(q), K), float("inf"), device=dev)
        best_i = torch.full((len(q), K), -1, device=dev, dtype=torch.long)
        for j in range(0, n, rb):
            r = flat(X[j:j + rb])
            d2 = (qn + (r * r).sum(1)[None] - 2 * q @ r.T).clamp_(min=0)
            rows = torch.arange(len(q), device=dev)
            cols = rows + i - j
            ok = (cols >= 0) & (cols < len(r))
            d2[rows[ok], cols[ok]] = float("inf")
            d, k = torch.topk(d2, min(K, len(r)), dim=1, largest=False)
            cat_d, cat_i = torch.cat([best_d, d], 1), torch.cat([best_i, k + j], 1)
            best_d, sel = torch.topk(cat_d, K, dim=1, largest=False)
            best_i = torch.gather(cat_i, 1, sel)
        rmse = (best_d / 4096).sqrt() * 255
        best_i[rmse > maxd] = -1
        idx[i:i + qb] = best_i.cpu().numpy()
        dist[i:i + qb] = rmse.cpu().numpy()
        if (i // qb) % 25 == 0:
            print(f"  searched {i:,}/{n:,}", flush=True)
    dist[idx < 0] = np.inf
    return idx, dist


class Union:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def sheet(groups, X, path, scale=3, cols=24):
    """One row per cluster (first `cols` members), on a grey checkerboard."""
    cell = 32 * scale + 2
    im = Image.new("RGB", (cols * cell, len(groups) * cell), (170, 170, 170))
    for y, g in enumerate(groups):
        for x, i in enumerate(g[:cols]):
            a = np.asarray(X[i], np.float32)
            al = a[..., 3:4] / 255
            board = np.full((32, 32, 3), 200, np.float32)
            board[::2, ::2] = board[1::2, 1::2] = 150
            rgb = (a[..., :3] + board * (1 - al)).clip(0, 255).astype(np.uint8)  # premultiplied over
            im.paste(Image.fromarray(rgb).resize((32 * scale, 32 * scale), Image.NEAREST), (x * cell + 1, y * cell + 1))
    im.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data_v13")
    ap.add_argument("--holdout", default="curation/holdout.txt")
    ap.add_argument("--out", default="outputs/cluster")
    ap.add_argument("--clusters", default="curation/clusters.tsv")
    ap.add_argument("--holdout-near", default="curation/holdout_near.tsv")
    ap.add_argument("--rmse", type=float, default=10.0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else
                    "cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(Path(args.data) / "_index.tsv"), delimiter="\t"))
    train = [f"{args.data}/{r['dst']}" for r in rows]
    source = [r["source"] for r in rows]
    hold = sorted(l.strip() for l in open(args.holdout) if l.strip() and not l.startswith("#"))
    paths = train + hold
    nt = len(train)
    X = load_pixels(paths, out, args.workers)
    print(f"{nt:,} training images, {len(hold):,} holdout images")

    # pixel-identical images -> one representative
    first, rep = {}, np.empty(len(paths), np.int64)
    for i in range(len(paths)):
        rep[i] = first.setdefault(hashlib.blake2b(X[i].tobytes(), digest_size=16).digest(), i)
    uniq = np.flatnonzero(rep == np.arange(len(paths)))
    print(f"{len(uniq):,} distinct training views")
    idx, dist = search(X[uniq], args.rmse, args.device)

    # clusters among training images only (a holdout image must not join two training clusters)
    uf = Union(len(paths))
    for i in range(nt):
        if rep[i] != i and rep[i] < nt:
            uf.join(i, rep[i])
    is_train = uniq < nt
    for a in np.flatnonzero(is_train):
        for b in idx[a]:
            if b >= 0 and is_train[b]:
                uf.join(uniq[a], uniq[b])
    # a training image whose representative is a holdout image: link it to the other training copies
    by_rep = defaultdict(list)
    for i in range(nt):
        if rep[i] >= nt:
            by_rep[rep[i]].append(i)
    for g in by_rep.values():
        for i in g[1:]:
            uf.join(g[0], i)
    root = np.array([uf.find(i) for i in range(nt)])
    size = Counter(root.tolist())
    members = defaultdict(list)
    for i, r in enumerate(root.tolist()):
        if size[r] > 1:
            members[r].append(i)
    with open(args.clusters, "w") as fh:
        fh.write("path\tcluster\n")
        for r in sorted(members):
            for i in members[r]:
                fh.write(f"{rows[i]['dst']}\t{r}\n")

    # holdout images with a training image within the threshold
    pos = {u: k for k, u in enumerate(uniq.tolist())}
    near = []
    for h in range(nt, len(paths)):
        if rep[h] < nt:  # pixel-identical training image
            near.append((paths[h], train[rep[h]], 0.0))
            continue
        k = pos[rep[h]]
        cand = [(float(d), int(uniq[b])) for b, d in zip(idx[k], dist[k]) if b >= 0 and uniq[b] < nt]
        twins = by_rep.get(rep[h])
        if twins:
            near.append((paths[h], train[twins[0]], 0.0))
        elif cand:
            d, b = min(cand)
            near.append((paths[h], train[b], d))
    with open(args.holdout_near, "w") as fh:
        fh.write("holdout\tnearest_train\trmse\n")
        for h, t, d in near:
            fh.write(f"{h}\t{t}\t{d:.2f}\n")

    # ------------------------------------------------------------ report
    clustered = sum(len(m) for m in members.values())
    sizes = sorted((len(m) for m in members.values()), reverse=True)
    cross = sum(1 for m in members.values() if len({source[i] for i in m}) > 1)
    per_src, per_src_n = Counter(source), Counter(source[i] for m in members.values() for i in m)
    bins = [(2, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, 10 ** 9)]
    L = [f"# Near-duplicate clusters in {args.data}/ at RMSE <= {args.rmse:g}", "",
         f"- {nt:,} training images; {clustered:,} ({clustered / nt:.1%}) are in {len(members):,} clusters of 2 or more.",
         f"- Counting each cluster as one design leaves {nt - clustered + len(members):,} designs.",
         f"- {cross:,} clusters span more than one source.",
         f"- Largest clusters: {', '.join(str(s) for s in sizes[:12])}.",
         f"- Holdout: {len(near):,} of {len(hold):,} holdout images have a training image within RMSE {args.rmse:g}"
         f" ({sum(1 for n in near if n[2] == 0):,} pixel-identical). Listed in `{args.holdout_near}`.", "",
         "| cluster size | clusters | images |", "|---|---:|---:|"]
    for lo, hi in bins:
        sel = [s for s in sizes if lo <= s <= hi]
        L.append(f"| {lo}{'' if lo == hi else '+' if hi > 10 ** 6 else f'-{hi}'} | {len(sel):,} | {sum(sel):,} |")
    L += ["", "| source | images | in a cluster | share |", "|---|---:|---:|---:|"]
    for s, n in sorted(per_src.items(), key=lambda kv: -per_src_n[kv[0]])[:40]:
        L.append(f"| {s} | {n:,} | {per_src_n[s]:,} | {per_src_n[s] / n:.0%} |")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:22]))

    rng = random.Random(0)
    groups = sorted(members.values(), key=len, reverse=True)
    sheet(groups[:40], X, out / "sheet_largest.png")
    for name, lo, hi in (("pairs", 2, 2), ("small", 3, 9), ("medium", 10, 49)):
        sel = [g for g in groups if lo <= len(g) <= hi]
        sheet(rng.sample(sel, min(40, len(sel))), X, out / f"sheet_{name}.png", cols=max(2, min(24, hi)))
    hsel = rng.sample(near, min(40, len(near)))
    ix = {p: i for i, p in enumerate(paths)}
    sheet([[ix[h], ix[t]] for h, t, _ in hsel], X, out / "sheet_holdout_near.png", cols=2)
    print(f"-> {args.clusters}, {args.holdout_near}, {out}/report.md")


if __name__ == "__main__":
    main()
