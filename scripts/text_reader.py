#!/usr/bin/env python3
"""Reads the string in a 32x32 image by template matching against the rendered_text generator's own fonts.

There is no OCR model that works at 32x32, but every string the benchmark asks for can be drawn by
scripts/gen_rendered_text.py, so reading is a closed-set decision: the image is matched against the target
string and a set of other strings (the other held-out strings and one-character edits of the target), each
drawn in every font, scale and line break the generator could have used. A sample counts as exact when the
target has the best match and that match is at least MIN_SCORE.

The match for one drawing of one string is the largest absolute normalised cross-correlation between its glyph
mask (with PAD background pixels around it, so that "HELP" does not match inside "HELP!") and the image, over
all positions and over four views of the image (R, G, B over mid grey, and alpha).

  python3 scripts/text_reader.py --validate 600     # reading accuracy on real renders (the ceiling of the metric)
"""

import argparse
import hashlib
import json
import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_rendered_text as G  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PAD = 2
MIN_SCORE = 0.8  # real renders: see --validate
TIE = 0.02
S = 32

_fonts = None


def fonts():
    global _fonts
    if _fonts is None:
        _fonts = G.load_fonts(G.DEFAULT_OUT / "_fonts")
    return _fonts


def drawings(text):
    """Every distinct glyph mask the generator could draw for `text` that fits in 32x32."""
    words = text.split()
    options = [[text]] + ([words] if len(words) > 1 else [])
    seen, out = set(), []
    for name, scale in G.FONT_VARIANTS:
        font = fonts()[name]
        for lines in options:
            if not all(font.can_render(t) for t in lines):
                continue
            multi = len(lines) > 1
            for extra in (0, 1):
                for gap in ((scale, 2 * scale) if multi else (scale,)):
                    for align in (("center", "left") if multi else ("center",)):
                        m = font.render(lines, scale, extra, gap, align)
                        if m is None or m.shape[0] > S or m.shape[1] > S:
                            continue
                        key = (m.shape, m.tobytes())
                        if key not in seen:
                            seen.add(key)
                            out.append(m)
    return out


@lru_cache(maxsize=4096)
def templates(text):
    """{(h, w) of the padded window: [k, h*w] zero-mean unit-norm templates}."""
    by_shape = {}
    for m in drawings(text):
        t = np.pad(m.astype(np.float32), PAD)
        t -= t.mean()
        n = np.linalg.norm(t)
        if n > 0:
            by_shape.setdefault(t.shape, []).append((t / n).ravel())
    return {s: np.stack(v) for s, v in by_shape.items()}


def views(img):
    """img: uint8 [32, 32, 4] straight alpha -> float [4, 32+2*PAD, 32+2*PAD]: R, G, B over mid grey, alpha."""
    a = img.astype(np.float32) / 255.0
    al = a[..., 3:]
    v = np.concatenate([a[..., :3] * al + 0.5 * (1 - al), al], -1).transpose(2, 0, 1)
    return np.pad(v, ((0, 0), (PAD, PAD), (PAD, PAD)), mode="edge")


class Windows:
    """Zero-mean unit-norm windows of one image, per window shape (shared by all candidate strings)."""

    def __init__(self, img):
        self.v = views(img)
        self.cache = {}

    def get(self, shape):
        if shape not in self.cache:
            w = np.lib.stride_tricks.sliding_window_view(self.v, shape, axis=(1, 2))
            w = w.reshape(-1, shape[0] * shape[1])
            w = w - w.mean(1, keepdims=True)
            n = np.linalg.norm(w, axis=1, keepdims=True)
            self.cache[shape] = np.where(n > 1e-3, w / np.maximum(n, 1e-3), 0.0)
        return self.cache[shape]


def score(win, text):
    best = 0.0
    for shape, t in templates(text).items():
        best = max(best, float(np.abs(win.get(shape) @ t.T).max()))
    return best


def _similar(ch):
    for group in (G.UPPER, G.LOWER, G.DIGITS, G.PUNCT):
        if ch in group:
            return group
    return ""


def edits(text):
    """One-character edits of `text`: every deletion and adjacent swap, three substitutions per position."""
    out = set()
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        if len(text) > 1:
            out.add(text[:i] + text[i + 1:])
        if i + 1 < len(text) and text[i + 1] != " ":
            out.add(text[:i] + text[i + 1] + ch + text[i + 2:])
        pool = [c for c in _similar(ch) if c != ch]
        rng = random.Random(hashlib.blake2b(f"{text}|{i}".encode(), digest_size=8).digest())
        for c in rng.sample(pool, min(3, len(pool))):
            out.add(text[:i] + c + text[i + 1:])
    out = {" ".join(t.split()) for t in out}
    return sorted(t for t in out if t and t != text and templates(t))


def candidates(text, others=G.HELDOUT):
    return [text] + [t for t in dict.fromkeys(list(others) + edits(text)) if t != text and templates(t)]


def read(img, text, others=G.HELDOUT):
    """(best string, its score, the target's score) for a uint8 [32, 32, 4] straight-alpha image."""
    win = Windows(img)
    scores = {t: score(win, t) for t in candidates(text, others)}
    # A string also matches inside any longer string that contains it (letter spacing can exceed PAD), so among
    # the matches within TIE of the best one the longest string, which explains the most ink, is the reading.
    top = max(scores.values())
    best = max((t for t in scores if scores[t] >= top - TIE), key=lambda t: (len(t.replace(" ", "")), scores[t]))
    return best, scores[best], scores[text]


def exact(img, text, others=G.HELDOUT):
    best, s, _ = read(img, text, others)
    return best == text and s >= MIN_SCORE


def _validate_one(job):
    path, text = job
    from PIL import Image
    img = np.asarray(Image.open(path).convert("RGBA"))
    best, s, st = read(img, text)
    return path, text, best, s, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, default=600, help="real renders to read")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    import csv
    from multiprocessing import Pool
    rows = list(csv.DictReader(open(G.DEFAULT_OUT / "_index.tsv"), delimiter="\t"))
    rows = random.Random(args.seed).sample(rows, args.validate)
    layout = {str(REPO / r["src"]): r["layout"] for r in rows}
    with Pool(args.workers) as pool:
        res = pool.map(_validate_one, [(str(REPO / r["src"]), r["text"]) for r in rows], chunksize=8)
    st = np.array([r[4] for r in res])
    ok = np.array([r[2] == r[1] for r in res])
    print(f"{len(res)} real renders: target is the best match for {ok.mean():.1%}; "
          f"best match is the target with score >= {MIN_SCORE}: {(ok & (st >= MIN_SCORE)).mean():.1%}")
    print("target score quantiles 1/5/25/50%:", np.round(np.quantile(st, [0.01, 0.05, 0.25, 0.5]), 3))
    per = {}
    for r in res:
        per.setdefault(layout[r[0]], []).append(r[2] == r[1] and r[4] >= MIN_SCORE)
    print("by layout:", {k: f"{np.mean(v):.0%} of {len(v)}" for k, v in sorted(per.items())})
    for r in [r for r in res if r[2] != r[1]][:20]:
        print(f"  misread {Path(r[0]).parent.name}/{Path(r[0]).name}: {r[1]!r} read as {r[2]!r} "
              f"({r[3]:.2f} vs {r[4]:.2f})")


if __name__ == "__main__":
    main()
