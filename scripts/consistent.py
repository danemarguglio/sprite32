"""Sets of sprites that differ in one thing only: the same sign with a different letter on each, the same
frame around different items. The template has one {} that is filled per sprite.

    python scripts/consistent.py RUN_DIR --template "yellow warning triangle with a black {}" --word SPRITE32 --out out/
    python scripts/consistent.py RUN_DIR --template "pixel art {} in a round blue badge" --items sword shield potion --out out/

Methods (--methods, one row of sheet.png each; append :value for the parameter):
  pick          the default. `trunk` for --tries seeds; keep the seeds where every sprite fits its own prompt
                better than any other (see prompt_errors; for --word the others are all letters and digits),
                drop those whose sprites differ in over 1.25x the fewest pixels, take the clearest of the rest
  lock          `pick`, then every pixel that barely differed between the sprites is replaced by its median over
                the set, so the sprites are identical outside the part that changes
  trunk:S       same starting noise, and one shared image down to noise level S: the mean of the per-prompt
                predictions drives every sample. Below S each prompt goes its own way. 5 is a good level for
                letters; lower blends the letters into each other, higher lets the outlines drift apart.
  seed          the same starting noise for every prompt, nothing else
  independent   a different seed per prompt (what you get without trying)
  edit:S        draw the first prompt, add noise of level S to it, redraw with each prompt

strip.png is the result of the last method as one transparent strip.
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hf_common import add_labels, build_scheduler, load_run_meta, null_labels, pick_device
from hf_sample import karras_sigmas, load_unet
from hf_text import build_text_encoder


def fill(template, item):
    """Single characters become "letter X" / "number 3"; anything longer is used as it is."""
    if len(item) == 1 and item.isalnum():
        item = ("number " if item.isdigit() else "letter ") + item
    return template.replace("{}", item)


class Sampler:
    def __init__(self, run_dir, device="auto", weights="ema0.9999", steps=35, guidance=3.0, quality=5):
        self.cfg = load_run_meta(run_dir)
        self.device = pick_device(device)
        self.unet = load_unet(run_dir, weights, self.device)
        self.text = build_text_encoder(self.cfg, self.device)
        self.edm = build_scheduler(self.cfg, "euler")
        self.guidance = 1.0 if getattr(self.unet, "distilled_guidance", None) else guidance
        self.quality = quality
        self.sigmas = karras_sigmas(steps, device=self.device)
        self.shape = (self.cfg.get("channels", 4), self.cfg.get("image_size", 32), self.cfg.get("image_size", 32))

    def noise(self, seed, n=1):
        return torch.randn(n, *self.shape, generator=torch.Generator("cpu").manual_seed(seed)).to(self.device)

    def denoiser(self, prompts):
        B = len(prompts)
        labels = null_labels(self.cfg, B, self.quality, 0)
        ctx, mask, pooled = self.text(prompts)
        pooled = add_labels(pooled, labels)
        uctx, umask, upooled = self.text([""] * B)
        upooled = add_labels(upooled, None if labels is None else tuple(torch.zeros_like(t) for t in labels))

        def denoise(x, sigma):
            s4 = sigma.expand(B)[:, None, None, None]
            x_in = self.edm.precondition_inputs(x, s4)
            c_noise = self.edm.precondition_noise(sigma.expand(B))
            out = self.unet(x_in, c_noise, encoder_hidden_states=ctx, class_labels=pooled,
                            encoder_attention_mask=mask).sample
            if self.guidance != 1.0:
                unc = self.unet(x_in, c_noise, encoder_hidden_states=uctx, class_labels=upooled,
                                encoder_attention_mask=umask).sample
                out = unc + self.guidance * (out - unc)
            return self.edm.precondition_outputs(x, out.float(), s4)
        return denoise

    @torch.no_grad()
    def sample(self, prompts, noise, hook=None, start_sigma=None, init=None):
        """DPM-Solver++(2M) on the denoised prediction. hook(denoised, sigma) may change the prediction.
        start_sigma + init: start from init + start_sigma * noise instead of pure noise."""
        denoise = self.denoiser(prompts)
        sig = self.sigmas
        i0 = 0 if start_sigma is None else int(torch.argmin((sig[:-1] - start_sigma).abs()))
        x = noise * sig[i0] if init is None else init + noise * sig[i0]
        old, h_last = None, None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            for i in range(i0, len(sig) - 1):
                s, sn = sig[i], sig[i + 1]
                d = denoise(x, s).float()
                if hook is not None:
                    d = hook(d, float(s))
                if sn == 0:
                    x = d
                    break
                h = math.log(float(s) / float(sn))
                dd = d if old is None else (1 + h_last / (2 * h)) * d - (h_last / (2 * h)) * old
                x = (sn / s) * x - math.expm1(-h) * dd
                old, h_last = d, h
        return x.clamp(-1, 1)


@torch.no_grad()
def prompt_errors(S, x, prompts, mask, sigmas=(1.5, 3.0), draws=2, seed=0, batch=256):
    """Which prompt does each finished sprite fit best? Noise the sprite, denoise it under every prompt
    (no guidance) and measure the error inside the mask. Returns (len(x), len(prompts)); lower = better fit."""
    P = len(prompts)
    ctx1, cmask1, pooled1 = S.text(prompts)
    gen = torch.Generator("cpu").manual_seed(seed)
    out = []
    for x0 in x.split(max(1, batch // P)):
        N = len(x0)
        ctx, cmask = ctx1.repeat(N, 1, 1), cmask1.repeat(N, 1)
        pooled = add_labels(pooled1.repeat(N, 1), null_labels(S.cfg, N * P, S.quality, 0))
        xx = x0.repeat_interleave(P, 0)
        err = torch.zeros(N * P, device=x.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=S.device.type == "cuda"):
            for sigma in sigmas:
                for _ in range(draws):
                    xn = xx + sigma * torch.randn(N, *S.shape, generator=gen).to(x.device).repeat_interleave(P, 0)
                    s = torch.tensor(sigma, device=x.device)
                    s4 = s.expand(N * P)[:, None, None, None]
                    f = S.unet(S.edm.precondition_inputs(xn, s4), S.edm.precondition_noise(s.expand(N * P)),
                               encoder_hidden_states=ctx, class_labels=pooled, encoder_attention_mask=cmask).sample
                    d = S.edm.precondition_outputs(xn, f.float(), s4)
                    err += (((d - xx) * mask) ** 2).flatten(1).sum(1)
        out.append((err / (len(sigmas) * draws * mask.sum().clamp(min=1))).view(N, P))
    return torch.cat(out)


def change_mask(x, thr=0.12, grow=1):
    """Pixels that differ between the samples of a set: std over the set, max over channels."""
    m = (x.std(0).amax(0) > thr).float()[None, None]
    for _ in range(grow):
        m = F.max_pool2d(m, 3, 1, 1)
    return m[0]  # (1, H, W)


def to_rgb(x, bg=(255, 255, 255)):
    """Premultiplied RGBA in [-1, 1] -> uint8 RGB over bg."""
    x = (x.float().cpu() + 1) / 2
    rgb, a = x[:3], x[3:4]
    out = rgb + (1 - a) * torch.tensor(bg).view(3, 1, 1) / 255
    return (out.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()


def sheet(rows, path, zoom=4, label_w=150):
    """rows: [(label, tensor (N, 4, H, W))]"""
    n = max(len(r[1]) for r in rows)
    H = rows[0][1].shape[-1]
    cell = H * zoom + 4
    im = Image.new("RGB", (label_w + n * cell, len(rows) * cell), (40, 40, 46))
    d = ImageDraw.Draw(im)
    for r, (label, xs) in enumerate(rows):
        d.text((6, r * cell + cell // 2 - 6), label, fill=(230, 230, 235))
        for c, x in enumerate(xs):
            tile = Image.fromarray(to_rgb(x)).resize((H * zoom, H * zoom), Image.NEAREST)
            im.paste(tile, (label_w + c * cell + 2, r * cell + 2))
    im.save(path)


def strip_image(xs, zoom=1):
    """The sprites side by side, straight alpha, transparent background."""
    x = ((xs.float().cpu() + 1) / 2).clamp(0, 1)
    rgb, a = x[:, :3], x[:, 3:4]
    rgba = torch.cat([(rgb / a.clamp(min=1e-3)).clamp(0, 1), a], 1)
    rgba = (rgba * 255).round().byte().permute(2, 0, 3, 1).flatten(1, 2).numpy()  # (H, N*W, 4)
    im = Image.fromarray(rgba)
    return im.resize((im.width * zoom, im.height * zoom), Image.NEAREST)


def strip(xs, path, zoom=1):
    strip_image(xs, zoom).save(path)


def margins(S, x, prompts, mask, others=()):
    """Per sprite: error under the best other prompt minus error under its own. Above 0 = fits its own best.
    others: more prompts that no sprite should fit."""
    n = len(prompts)
    err = prompt_errors(S, x, list(prompts) + list(others), mask).view(-1, n, n + len(others))
    own = err[:, torch.arange(n), torch.arange(n)]
    err[:, torch.arange(n), torch.arange(n)] = float("inf")
    for i, p in enumerate(prompts):  # a repeated letter is not a wrong answer
        for j, q in enumerate(prompts):
            if p == q:
                err[:, i, j] = float("inf")
    return err.amin(2) - own  # (len(x) / n, n)


def item_prompts(template, items, letters):
    """(prompts, others): one prompt per item, and for single characters the prompts of every other letter
    and digit, which no sprite of the set should fit better than its own."""
    prompts = [fill(template, c) for c in items]
    others = []
    if letters:
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        if any(c.islower() for c in items):
            alphabet = alphabet.lower()
        others = [p for p in (fill(template, c) for c in alphabet) if p not in prompts]
    return prompts, others


def trunk(S, prompts, seed, level):
    """Same noise for every prompt; above noise level `level` the mean prediction drives all of them."""
    hook = lambda d, s: d.mean(0, keepdim=True).expand_as(d) if s > level else d
    return S.sample(prompts, S.noise(seed).expand(len(prompts), -1, -1, -1), hook)


def lock(x, mask):
    """Outside the mask every sprite becomes the median of the set."""
    return mask * x + (1 - mask) * x.median(0, keepdim=True).values


def tried_sets(S, prompts, others=(), seed=1, tries=8, share=5.0, mask_thr=0.4, progress=None):
    """trunk for `tries` seeds. One dict per seed: seed, set, mask, fit (sprites that fit their own prompt
    best), margin (the weakest sprite's), px (pixels that differ over the set), picked."""
    sets = []
    for k, sd in enumerate(range(seed, seed + tries)):
        x = trunk(S, prompts, sd, share)
        mask = change_mask(x, mask_thr)
        mg = margins(S, x, prompts, mask, others)[0]
        sets.append(dict(seed=sd, set=x, mask=mask, fit=int((mg > 0).sum()), margin=float(mg.min()),
                         px=int(change_mask(x, mask_thr, grow=0).sum()), picked=False))
        if progress:
            progress((k + 1) / tries)
    best = [c for c in sets if c["fit"] == max(c["fit"] for c in sets)]
    best = [c for c in best if c["px"] <= 1.25 * min(c["px"] for c in best)]
    max(best, key=lambda c: c["margin"])["picked"] = True
    return sets


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--template", default="yellow warning triangle with a black {}")
    ap.add_argument("--word", default=None, help="one sprite per character")
    ap.add_argument("--items", nargs="+", default=None, help="one sprite per item, instead of --word")
    ap.add_argument("--methods", nargs="+", default=["pick"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tries", type=int, default=8, help="pick, lock: seeds tried")
    ap.add_argument("--share", type=float, default=5.0, help="pick, lock: noise level down to which the image is shared")
    ap.add_argument("--mask_thr", type=float, default=0.4, help="how much a pixel must vary over the set to count as different")
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--weights", default="ema0.9999")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--zoom", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    S = Sampler(args.run_dir, args.device, args.weights, steps=args.steps, guidance=args.guidance)
    items = args.items or list(args.word or "SPRITE32")
    prompts, others = item_prompts(args.template, items, not args.items)
    n = len(items)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows, picked = [], {}

    def pick():
        if not picked:
            sets = tried_sets(S, prompts, others, args.seed, args.tries, args.share, args.mask_thr)
            for c in sets:
                print(f"seed {c['seed']}: {c['fit']}/{n} fit their prompt, {c['px']} px differ, "
                      f"lowest margin {c['margin']:.3f}" + ("  <- picked" if c["picked"] else ""))
            picked.update(next(c for c in sets if c["picked"]))
        return picked["set"]

    for m in args.methods:
        name, _, val = m.partition(":")
        if name == "independent":
            x = S.sample(prompts, S.noise(args.seed, n))
        elif name == "seed":
            x = S.sample(prompts, S.noise(args.seed).expand(n, -1, -1, -1))
        elif name == "trunk":
            x = trunk(S, prompts, args.seed, float(val or args.share))
        elif name == "edit":
            ref = S.sample(prompts[:1], S.noise(args.seed))
            x = S.sample(prompts, S.noise(args.seed + 1000).expand(n, -1, -1, -1), start_sigma=float(val or 5), init=ref)
        elif name == "pick":
            x = pick()
        elif name == "lock":
            x = lock(pick(), picked["mask"])
        else:
            raise SystemExit(f"unknown method {m}")
        label = m if name not in ("pick", "lock") else f"{m} (seed {picked['seed']})"
        rows.append((label, x.cpu()))
    sheet(rows, out / "sheet.png", zoom=args.zoom)
    strip(rows[-1][1], out / "strip.png", zoom=args.zoom)
    print("saved", out / "sheet.png", out / "strip.png")


if __name__ == "__main__":
    main()
