"""Sample PNG grids from a run trained by hf_train.py.

    .venv/bin/python hf_sample.py RUN_DIR --weights ema0.9999 --prompts "red heart icon" "green sword"
    .venv/bin/python hf_sample.py RUN_DIR --images data_v13/openmoji/1F600.png ...  # prompts from sidecars,
                                                                              # real image as row 0
Grid layout: one column per prompt, one row per seed.
Guidance: CFG against the encoded "" branch, optional guidance interval
(Kynkaenniemi et al. 2024) and autoguidance (--guide_weights: a weaker UNet run
with the same prompt replaces the unconditional branch, Karras et al. 2024).
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

import hf_common
from hf_common import CATEGORIES, add_labels, build_scheduler, load_run_meta, null_labels, pick_device, to_pil_grid


def karras_sigmas(num_steps, sigma_min=0.002, sigma_max=80.0, rho=7.0, device="cpu"):
    """EDM (Karras et al. 2022) sigma schedule: num_steps sigmas + a final 0."""
    i = torch.arange(num_steps, dtype=torch.float32, device=device)
    s = (sigma_max ** (1 / rho) + i / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([s, torch.zeros(1, device=device)])


@torch.no_grad()
def heun_sample(unet, text, edm, shape, device, num_steps=20, captions=None, guidance=1.0,
                interval=None, sigma_min=0.002, sigma_max=80.0, rho=7.0, labels=None):
    """EDM Algorithm 1 (deterministic 2nd-order Heun). The starting noise is torch.randn(shape) * sigma_max
    from the global RNG.
    `edm` is any diffusers EDM scheduler; only its precondition_* functions are used.
    labels: (quality, category) indices for a label_cond model, else None; the unconditional
    branch gets null labels."""
    B = shape[0]
    if getattr(unet, "distilled_guidance", None):  # already in the weights (distill_guidance.py)
        guidance = 1.0
    captions = captions if captions is not None else [""] * B
    ctx, mask, pooled = text(captions)
    pooled = add_labels(pooled, labels)
    if guidance != 1.0:
        uctx, umask, upooled = text([""] * B)
        upooled = add_labels(upooled, None if labels is None else tuple(torch.zeros_like(t) for t in labels))

    def denoise(x, sigma):
        s4 = sigma.expand(B)[:, None, None, None]
        x_in = edm.precondition_inputs(x, s4)
        c_noise = edm.precondition_noise(sigma.expand(B))
        out = unet(x_in, c_noise, encoder_hidden_states=ctx, class_labels=pooled,
                   encoder_attention_mask=mask).sample
        if guidance != 1.0 and (interval is None or interval[0] < float(sigma) <= interval[1]):
            unc = unet(x_in, c_noise, encoder_hidden_states=uctx, class_labels=upooled,
                       encoder_attention_mask=umask).sample
            out = unc + guidance * (out - unc)
        return edm.precondition_outputs(x, out.float(), s4)

    sigmas = karras_sigmas(num_steps, sigma_min, sigma_max, rho, device)
    x = torch.randn(shape, device=device) * sigmas[0]
    for i in range(num_steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        d_cur = (x - denoise(x, sigma)) / sigma
        x_next = x + (sigma_next - sigma) * d_cur
        if sigma_next > 0:
            d_next = (x_next - denoise(x_next, sigma_next)) / sigma_next
            x_next = x + (sigma_next - sigma) * (d_cur + d_next) / 2
        x = x_next
    return x


@torch.no_grad()
def sample_images(unet, text, scheduler, prompts, n_per_prompt=4, steps=35, guidance=3.0,
                  interval=None, seed=0, guide_unet=None, s_churn=0.0, channels=4, size=32, labels=None):
    """Returns (n_per_prompt * len(prompts), C, H, W) premultiplied RGBA in [-1, 1],
    ordered row-major: row r = seed r, column p = prompt p. labels as heun_sample."""
    device = next(unet.parameters()).device
    dtype = next(unet.parameters()).dtype
    if getattr(unet, "distilled_guidance", None):  # already in the weights (distill_guidance.py)
        guidance = 1.0
    batch_prompts = [p for _ in range(n_per_prompt) for p in prompts]
    B = len(batch_prompts)
    ctx, mask, pooled = text(batch_prompts)
    uctx, umask, upooled = text([""] * B)
    pooled = add_labels(pooled, labels)
    upooled = add_labels(upooled, None if labels is None else tuple(torch.zeros_like(t) for t in labels))
    ctx, uctx, pooled, upooled = (t.to(dtype) for t in (ctx, uctx, pooled, upooled))

    scheduler.set_timesteps(steps, device=device)
    gen = torch.Generator("cpu").manual_seed(seed)
    x = torch.randn(B, channels, size, size, generator=gen).to(device) * scheduler.init_noise_sigma
    step_kw = {"s_churn": s_churn, "generator": gen} if s_churn else {}
    for i, t in enumerate(scheduler.timesteps):
        sigma = float(scheduler.sigmas[i])
        x_in = scheduler.scale_model_input(x, t).to(dtype)
        out = unet(x_in, t, encoder_hidden_states=ctx, class_labels=pooled,
                   encoder_attention_mask=mask).sample
        guided = guidance != 1.0 and (interval is None or interval[0] < sigma <= interval[1])
        if guided:
            # D is affine in the network output F with the same coefficients for both
            # branches, so guiding F is identical to guiding D.
            if guide_unet is not None:
                bad = guide_unet(x_in, t, encoder_hidden_states=ctx, class_labels=pooled,
                                 encoder_attention_mask=mask).sample
            else:
                bad = unet(x_in, t, encoder_hidden_states=uctx, class_labels=upooled,
                           encoder_attention_mask=umask).sample
            out = bad + guidance * (out - bad)
        x = scheduler.step(out.float(), t, x, **step_kw).prev_sample
    return x.clamp(-1, 1)


def load_unet(run_dir, weights, device):
    return hf_common.load_unet(Path(run_dir) / f"unet_{weights}", load_run_meta(run_dir), device)


def caption_for(path, key):
    meta = json.load(open(f"{path}.json"))
    return meta.get(key) or meta.get("description") or ""


def load_reference(path, size=32):
    """Real image through the training pipeline (premultiplied RGBA, [-1, 1])."""
    from hf_data import premultiply_alpha
    from torchvision import transforms
    img = premultiply_alpha(Image.open(path))
    tf = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize([0.5] * 4, [0.5] * 4)])
    return tf(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--weights", default=None, help="raw | ema<decay> (default: longest EMA)")
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--images", nargs="*", default=None, help="use these images' sidecar captions")
    ap.add_argument("--caption_key", default="description")
    ap.add_argument("--n", type=int, default=4, help="samples per prompt (rows)")
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--interval", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--guide_weights", default=None, help="autoguidance: RUN_DIR/unet_<name> or a path")
    ap.add_argument("--sampler", default="dpmpp_2m", choices=["dpmpp_2m", "dpmpp_2m_sde", "euler", "heun"],
                    help="heun = EDM Algorithm 1, the benchmark's sampler")
    ap.add_argument("--s_churn", type=float, default=0.0, help="euler only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quality", type=int, default=5, help="label_cond runs: 1-5, 0 = no label")
    ap.add_argument("--category", default=None, choices=CATEGORIES, help="label_cond runs (default: no label)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_run_meta(args.run_dir)
    device = pick_device(args.device)
    weights = args.weights or f"ema{max(cfg['ema_decays'])}"
    unet = load_unet(args.run_dir, weights, device)
    guide = None
    if args.guide_weights:
        p = Path(args.guide_weights)
        p = p if p.exists() else Path(args.run_dir) / f"unet_{args.guide_weights}"
        guide = hf_common.load_unet(p, cfg, device)  # a weaker snapshot of the same architecture
    from hf_text import build_text_encoder
    text = build_text_encoder(cfg, device)
    scheduler = build_scheduler(cfg, "euler" if args.sampler == "heun" else args.sampler)

    refs = None
    if args.images:
        prompts = [caption_for(p, args.caption_key) for p in args.images]
        refs = torch.stack([load_reference(p, cfg.get("image_size", 32)) for p in args.images])
    else:
        prompts = args.prompts or cfg["sample_prompts"]
    labels = null_labels(cfg, args.n * len(prompts), args.quality,
                         CATEGORIES.index(args.category) + 1 if args.category else 0)
    if args.sampler == "heun":
        torch.manual_seed(args.seed)
        size = cfg.get("image_size", 32)
        x = heun_sample(unet, text, scheduler, (args.n * len(prompts), cfg.get("channels", 4), size, size),
                        device, args.steps, [p for _ in range(args.n) for p in prompts], args.guidance,
                        args.interval, labels=labels).clamp(-1, 1)
    else:
        x = sample_images(unet, text, scheduler, prompts, args.n, args.steps, args.guidance,
                          args.interval, args.seed, guide, args.s_churn, cfg.get("channels", 4),
                          cfg.get("image_size", 32), labels)
    if refs is not None:
        x = torch.cat([refs.to(x.device), x])
    out = Path(args.out or Path(args.run_dir) / "samples" / f"{weights}_g{args.guidance}_s{args.steps}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    to_pil_grid(x, nrow=len(prompts)).save(out)
    to_pil_grid(x, nrow=len(prompts), background="alpha").save(out.with_suffix(".rgba.png"))
    with open(out.with_suffix(".prompts.txt"), "w") as fh:
        fh.write("\n".join(f"col {i}: {p}" for i, p in enumerate(prompts)) + "\n")
    xs = x.float().cpu()
    print(f"saved {out} ({len(prompts)} prompts x {args.n}{' + reference row' if refs is not None else ''}); "
          f"finite={bool(torch.isfinite(xs).all())} std={xs.std():.3f}")


if __name__ == "__main__":
    main()
