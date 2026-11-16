"""FID (and kept samples for the CLIP scorer) of an hf_train.py run against held-out images.

Samples come from EDM Heun (hf_sample.heun_sample; --sampler dpmpp_2m for the sampler of hf_sample.py and the
web UI) with captions of the held-out images, are composited over white and rounded to 8 bits like the reference images, and go through clean-fid (anti-aliased
resizing, which matters at 32x32). See BENCHMARK.md.

    # reference statistics of the held-out images, once
    python hf_eval.py --precompute --holdout_list curation/holdout.txt --stats_path STATS.npz

    python hf_eval.py RUN_DIR --stats_path STATS.npz --holdout_list curation/holdout.txt --amp \
        [--caption_key description] [--guidance_scale 3] [--keep_dir DIR]
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from hf_common import N_QUALITY, build_scheduler, label_index, load_run_meta, pick_device
from hf_data import ImageDataset
from hf_sample import heun_sample, load_unet, sample_images
from hf_text import build_text_encoder

# clean-fid's resizer is a local function, which spawn-based DataLoader workers (macOS) can't pickle
_FID_WORKERS = 4 if sys.platform.startswith("linux") else 0


def holdout_paths(holdout_list):
    """The reference images: one path per line (repo-relative or absolute; `#` lines are comments)."""
    with open(holdout_list) as fh:
        paths = [Path(l.strip()) for l in fh if l.strip() and not l.startswith('#')]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(paths)} holdout images missing, e.g. {missing[0]}")
    print(f"Holdout list {holdout_list}: {len(paths)} images")
    return paths


def holdout_captions(holdout_list, key):
    """Captions of the held-out images from their `<image>.json` sidecars. The sidecar keys are inverted
    relative to their content: `description` is the long caption, `verbose_description` the short one."""
    paths = holdout_paths(holdout_list)
    caps = []
    for p in paths:
        try:
            text = json.load(open(f"{p}.json")).get(key)
        except Exception:
            continue
        if isinstance(text, str) and text.strip():
            caps.append(text.strip())
    print(f"{len(caps)} '{key}' captions from {len(paths)} held-out images")
    return caps


def _features(folder, device):
    from cleanfid import fid as cleanfid
    model = cleanfid.build_feature_extractor("clean", device)
    feats = cleanfid.get_folder_features(folder, model=model, mode="clean", num_workers=_FID_WORKERS,
                                         batch_size=64, device=device)
    return np.mean(feats, axis=0), np.cov(feats, rowvar=False)


def precompute_real_stats(holdout_list, stats_path):
    """Inception statistics (mu, sigma) of the held-out images, each rendered exactly as training sees it
    (32x32 via ImageDataset) and composited over white like the generated samples, so real and generated
    images differ only in content, not in resolution or alpha handling."""
    stats_path = Path(stats_path)
    if stats_path.exists():
        print(f"Real stats already exist at {stats_path}, skipping precompute")
        return
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    paths = holdout_paths(holdout_list)
    tmpdir = tempfile.mkdtemp(prefix="fid_real_")
    try:
        ds = ImageDataset(".", image_paths=paths)
        for i in range(len(ds)):
            x = (ds[i] + 1) / 2
            x = x[:3] + (1 - x[3:4])  # premultiplied RGBA over white
            x = (x.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
            Image.fromarray(x).save(Path(tmpdir) / f"{i:06d}.png")
        mu, sigma = _features(tmpdir, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        np.savez(stats_path, mu=mu, sigma=sigma)
        print(f"Saved real stats to {stats_path}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def compute_fid(sample, device, stats_path, num_samples=10000, batch_size=64, captions=None, keep_dir=None):
    """FID of num_samples generated images against the statistics in stats_path.

    sample(captions) returns one image per caption, premultiplied RGBA in [-1, 1]. The captions are cycled
    through in order; without captions every image is sampled from the empty caption, the model's trained
    unconditional branch. keep_dir: also save the PNGs and their prompts (prompts.jsonl) there."""
    from cleanfid import fid as cleanfid

    stats = np.load(stats_path)
    mu_real, sigma_real = stats["mu"], stats["sigma"]
    tmpdir = tempfile.mkdtemp(prefix="fid_gen_")
    try:
        generated = 0
        pbar = tqdm(total=num_samples, desc="Generating for FID")
        while generated < num_samples:
            current = min(batch_size, num_samples - generated)
            if captions:
                offset = generated % len(captions)
                batch_captions = [captions[(offset + j) % len(captions)] for j in range(current)]
            else:
                batch_captions = [""] * current
            samples = (sample(batch_captions) + 1) / 2
            samples = samples[:, :3] + (1 - samples[:, 3:4])  # premultiplied RGBA over white
            # round, as the reference images are (precompute_real_stats)
            samples = (torch.clamp(samples, 0, 1) * 255).round().byte().cpu()
            for j in range(current):
                Image.fromarray(samples[j].permute(1, 2, 0).numpy()).save(Path(tmpdir) / f"{generated + j:06d}.png")
            generated += current
            pbar.update(current)
        pbar.close()

        mu_gen, sigma_gen = _features(tmpdir, device)
        fid_score = cleanfid.frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        print(f"FID: {fid_score:.2f} ({num_samples} samples)")
        if keep_dir:
            keep = Path(keep_dir)
            keep.mkdir(parents=True, exist_ok=True)
            with open(keep / "prompts.jsonl", "w") as fh:
                for i in range(num_samples):
                    name = f"{i:06d}.png"
                    shutil.copy(Path(tmpdir) / name, keep / name)
                    fh.write(json.dumps({"file": name,
                                         "prompt": captions[i % len(captions)] if captions else ""}) + "\n")
        return fid_score
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--precompute", action="store_true",
                    help="write the reference statistics of --holdout_list to --stats_path and exit")
    ap.add_argument("--weights", default=None, help="raw | ema<decay> (default: longest EMA)")
    ap.add_argument("--stats_path", required=True)
    ap.add_argument("--num_samples", type=int, default=10000)
    ap.add_argument("--num_steps", type=int, default=20)
    ap.add_argument("--sampler", default="heun", choices=["heun", "dpmpp_2m"],
                    help="heun: 2 * num_steps - 1 UNet evaluations (the benchmark); dpmpp_2m: num_steps")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--guidance_scale", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--caption_key", default=None, help="sample from the held-out captions under this key")
    ap.add_argument("--holdout_list", default=None, help="the held-out images (curation/holdout.txt)")
    ap.add_argument("--keep_dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", default="image", choices=["image", "null", "q5"],
                    help="label_cond runs: each caption's own image labels (default), no labels, or quality 5")
    args = ap.parse_args()
    if (args.precompute or args.caption_key) and not args.holdout_list:
        ap.error("--precompute and --caption_key need --holdout_list")
    if args.precompute:
        precompute_real_stats(args.holdout_list, args.stats_path)
        return
    if not args.run_dir:
        ap.error("run_dir is required")

    cfg = load_run_meta(args.run_dir)
    device = pick_device()
    weights = args.weights or f"ema{max(cfg['ema_decays'])}"
    print(f"{args.run_dir} weights {weights}")
    unet = load_unet(args.run_dir, weights, device)
    text = build_text_encoder(cfg, device)
    edm = build_scheduler(cfg, "euler")
    dpm = build_scheduler(cfg, "dpmpp_2m") if args.sampler == "dpmpp_2m" else None
    batch_seed = iter(range(args.seed, 2**31))  # sample_images draws its noise from the seed
    captions = holdout_captions(args.holdout_list, args.caption_key) if args.caption_key else None
    # label_cond runs: caption -> (quality, category) of the image it came from; "" -> no labels
    caption_labels, default = {}, (0, 0)
    if cfg.get("label_cond"):
        if args.labels == "q5":
            default = (N_QUALITY, 0)
        elif args.labels == "image" and args.caption_key:
            for p in holdout_paths(args.holdout_list):
                try:
                    meta = json.load(open(f"{p}.json"))
                except (OSError, ValueError):
                    continue
                if isinstance(meta.get(args.caption_key), str):
                    caption_labels[meta[args.caption_key].strip()] = label_index(meta)
        print(f"labels: {args.labels} ({len(caption_labels)} captions mapped)")

    @torch.no_grad()
    def sample(caps):
        labels = None
        if cfg.get("label_cond"):
            ql = [(0, 0) if not c else caption_labels.get(c, default) for c in caps]
            labels = (torch.tensor([q for q, _ in ql]), torch.tensor([k for _, k in ql]))
        shape = (len(caps), cfg.get("channels", 4), 32, 32)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            if dpm is not None:
                return sample_images(unet, text, dpm, caps, 1, args.num_steps, args.guidance_scale,
                                     seed=next(batch_seed), channels=shape[1], labels=labels)
            return heun_sample(unet, text, edm, shape, device, args.num_steps, caps, args.guidance_scale, None,
                               cfg.get("sigma_min", 0.002), cfg.get("sigma_max", 80.0), cfg.get("rho", 7.0),
                               labels=labels)

    unet.eval()
    torch.manual_seed(args.seed)
    fid = compute_fid(sample, device, args.stats_path, num_samples=args.num_samples,
                      batch_size=args.batch_size, captions=captions, keep_dir=args.keep_dir)
    print(f"Final FID: {fid:.2f}")


if __name__ == "__main__":
    main()
