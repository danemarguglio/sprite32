"""Dataset and loader: PNG + JSON sidecars, premultiplied RGBA, LANCZOS resize + centre crop to 32x32,
source balancing, exposure caps and recolouring.

Sidecar keys are inverted relative to their content: `description` is the LONG caption,
`verbose_description` the SHORT one. openmoji also has `caption` (the emoji name, e.g.
"flat ticket emoji white background").
"""

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from hf_common import label_index

EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')


def premultiply_alpha(img: Image.Image) -> Image.Image:
    """Convert any PIL image to premultiplied RGBA (RGB * A), as a mode 'RGBa' image.

    Premultiplying BEFORE resize is the correct order: LANCZOS on straight
    alpha bleeds undefined RGB from fully-transparent pixels into edges.
    Sources without alpha (JPEG) come out with A=255 everywhere.

    The mode matters: Pillow premultiplies an 'RGBA' image again inside resize() and divides the
    result back, which left edge pixels of every resized image with roughly straight colour
    (RGB > A). 'RGBa' is resampled band by band. Until 2026-11-13 this returned 'RGBA'.
    """
    arr = np.asarray(img.convert('RGBA'), dtype=np.float32)
    alpha = arr[..., 3:4] / 255.0
    arr[..., :3] *= alpha
    return Image.frombytes('RGBa', img.size, arr.astype(np.uint8).tobytes())  # fromarray(mode=) is deprecated


def image_source(path: Path, image_dir: Path) -> str:
    """Source name of an image = its top-level directory under image_dir."""
    try:
        parts = Path(path).relative_to(image_dir).parts
    except ValueError:
        parts = Path(path).parts
    return parts[0] if len(parts) > 1 else "."


def source_balanced_weights(
    paths: List[Path],
    image_dir: str,
    alpha: float = 1.0,
    source_weights: Optional[dict] = None,
):
    """Per-image sampling weights so source s gets total mass ~ w_s * n_s**alpha.

    alpha=1 reproduces uniform-over-images sampling; alpha=0
    gives every source equal mass; in between flattens the mix (top-3 sources
    are ~85% of the data at alpha=1). source_weights multiplies a source's
    mass (e.g. {"favicons": 0.5}); unknown names are ignored.
    Returns (weights tensor, {source: (count, share)}).
    """
    image_dir = Path(image_dir).resolve()
    sources = [image_source(p, image_dir) for p in paths]
    counts = Counter(sources)
    source_weights = source_weights or {}
    mass = {s: (n ** alpha) * float(source_weights.get(s, 1.0)) for s, n in counts.items()}
    total = sum(mass.values())
    weights = torch.tensor([mass[s] / counts[s] for s in sources], dtype=torch.double)
    summary = {s: (counts[s], mass[s] / total) for s in sorted(counts, key=lambda k: -counts[k])}
    return weights, summary


def capped_probabilities(weights, clusters, max_image: float, max_cluster: float, n_iter: int = 200):
    """Sampling probabilities proportional to `weights`, with no image above `max_image` and no cluster's
    members adding up to more than `max_cluster` (both as probabilities). clusters: LongTensor, one id per
    image (singletons have their own id). The mass taken off capped images goes to the others in proportion
    to their weight. Raises ValueError when the caps cannot hold a total of 1."""
    p = weights.double() / weights.sum()
    size = torch.bincount(clusters).double()
    # a cluster cannot use more than its members' own caps
    cluster_cap = torch.clamp(size * max_image, max=max_cluster)
    if float(cluster_cap.sum()) < 1.0:
        raise ValueError("exposure caps are below the mean exposure per image; raise them or shorten the run")
    for _ in range(n_iter):
        p = torch.clamp(p / p.sum(), max=max_image)
        mass = torch.zeros_like(cluster_cap).index_add_(0, clusters, p)
        p = p * torch.clamp(cluster_cap / mass.clamp(min=1e-300), max=1.0)[clusters]
        if float(p.sum()) > 1 - 1e-9:
            break
    return p / p.sum()


def load_clusters(cluster_file: str, paths: List[Path], image_dir: str):
    """Cluster id per image (LongTensor) from a `path<TAB>cluster` file with paths relative to image_dir
    (scripts/cluster.py); images not listed get an id of their own."""
    image_dir = Path(image_dir).resolve()
    listed = {}
    with open(cluster_file) as fh:
        next(fh)
        for line in fh:
            path, cid = line.rstrip("\n").split("\t")
            listed[path] = cid
    ids, out = {}, []
    for i, p in enumerate(paths):
        try:
            rel = Path(p).resolve().relative_to(image_dir).as_posix()
        except ValueError:
            rel = None
        key = listed.get(rel)
        out.append(ids.setdefault(key, len(ids)) if key is not None else ids.setdefault(("single", i), len(ids)))
    return torch.tensor(out, dtype=torch.long)


def _train_sampler(dataset, alpha: float, source_weights: Optional[dict], cluster_file: Optional[str] = None,
                   max_exposures: Optional[float] = None, max_cluster_exposures: Optional[float] = None,
                   total_draws: Optional[int] = None):
    """WeightedRandomSampler for source balancing, or None when it's a no-op.

    max_exposures / max_cluster_exposures cap how often one image / the members of one near-duplicate
    cluster (cluster_file) are expected to be drawn over the whole run of total_draws samples."""
    capped = max_exposures is not None or max_cluster_exposures is not None
    if alpha == 1.0 and not source_weights and not capped:
        return None
    weights, summary = source_balanced_weights(
        dataset.image_paths, dataset.image_dir, alpha, source_weights)
    n_total = len(dataset)
    if capped:
        if not total_draws:
            raise ValueError("exposure caps need total_draws (num_steps * batch_size)")
        clusters = (load_clusters(cluster_file, dataset.image_paths, dataset.image_dir) if cluster_file
                    else torch.arange(n_total))
        max_image = (max_exposures or float("inf")) / total_draws
        max_cluster = (max_cluster_exposures or float("inf")) / total_draws
        before = weights / weights.sum() * total_draws
        weights = capped_probabilities(weights, clusters, min(max_image, 1.0), min(max_cluster, 1.0))
        after = weights * total_draws
        sources = [image_source(p, Path(dataset.image_dir).resolve()) for p in dataset.image_paths]
        share = Counter()
        for s, w in zip(sources, weights.tolist()):
            share[s] += w
        summary = {s: (n, share[s]) for s, (n, _) in summary.items()}
        q = torch.tensor([0.5, 0.99, 1.0], dtype=torch.double)
        print(f"Exposure caps over {total_draws:,} draws (mean {total_draws / n_total:.0f} per image): "
              f"image <= {max_exposures}, cluster <= {max_cluster_exposures} "
              f"({int((torch.bincount(clusters) > 1).sum()):,} clusters of 2+ from {cluster_file})")
        print("  expected draws per image, median / 99% / max: before "
              + " / ".join(f"{v:.0f}" for v in torch.quantile(before, q).tolist())
              + ", after " + " / ".join(f"{v:.0f}" for v in torch.quantile(after, q).tolist()))
    print(f"Source-balanced sampling (alpha={alpha}, weights={source_weights or {}}):")
    for s, (n, share) in summary.items():
        print(f"  {s:<24} {n:>8} imgs  {n / n_total:6.1%} -> {share:6.1%}")
    return WeightedRandomSampler(weights, num_samples=n_total, replacement=True)


def scan_image_paths(
    image_dir: str,
    extensions: tuple = EXTENSIONS,
) -> List[Path]:
    """Scan a directory for all image files and return sorted paths."""
    image_dir = Path(image_dir).resolve()
    paths = []
    for ext in extensions:
        paths.extend(image_dir.rglob(f'*{ext}'))
        paths.extend(image_dir.rglob(f'*{ext.upper()}'))
    # files only: data_v12 has sprite-sheet slice dirs named like `sheet.png/`
    paths = [p for p in paths if p.is_file()]
    # Sort lexicographically for determinism
    paths.sort(key=lambda p: str(p))
    return paths


def split_dataset_paths(
    image_dir: str,
    val_split: float = 0.05,
    extensions: tuple = EXTENSIONS,
) -> tuple:
    """
    Split image paths into train/val sets deterministically.

    Args:
        image_dir: Path to directory containing images
        val_split: Fraction of images for validation (e.g. 0.05 = 5%)
        extensions: Tuple of valid image extensions

    Returns:
        (train_paths, val_paths) tuple of lists of Path objects
    """
    all_paths = scan_image_paths(image_dir, extensions)
    n = len(all_paths)
    n_val = int(n * val_split)
    n_train = n - n_val

    # Deterministic shuffle
    gen = torch.Generator().manual_seed(42)
    indices = torch.randperm(n, generator=gen).tolist()

    train_paths = [all_paths[i] for i in indices[:n_train]]
    val_paths = [all_paths[i] for i in indices[n_train:]]

    return train_paths, val_paths


class ImageDataset(Dataset):
    """Images under image_dir (or exactly image_paths) as [-1, 1] tensors of image_size x image_size, with a
    caption from the `<image>.json` sidecar when include_captions is set.

    caption_keys: each access picks one available key at random, which mixes the caption styles per step.
    rgba: 4-channel premultiplied RGBA; otherwise RGB with alpha flattened to black.
    recolor_list, recolor_prob: recolour the flat black icons on the list (recolor.py) with this probability,
        changing the colour word in the caption to match.
    """

    def __init__(
        self,
        image_dir: str,
        image_size: int = 32,
        include_captions: bool = False,
        caption_keys: List[str] = ("description", "verbose_description"),
        image_paths: Optional[List[Path]] = None,
        rgba: bool = True,
        recolor_list: Optional[str] = None,
        recolor_prob: float = 0.0,
    ):
        self.image_dir = Path(image_dir).resolve()
        self.image_size = image_size
        self.include_captions = include_captions
        self.caption_keys = list(caption_keys)
        self.rgba = rgba
        self.image_paths = list(image_paths) if image_paths is not None else scan_image_paths(image_dir)
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        print(f"Found {len(self.image_paths)} images in {image_dir}")

        self.recolor_prob = recolor_prob if recolor_list else 0.0
        self.recolor_targets = None  # per image: target colour names, or None
        if self.recolor_prob > 0:
            import recolor
            listed = recolor.load_list(recolor_list)
            self.recolor_targets = [listed.get(self._rel(p)) for p in self.image_paths]
            print(f"Recolouring: {sum(t is not None for t in self.recolor_targets)} listed images, "
                  f"probability {self.recolor_prob}")

        # resize, to tensor, scale to [-1, 1]
        num_channels = 4 if rgba else 3
        self.base_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * num_channels, [0.5] * num_channels),
        ])

    def __len__(self):
        return len(self.image_paths)

    def _rel(self, path) -> Optional[str]:
        """Path relative to image_dir (posix), or None."""
        try:
            return Path(path).resolve().relative_to(self.image_dir).as_posix()
        except ValueError:
            return None

    def _recolor(self, idx, img, caption):
        """(img, caption), recoloured when this image is listed, the draw says so and the caption can be
        rewritten. caption is a string or a tuple whose first item is the caption (LabeledImageDataset)."""
        targets = self.recolor_targets[idx]
        if targets is None or random.random() >= self.recolor_prob:
            return img, caption
        import recolor
        target = random.choice(targets)
        text = caption[0] if isinstance(caption, tuple) else caption
        new = recolor.swap_caption(text, target)
        if new is None:
            return img, caption
        return recolor.recolor_image(img, target), ((new, *caption[1:]) if isinstance(caption, tuple) else new)

    def _load_caption(self, img_path: Path) -> str:
        """One of the caption_keys of the `<image>.json` sidecar, picked at random; "" without one."""
        try:
            with open(f"{img_path}.json", "r") as fh:
                metadata = json.load(fh)
        except Exception:
            return ""
        candidates = [
            metadata[k].strip() for k in self.caption_keys
            if isinstance(metadata.get(k), str) and metadata[k].strip()
        ]
        return random.choice(candidates) if candidates else ""

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        try:
            img = Image.open(img_path)
            caption = self._load_caption(img_path) if self.include_captions else None
            if self.recolor_targets is not None and caption is not None:
                img, caption = self._recolor(idx, img, caption)

            if self.rgba:
                img = premultiply_alpha(img)
            elif img.mode != 'RGB':
                img = img.convert('RGB')  # flattens alpha to black

            img = self.base_transform(img)

            if self.include_captions:
                return img, caption

            return img

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a random other image if this one fails
            return self.__getitem__(random.randint(0, len(self) - 1))


class LabeledImageDataset(ImageDataset):
    """ImageDataset whose caption item is (caption, quality, category) from the same sidecar, so the
    fallback to a random other image on a load error keeps image and labels together."""

    def _load_caption(self, img_path):
        caption = super()._load_caption(img_path)
        try:
            meta = json.load(open(f"{img_path}.json"))
        except (OSError, ValueError):
            meta = {}
        return (caption, *label_index(meta))


def _exclude(paths, exclude_list, data_dir):
    """Drop the paths listed in exclude_list (one per line, `#` lines are comments; relative entries are
    resolved against the directory that holds data_dir)."""
    if not exclude_list:
        return paths
    root = Path(data_dir).resolve().parent
    excluded = set()
    for line in open(exclude_list):
        line = line.strip()
        if line and not line.startswith("#"):
            p = Path(line)
            excluded.add(str(p if p.is_absolute() else (root / p).resolve()))
    kept = [p for p in paths if str(Path(p).resolve()) not in excluded]
    print(f"exclude list: dropped {len(paths) - len(kept)} of {len(paths)}")
    return kept


def train_paths(cfg):
    data_dir = Path(cfg["data_dir"]).expanduser()
    if cfg.get("sources"):  # subset of top-level source dirs (smoke tests)
        paths = [p for s in cfg["sources"] for p in scan_image_paths(str(data_dir / s))]
    else:
        paths, _ = split_dataset_paths(str(data_dir), cfg.get("val_split", 0.05))
    paths = _exclude(paths, cfg.get("exclude_list"), data_dir)
    if cfg.get("name_regex"):  # filename filter (smoke tests)
        paths = [p for p in paths if re.fullmatch(cfg["name_regex"], Path(p).name)]
    if cfg.get("max_images"):
        paths = sorted(paths, key=str)
        random.Random(cfg.get("seed", 0)).shuffle(paths)
        paths = paths[: cfg["max_images"]]
    return paths


def make_dataset(cfg, paths=None, caption_keys=None):
    """Training set by default; `paths` and `caption_keys` override (validation sets). Items are
    (image, caption), or (image, (caption, quality, category)) when cfg label_cond is set."""
    cls = LabeledImageDataset if cfg.get("label_cond") else ImageDataset
    return cls(
        image_dir=str(Path(cfg["data_dir"]).expanduser()),
        image_size=cfg.get("image_size", 32),
        include_captions=True,
        caption_keys=caption_keys or cfg.get("caption_keys", ["description", "verbose_description"]),
        image_paths=train_paths(cfg) if paths is None else paths,
        rgba=cfg.get("channels", 4) == 4,
        # augmentation of the training set only; validation sets (paths given) stay as they are on disk
        recolor_list=cfg.get("recolor_list") if paths is None else None,
        recolor_prob=cfg.get("recolor_prob", 0.0),
    )


def val_paths(cfg, n, seed=0):
    """n fixed images of the val split (never trained on; minus the exclude list) and n fixed training images,
    for the held-out vs. train loss curves."""
    data_dir = Path(cfg["data_dir"]).expanduser()
    train, val = split_dataset_paths(str(data_dir), cfg.get("val_split", 0.05))
    val = _exclude(val, cfg.get("exclude_list"), data_dir)
    train = _exclude(train, cfg.get("exclude_list"), data_dir)
    rng = random.Random(seed)
    return rng.sample(sorted(val, key=str), min(n, len(val))), rng.sample(sorted(train, key=str), min(n, len(train)))


def make_loader(cfg, dataset):
    sampler = _train_sampler(dataset, cfg.get("source_balance_alpha", 1.0), cfg.get("source_weights"),
                             cluster_file=cfg.get("cluster_file"), max_exposures=cfg.get("max_exposures"),
                             max_cluster_exposures=cfg.get("max_cluster_exposures"),
                             total_draws=cfg["num_steps"] * cfg["batch_size"])
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.get("num_workers", 4),
        persistent_workers=cfg.get("num_workers", 4) > 0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def rms_sigma_data(loader, max_batches=50):
    """sigma_data = sqrt(E[y^2])."""
    total, count = 0.0, 0
    for i, (x, _) in enumerate(loader):
        total += x.double().pow(2).sum().item()
        count += x.numel()
        if i + 1 >= max_batches:
            break
    return (total / count) ** 0.5
