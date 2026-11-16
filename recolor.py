"""Recolouring of flat black icons, with the colour word in the caption changed to match.

About a fifth of the training images are single-colour black drawings on a transparent background (mostly
iconify_mono), so "black" is the only colour the model sees for most of those subjects. For the images listed
in a recolour list (scripts/build_recolor_list.py: path and a few target colours fixed per image), the dataset
replaces the RGB of every pixel with a shade of one target colour (alpha untouched) with probability
recolor_prob, and replaces the word "black" in the caption with the colour's name. The image is only
recoloured when the caption drawn for that sample names the colour and can be rewritten (swap_caption), so a
recoloured image always comes with a caption that says so. The targets are fixed per image so that
scripts/build_text_cache.py can encode every rewritten caption ahead of training.
"""
import hashlib
import random
import re

import numpy as np
from PIL import Image

# colour name -> shades; names and shades as scripts/gen_rendered_text.py PALETTE, without black
TARGETS = {
    "dark gray": [(64, 64, 64), (74, 74, 84), (51, 51, 51)],
    "gray": [(128, 128, 128), (140, 140, 150)],
    "white": [(255, 255, 255), (248, 248, 248)],
    "red": [(220, 40, 40), (200, 30, 45), (235, 60, 50)],
    "dark red": [(140, 20, 30), (120, 25, 25)],
    "orange": [(240, 140, 30), (255, 150, 40)],
    "yellow": [(250, 215, 40), (255, 230, 60)],
    "green": [(50, 170, 70), (60, 190, 90)],
    "dark green": [(20, 100, 50), (30, 90, 40)],
    "teal": [(20, 140, 140), (0, 150, 136)],
    "sky blue": [(110, 180, 245), (135, 200, 250)],
    "blue": [(40, 100, 220), (50, 120, 235)],
    "navy blue": [(20, 35, 95), (25, 40, 110)],
    "purple": [(130, 60, 190), (150, 80, 200)],
    "pink": [(250, 140, 180), (255, 160, 200)],
    "brown": [(130, 80, 40), (120, 72, 48)],
}
NAMES = sorted(TARGETS)
N_TARGETS = 3

_BLACK = re.compile(r"\bblack\b", re.I)
# captions that already name another colour, or use "black" as part of a name, are left alone
_BLOCK = re.compile(
    r"\b(?:gr[ae]y(?:scale|ish)?|red|orange|yellow|green|blue|purple|violet|pink|brown|teal|cyan|magenta|navy|tan|"
    r"beige|cream|gold(?:en)?|silver|maroon|turquoise|lime|olive|indigo|colou?r(?:ed|ful)|multicolou?r(?:ed)?|"
    r"rainbow)\b|\bblack[- ]and[- ]white\b|\bblack[- ](?:hole|widow|belt|tie|friday|pepper|panther|bear|jack|"
    r"board|list|market|box|berry|bird|smith)", re.I)
_WHITE = re.compile(r"\bwhite\b", re.I)


def targets_for(rel_path, n=N_TARGETS):
    """The n target colour names of an image, fixed by its path."""
    seed = int.from_bytes(hashlib.blake2b(rel_path.encode(), digest_size=8).digest(), "little")
    return random.Random(seed).sample(NAMES, n)


def swap_caption(caption, target):
    """`caption` with "black" replaced by the target colour name, or None when the caption does not name
    black, names another colour, or already uses the target's name (white cut-outs in a white icon)."""
    if not caption or not _BLACK.search(caption) or _BLOCK.search(caption):
        return None
    if target == "white" and _WHITE.search(caption):
        return None

    def sub(m):
        return target.capitalize() if m.group(0)[0].isupper() else target
    return _BLACK.sub(sub, caption)


def variants(caption, targets):
    """Every rewritten caption the dataset can produce for this caption (for the text cache)."""
    return [c for c in (swap_caption(caption, t) for t in targets) if c]


def recolor_image(img, target, rng=random):
    """RGBA copy of img with every pixel's RGB set to a shade of the target colour; alpha is kept."""
    a = np.array(img.convert("RGBA"))
    a[..., :3] = rng.choice(TARGETS[target])
    return Image.fromarray(a, "RGBA")


def load_list(path):
    """{path relative to the data dir: [target colour names]} from a recolour list."""
    out = {}
    with open(path) as fh:
        next(fh)
        for line in fh:
            rel, targets = line.rstrip("\n").split("\t")
            out[rel] = targets.split(",")
    return out
