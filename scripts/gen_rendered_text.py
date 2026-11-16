#!/usr/bin/env python3
"""Generate a "rendered text" training source: 32x32 RGBA pixel renders of short strings with exact captions.

Every image is a short string (letter, digit, number, word, two-word phrase, random letters/digits) drawn
with a bitmap/pixel font, no anti-aliasing (alpha is 0 or 255), either alone or inside a simple container
(speech bubble, button, sign, banner, badge, label, price tag, keyboard key, dialog box, chip, tile).
Because the strings are rendered here, both captions name the exact string (case and punctuation).

Output (default data_gen/rendered_text/):
  images/<layout>/<nnnnnn>.png + .png.json   32x32 RGBA + sidecar in the data_v12 convention
                                             (description = LONG caption, verbose_description = SHORT)
  _index.tsv            data_v12/_index.tsv columns (dst, src, group, source, caption_source, category,
                        quality) + text, layout, font, font_scale, n_chars, n_lines, short_caption
  _stats.json           counts by layout / font / string length / string kind / category / quality
  heldout_strings.txt   strings that never appear in the set (case-insensitive), for generalisation tests
  _sheets/*.png         contact sheets (nearest-neighbour upscaled, expected string printed under each cell)
  _fonts/               downloaded fonts (see FONT_SOURCES; fetch with --download-fonts)

Deterministic for a given --seed and --count (single random.Random stream, no set-order dependence).
Needs only Pillow and numpy.

  python3 scripts/gen_rendered_text.py --download-fonts          # once; fetches + checksums the fonts
  python3 scripts/gen_rendered_text.py                            # 60,000 images, seed 0
  python3 scripts/gen_rendered_text.py --count 2000 --out /tmp/rt_smoke
"""

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "data_gen" / "rendered_text"
SOURCE = "rendered_text"
S = 32  # canvas size

# ------------------------------------------------------------------------------------------- fonts

X11_URL = "https://www.cl.cam.ac.uk/~mgk25/download/ucs-fonts.tar.gz"
X11_HOME = "https://www.cl.cam.ac.uk/~mgk25/ucs-fonts.html"
X11_SHA256 = "702fd1cdef9123e1871622a897727977c0933a420c50c94198f5bb22de8f0f8a"
X11_MEMBERS = ["README", "5x7.bdf", "5x8.bdf", "6x10.bdf", "7x13B.bdf", "9x15B.bdf", "10x20.bdf"]
GF = "https://raw.githubusercontent.com/google/fonts/main/ofl/"
GF_HOME = "https://github.com/google/fonts/tree/main/ofl/"
OFL = "SIL Open Font License 1.1"
PD = "Public domain"

# name -> dict(kind, path (under _fonts/), size (ttf px), licence, url, credit, style word for captions)
FONT_SOURCES = {
    "mini3x5": dict(kind="builtin", licence="CC0-1.0", url="scripts/gen_rendered_text.py (MINI3X5)", avoid="0Z",
                    credit="3x5 bitmap font hand-defined in the generator script"),
    "x11-5x7": dict(kind="bdf", path="x11-misc-fixed/5x7.bdf", licence=PD, url=X11_HOME, avoid="MW&mw",
                    credit="X11 misc-fixed 5x7 (Markus Kuhn's ucs-fonts distribution)"),
    "x11-5x8": dict(kind="bdf", path="x11-misc-fixed/5x8.bdf", licence=PD, url=X11_HOME, avoid="MW&%",
                    credit="X11 misc-fixed 5x8 (Markus Kuhn's ucs-fonts distribution)"),
    "x11-6x10": dict(kind="bdf", path="x11-misc-fixed/6x10.bdf", licence=PD, url=X11_HOME,
                     credit="X11 misc-fixed 6x10 (Markus Kuhn's ucs-fonts distribution)"),
    "x11-7x13B": dict(kind="bdf", path="x11-misc-fixed/7x13B.bdf", licence=PD, url=X11_HOME, bold=True,
                      avoid="MWmw",
                      credit="X11 misc-fixed 7x13 bold (Markus Kuhn's ucs-fonts distribution)"),
    "x11-9x15B": dict(kind="bdf", path="x11-misc-fixed/9x15B.bdf", licence=PD, url=X11_HOME, bold=True,
                      credit="X11 misc-fixed 9x15 bold (Markus Kuhn's ucs-fonts distribution)"),
    "x11-10x20": dict(kind="bdf", path="x11-misc-fixed/10x20.bdf", licence=PD, url=X11_HOME,
                      credit="X11 misc-fixed 10x20 (Markus Kuhn's ucs-fonts distribution)"),
    "pressstart2p": dict(kind="ttf", path="pressstart2p/PressStart2P-Regular.ttf", size=8, licence=OFL, bold=True,
                         url=GF_HOME + "pressstart2p", dl=GF + "pressstart2p/PressStart2P-Regular.ttf",
                         credit="Press Start 2P, Copyright 2012 The Press Start 2P Project Authors (cody@zone38.net)"),
    "silkscreen": dict(kind="ttf", path="silkscreen/Silkscreen-Regular.ttf", size=8, licence=OFL, avoid="&",
                       url=GF_HOME + "silkscreen", dl=GF + "silkscreen/Silkscreen-Regular.ttf",
                       credit="Silkscreen, Copyright 2001 The Silkscreen Project Authors (Jason Kottke)"),
}

# (font, integer scale). Scaled variants are for 1-3 character strings.
FONT_VARIANTS = [
    ("mini3x5", 1), ("mini3x5", 2), ("mini3x5", 3), ("mini3x5", 4),
    ("x11-5x7", 1), ("x11-5x7", 2), ("x11-5x7", 3),
    ("x11-5x8", 1), ("x11-5x8", 2),
    ("x11-6x10", 1), ("x11-6x10", 2),
    ("x11-7x13B", 1), ("x11-7x13B", 2),
    ("x11-9x15B", 1), ("x11-9x15B", 2),
    ("x11-10x20", 1),
    ("pressstart2p", 1), ("pressstart2p", 2), ("pressstart2p", 3),
    ("silkscreen", 1), ("silkscreen", 2), ("silkscreen", 3),
]

PUNCT = "!?.:-+%$#/&'"
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" + PUNCT + " "

# Hand-defined 3x5 font (uppercase, digits, a little punctuation, and a lowercase x for "x3").
# M and W are 5 px wide, N and K are 4 px wide: at 3 px they cannot be told apart from H (or X). The 0 and Z
# glyphs are one pixel away from 8/U and 2, so they are defined but not used (FONT_SOURCES "avoid").
MINI3X5 = {
    "A": ".#. #.# ### #.# #.#", "B": "##. #.# ##. #.# ##.", "C": ".## #.. #.. #.. .##", "D": "##. #.# #.# #.# ##.",
    "E": "### #.. ##. #.. ###", "F": "### #.. ##. #.. #..", "G": ".## #.. #.# #.# .##", "H": "#.# #.# ### #.# #.#",
    "I": "### .#. .#. .#. ###", "J": "..# ..# ..# #.# .#.", "K": "#..# #.#. ##.. #.#. #..#", "L": "#.. #.. #.. #.. ###",
    "M": "#...# ##.## #.#.# #...# #...#", "N": "#..# ##.# #.## #..# #..#", "O": ".#. #.# #.# #.# .#.", "P": "##. #.# ##. #.. #..",
    "Q": ".#. #.# #.# ### .##", "R": "##. #.# ##. #.# #.#", "S": ".## #.. .#. ..# ##.", "T": "### .#. .#. .#. .#.",
    "U": "#.# #.# #.# #.# ###", "V": "#.# #.# #.# #.# .#.", "W": "#...# #...# #.#.# ##.## #...#", "X": "#.# #.# .#. #.# #.#",
    "Y": "#.# #.# .#. .#. .#.", "Z": "### ..# .#. #.. ###",
    "0": "### #.# #.# #.# ###", "1": ".#. ##. .#. .#. ###", "2": "##. ..# .#. #.. ###", "3": "##. ..# .#. ..# ##.",
    "4": "#.# #.# ### ..# ..#", "5": "### #.. ##. ..# ##.", "6": ".## #.. ### #.# ###", "7": "### ..# .#. .#. .#.",
    "8": "### #.# ### #.# ###", "9": "### #.# ### ..# ##.",
    "!": ".#. .#. .#. ... .#.", "?": "##. ..# .#. ... .#.", ".": "... ... ... ... .#.", ":": "... .#. ... .#. ...",
    "-": "... ... ### ... ...", "+": "... .#. ### .#. ...", "%": "#.# ..# .#. #.. #.#", "#": "#.# ### #.# ### #.#",
    "/": "..# ..# .#. #.. #..", "x": "... #.# .#. #.# ...", " ": "... ... ... ... ...",
}


class BitFont:
    """A bitmap font: glyphs[ch] = bool array (cell_h, advance); ink may be anywhere in the cell."""

    def __init__(self, name, glyphs, cell_h):
        self.name, self.glyphs, self.cell_h = name, glyphs, cell_h
        self.info = FONT_SOURCES[name]
        self._drop_ambiguous()
        rows = np.where(np.any(np.concatenate([self.glyphs[c] for c in "HEX" if c in self.glyphs], 1), 1))[0]
        self.cap_h = int(rows[-1] - rows[0] + 1)
        # Glyphs judged too easy to misread at this size (3-4 px wide M/W/N look like H, lowercase with
        # a 3 px x-height): strings containing them are drawn in another font instead.
        self.avoided = [c for c in self.info.get("avoid", "") if c in self.glyphs]
        for c in self.avoided:
            del self.glyphs[c]
        self.has_lower = sum(c in self.glyphs for c in "abcdefghijklmnopqrstuvwxyz") >= 20

    def _drop_ambiguous(self):
        """Remove characters whose glyph bitmap is identical to another character's (e.g. small-caps
        lowercase in Silkscreen, or O/0 in a tiny font): a caption could not tell them apart."""
        seen = defaultdict(list)
        for ch, g in self.glyphs.items():
            if ch == " ":
                continue
            ys, xs = np.where(g)
            if len(ys) == 0:
                seen[b"empty"].append(ch)
                continue
            # keep the vertical position (distinguishes ' from ,) but not the horizontal one
            key = (g.shape[0], ys.min(), g[:, xs.min():xs.max() + 1].tobytes(), xs.max() - xs.min())
            seen[key].append(ch)
        self.dropped = []
        for key, chars in seen.items():
            if key == b"empty":
                drop = chars
            elif len(chars) > 1:
                # lowercase duplicates of uppercase (small caps): keep the uppercase one
                ups = [c for c in chars if c.isupper()]
                drop = [c for c in chars if c not in ups] if len(ups) == 1 else chars
            else:
                drop = []
            for c in drop:
                self.dropped.append(c)
                del self.glyphs[c]

    def can_render(self, text):
        return all(c in self.glyphs for c in text if c != "\n")

    def line(self, text, extra=0):
        """One line as a bool array (cell_h, width); `extra` adds letter spacing."""
        cols = []
        for i, c in enumerate(text):
            cols.append(self.glyphs[c])
            if extra and i < len(text) - 1:
                cols.append(np.zeros((self.cell_h, extra), bool))
        return np.concatenate(cols, 1)

    def render(self, lines, scale=1, extra=0, gap=1, align="center"):
        """Lines of text as one tight bool mask (cropped to ink), integer-upscaled by `scale`."""
        rasters = [self.line(t, extra) for t in lines]
        ink_rows = np.where(np.any(np.concatenate(rasters, 1), 1))[0]
        if len(ink_rows) == 0:
            return None
        top, bot = ink_rows[0], ink_rows[-1] + 1
        trimmed = []
        for r in rasters:
            r = r[top:bot]
            xs = np.where(np.any(r, 0))[0]
            if len(xs) == 0:
                return None
            trimmed.append(r[:, xs[0]:xs[-1] + 1])
        w = max(r.shape[1] for r in trimmed)
        h = bot - top
        out = np.zeros((h * len(trimmed) + gap * (len(trimmed) - 1), w), bool)
        for i, r in enumerate(trimmed):
            x = 0 if align == "left" else (w - r.shape[1]) // 2
            out[i * (h + gap):i * (h + gap) + h, x:x + r.shape[1]] = r
        if scale > 1:
            out = np.kron(out, np.ones((scale, scale), bool))
        ys = np.where(np.any(out, 1))[0]
        return out[ys[0]:ys[-1] + 1]


def load_builtin(name):
    glyphs = {}
    for ch, rows in MINI3X5.items():
        g = np.array([[c == "#" for c in r] for r in rows.split()], bool)
        assert g.shape[0] == 5 and g.ndim == 2, ch
        glyphs[ch] = np.concatenate([g, np.zeros((5, 1), bool)], 1)  # 1 px letter spacing
    return BitFont(name, glyphs, 5)


def load_bdf(name, path):
    """Minimal BDF reader (ASCII range only)."""
    asc = desc = None
    glyphs, cur = {}, None
    with open(path, encoding="latin-1") as fh:
        lines = fh.read().splitlines()
    i = 0
    while i < len(lines):
        t = lines[i].split()
        i += 1
        if not t:
            continue
        if t[0] == "FONT_ASCENT":
            asc = int(t[1])
        elif t[0] == "FONT_DESCENT":
            desc = int(t[1])
        elif t[0] == "ENCODING":
            cur = {"enc": int(t[1])}
        elif t[0] == "DWIDTH" and cur is not None:
            cur["adv"] = int(t[1])
        elif t[0] == "BBX" and cur is not None:
            cur["bbx"] = tuple(int(v) for v in t[1:5])
        elif t[0] == "BITMAP" and cur is not None:
            w, h, xo, yo = cur["bbx"]
            rows = []
            for _ in range(h):
                bits = bin(int(lines[i], 16))[2:].zfill(len(lines[i]) * 4)
                rows.append([b == "1" for b in bits[:w]])
                i += 1
            if 32 <= cur["enc"] < 127 and chr(cur["enc"]) in CHARSET:
                cell = np.zeros((asc + desc, max(cur["adv"], xo + w)), bool)
                y0 = asc - (yo + h)
                if h and 0 <= y0 and y0 + h <= asc + desc and xo >= 0:
                    cell[y0:y0 + h, xo:xo + w] = np.array(rows, bool).reshape(h, w)
                    glyphs[chr(cur["enc"])] = cell
                elif h == 0 or chr(cur["enc"]) == " ":
                    glyphs[chr(cur["enc"])] = cell
            cur = None
    return BitFont(name, glyphs, asc + desc)


def load_ttf(name, path, size):
    """Rasterize a pixel TTF at its native pixel size with anti-aliasing off (ImageDraw.fontmode '1')."""
    font = ImageFont.truetype(str(path), size)
    asc, desc = font.getmetrics()
    cell_h = asc + desc + 2
    glyphs = {}
    for ch in CHARSET:
        adv = int(round(font.getlength(ch)))
        im = Image.new("L", (adv + size, cell_h), 0)
        d = ImageDraw.Draw(im)
        d.fontmode = "1"
        d.text((0, 0), ch, font=font, fill=255)
        a = np.asarray(im) > 127
        xs = np.where(np.any(a, 0))[0]
        w = max(adv, int(xs[-1]) + 1) if len(xs) else adv
        if w > 0:
            glyphs[ch] = a[:, :w].copy()
    return BitFont(name, glyphs, cell_h)


def load_fonts(font_dir):
    fonts = {}
    for name, info in FONT_SOURCES.items():
        if info["kind"] == "builtin":
            fonts[name] = load_builtin(name)
            continue
        path = Path(font_dir) / info["path"]
        if not path.exists():
            sys.exit(f"missing font {path}; run: python3 scripts/gen_rendered_text.py --download-fonts")
        fonts[name] = load_bdf(name, path) if info["kind"] == "bdf" else load_ttf(name, path, info["size"])
    return fonts


def download_fonts(font_dir):
    """Fetch the fonts into font_dir (X11 misc-fixed BDFs from the ucs-fonts tarball, OFL TTFs + OFL.txt from
    google/fonts) and write SHA256SUMS. The tarball is checked against X11_SHA256."""
    font_dir = Path(font_dir)
    xdir = font_dir / "x11-misc-fixed"
    xdir.mkdir(parents=True, exist_ok=True)
    tgz = xdir / "ucs-fonts.tar.gz"
    if not tgz.exists():
        print("fetching", X11_URL)
        tgz.write_bytes(urllib.request.urlopen(X11_URL, timeout=120).read())
    digest = hashlib.sha256(tgz.read_bytes()).hexdigest()
    if digest != X11_SHA256:
        sys.exit(f"{tgz}: sha256 {digest} != expected {X11_SHA256}")
    with tarfile.open(tgz) as tf:
        for m in X11_MEMBERS:
            data, dst = tf.extractfile(m).read(), xdir / m
            if not dst.exists() or dst.read_bytes() != data:
                dst.unlink(missing_ok=True)
                dst.write_bytes(data)
    for name, info in FONT_SOURCES.items():
        if info["kind"] != "ttf":
            continue
        path = font_dir / info["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        for url, dst in ((info["dl"], path), (info["dl"].rsplit("/", 1)[0] + "/OFL.txt", path.parent / "OFL.txt")):
            if not dst.exists():
                print("fetching", url)
                dst.write_bytes(urllib.request.urlopen(url, timeout=120).read())
    sums = []
    for p in sorted(font_dir.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(font_dir)}")
    (font_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(f"fonts ready in {font_dir} ({len(sums)} files, checksums in SHA256SUMS)")


# ------------------------------------------------------------------------------------------- strings

# Never rendered (case-insensitive; see is_heldout). Written to heldout_strings.txt.
HELDOUT = [
    "HELP!", "HELP", "QUIT", "JUMP", "MAGIC", "PIZZA", "ROBOT", "LASER", "GHOST", "TURBO", "ZAP!", "OUCH", "WOW!",
    "YES!", "NOPE", "RETRY", "BONUS", "QUEST", "VOTE", "TAXI", "CAFE", "MILK", "FROG", "KING", "NINJA", "PANDA",
    "Hello", "Sorry", "thanks", "pixel", "RUN!", "BOSS", "LOOT", "SHOP", "MENU", "SAVE", "FIRE!", "BYE",
    "37", "64", "256", "9000", "73%", "x7", "$15", "#8", "4:20", "+25", "LV9", "1UP",
    "QZX", "K7P", "BVM2", "wjf", "Xq9",
    "NICE TRY", "TOP TEN", "BE KIND", "GET READY", "TIME UP",
]

COMMON_WORDS = """
able acid act add age ago aid aim air all also and angel angry ant any apple arch arm army art ash ask atom aunt
auto away axe baby back bad bag bake ball band bank bar barn base bat bath bay beach bead beam bean bear beat bed
bee beef beep bell belt bench bend best bet big bike bill bin bird bit bite black blade blank blast blow blue boat
body boil bold bolt bomb bone book boom boot born both bowl box boy brain brake brave bread break brick bring
broom brown brush bud bug build bulb bull bump burn bus bush busy but buy cab cake calm camp can candy cap car
card care cart case cash cast cat catch cave chain chair chalk chat cheap check cheek chef chess chest chew chip
city clap claw clay clean clear clerk click cliff climb clip clock close cloth cloud clown club coal coast coat
code coin cold comb come cook cool copy coral cord core corn cost couch count cow crab crane crash crew crisp
crop cross crow crowd crown cry cube cup curl cut cute dad daily dance dark dash data date dawn day deal dear
deck deep deer desk dial dice did diet dig dime dine dirt dish dive dock dog doll done door dose dot down drag
draw dream dress drink drip drive drop drum dry duck dune dust each eagle ear early earth east easy eat echo
edge egg eight elbow elf else empty end enjoy enter equal even event ever evil exam extra eye face fact fade
fair fall fame fan far farm fast fat fear feed feel feet fence few field fig fight file fill film find fine fish
fist fit five fix flag flame flash flat flip float flock floor flour flow fly foam fog fold folk food fool foot
for fork form fort four fox frame free fresh from front fruit fuel full fun funny fur fuse gain game gap gas
gate gear gem get giant gift girl give glad glass glow glue goal goat gold golf gone good goose grab grade grain
grape grass gray great green grid grin grip group grow guard guess guest guide gum gym hair half hall hand hang
happy hard harp hat have hawk hay head heal heap hear heart heat heavy heel hen herb here hero hide high hike
hill hint hip hit hold hole home honey hood hook hop hope horn horse hose host hot hour house hug huge human
hunt hurry hut ice idea inch ink inn iron item ivy jam jar jaw jazz jeans jelly jet job jog join joke joy judge
juice july june just keen keep kept key kick kid kind kiss kite knee knife knot know lab lace lake lamb lamp
land lane large last late laugh law lawn lazy lead leaf lean learn left leg lemon lens less let lid life lift
light like lime line link lion lip list live load loaf lock log long look loop lost loud love low luck lucky
lunch mail main make male mall man many map march mark mask mat match math may meal mean meat meet melon melt
mess metal mice mild mile mind mine mint miss mist mix model mom money month moon more moss most moth mouse
mouth move much mud mug music must nail name nap navy near neat neck need nest net never news next nice night
nine noise none noon north nose note now nurse nut oak oar oat ocean odd off often oil old olive once one onion
only open orbit order other our out oven over owl own pace pack page pail pain paint pair palm pan pants paper
park part party pass past path paw pay pea peace peach peak pear pen pet phone photo piano pick pie piece pig
pile pill pilot pin pine pink pipe pit place plain plan plane plant plate play plot plug plum plus poem point
pole pond pony pool poor pop port pot pour power press price pride print prize proud pull pump punch pupil
puppy pure push put queen quick quiet quiz race rack radio raft rail rain ramp ranch range rare rat raw ray
reach read ready real red rent rest rice rich ride right ring ripe rise river road roar rock rod roll roof room
root rope rose round row royal rub rug rule rush rust sad safe sail salad salt same sand scale scarf score sea
seal seat seed seek sell send seven shade shake shape share shark sharp sheep shelf shell shine ship shirt shoe
shoot short show shut shy sick side sight silk silly sing sink sit six size skate skill skin skip skirt sky
sleep slice slide slim slip slow small smart smell smile smoke snack snail snake snow soap sock soda sofa soft
soil solid some song soon sort sound soup south space spare speak speed spell spend spice spin spoon sport spot
spray spy squad stack staff stage stair stamp stand star stay steam steel stem step stew stick still sting
stir stone stool stop store storm story stove straw street strong study stuff style sugar suit sun sunny super
sure swan sweet swim swing sword table tail take talk tall tank tap tape task taste tea teach team tear teeth
tell ten tent test than that the then they thick thin thing think third this three throw thumb tide tie tiger
tight tile time tiny tip tire toad toast today toe token tool tooth top torch total touch tour towel tower town
toy track trade trail train trap tray treat tree trick trip truck true trunk trust truth try tub tube tulip
tune turn twig twin two type uncle under unit until upon use usual van vase verb very vest video view vine
visit voice wade wage wagon wait wake walk wall wand want warm wash watch water wave wax way weak wear web week
well west wet whale what wheat wheel when white who whole wide wife wild will wind wing wink wire wise wish
with wolf wood wool word work world worm worry worth wrap wrist write wrong yard yarn year yell yes yet yoga
you young your zebra zero zinc zone zoo
""".split()

UI_WORDS = """
OK GO START EXIT SALE NEW WIN HP LV MP XP ON OFF YES NO PLAY PAUSE STOP BACK NEXT PREV DONE SKIP OPEN CLOSE EDIT
COPY PASTE UNDO REDO SEND LOGIN LOGOUT SIGN JOIN ADD DEL BUY SELL FREE HOT TOP MAX MIN MID END WARP LEVEL STAGE
ROUND SCORE COINS LIVES TIME BEST HIGH EASY HARD NORMAL MODE ITEMS ITEM EQUIP SKILL ATK DEF SPD STR INT DEX LUK
EXP GOLD GEMS KEYS MAP INFO HINT TIPS LOAD RESET CLEAR ERROR FAIL PASS LOCK ALERT WARN NOTE TODO PRO VIP BETA
DEMO TEST DEBUG ADMIN USER GUEST HOME SHARE LIKE STAR MUTE LOUD ZOOM FIT FULL AUTO FAST SLOW TURN MOVE GRAB USE
TALK LOOK TAKE GIVE DROP OPENS READY SET FIGHT MISS HIT CRIT COMBO PERFECT GREAT GOOD COOL NICE BAD LOSE DRAW
TIE KO BOOM BANG POW ZAP WHAM CRASH SPLAT OOPS HEY HI YO WOW LOL OMG ZZZ HA HAHA HMM UH OH AH YAY NOO WAIT
LOOK RUN HIDE DUCK FIRE ICE WIND EARTH DARK HOLY POISON HEAL REVIVE BUFF CAST BUYS RENT CASH PRICE DEAL OFFER
BARGAIN CHEAP PAID SOLD ONLY LAST LIMIT FRESH EXTRA MEGA SUPER ULTRA HYPER MINI BIG TINY PLUS LITE CAFE BAR INN
HOTEL BANK POST TOLL EXITS ENTER PUSH PULL UP DOWN LEFT RIGHT NORTH SOUTH EAST WEST IN OUT WC BUS STOPS ZOO
PARK POOL GYM LAB LIB ATM TV PC CD DVD USB HD FM AM VS AI UI OS ID IP URL APP WEB NET SMS TXT PDF PNG GIF JPG
ZIP MP3 SQL CSS XML API FPS RPG NPC DLC PVP CPU GPU RAM ROM LED LCD AC DC SOS RIP FAQ DIY ETA TBA TBD NA EU
JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC MON TUE WED THU FRI SAT SUN AM PM
""".split()

KEY_LABELS = ["Esc", "Tab", "Ctrl", "Alt", "Del", "Shift", "Enter", "Fn", "Cmd", "Home", "End", "Ins", "PgUp",
              "PgDn", "Caps", "Space", "Win", "Opt", "Num", "ESC", "TAB", "CTRL", "ALT", "DEL", "ENTER",
              "SHIFT"] + [f"F{i}" for i in range(1, 13)]

# two- and three-word phrases; drawn on one line when that fits, else one word per line
PHRASES = """
GAME OVER|YOU WIN|YOU LOSE|TRY AGAIN|LEVEL UP|PRESS START|NEW GAME|GOOD JOB|THE END|GO GO|HA HA|OH NO|NO WAY|
HI SCORE|ON SALE|BUY NOW|FOR SALE|KEEP OUT|ONE WAY|STAY OUT|ALL DAY|BIG WIN|LAST ONE|NEXT UP|SO COOL|GOOD LUCK|
WELL DONE|THANK YOU|COME IN|OPEN NOW|HOT DOG|ICE CREAM|TOP SCORE|NEW ITEM|GET IT|LETS GO|LOG IN|SIGN UP|LOG OUT|
TURN ON|PLAY NOW|GAME ON|ROLL DICE|YOUR TURN|MY TURN|END TURN|SAVE GAME|LOAD GAME|HIGH FIVE|GOOD DAY|BAD DAY|
NO ENTRY|WET PAINT|DEAD END|BUS STOP|CAR PARK|TEA TIME|NAP TIME|SHOW ME|TELL ME|FIND IT|USE KEY|OPEN DOOR|
GET GOLD|NEW HIGH|FULL HP|LOW HP|NO MP|ITEM GET|KEY GET|BOSS KEY|TOO BAD|SO SAD|NOT YET|ALL DONE|WAIT UP|HOLD ON|
COME ON|WAKE UP|GET UP|SIT DOWN|LOOK OUT|WATCH OUT|BE SAFE|STAY COOL|HAVE FUN|GOOD BOY|BAD DOG|NICE CAT|
RED TEAM|BLUE TEAM|TEAM A|TEAM B|PART 1|PART 2|SIDE A|SIDE B|ACT 1|ACT 2|DAY ONE|NO ONE|ANY KEY|FREE GIFT|
HALF OFF|BEST BUY|BIG SALE|HOT DEAL|NEW YEAR|GOOD BYE|I WIN|WE WIN|MY BAD|OH YES|OH MY|UH OH|HEY YOU|HI MOM|
YOU ARE HERE|DO NOT ENTER|I LOVE YOU|ONE TWO THREE|WE ARE OPEN|ON THE WAY|OUT OF MP|PUSH TO OPEN|TAP TO PLAY|
ALL YOU NEED|ONE MORE TRY|YOU DID IT|NOT FOR SALE|UP AND DOWN|RED OR BLUE|YES OR NO
""".replace("\n", "").split("|")
PHRASE_NUMBERED = ["LEVEL", "ROUND", "STAGE", "DAY", "ZONE", "ROOM", "PLAYER", "LAP", "WAVE", "FLOOR", "WORLD", "GATE"]

# random strings must not contain these (case-insensitive)
BLOCKLIST = ["ass", "sex", "fuc", "fuk", "fck", "shit", "sht", "cum", "tit", "dic", "dik", "coc", "cok", "nig", "fag",
             "kkk", "nazi", "rape", "slut", "hoe", "cnt", "cunt", "piss", "wtf", "gay", "jew", "kike", "spic", "anal",
             "anus", "poo", "god", "die", "kill", "dead", "666", "damn", "hell", "bich", "btch", "puss", "pus", "porn"]

UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWER = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
_HELD = {h.casefold() for h in HELDOUT}


def is_heldout(text):
    """True if `text` (lines joined by spaces) is, or contains, a held-out string, case-insensitively:
    the whole string, any whitespace-separated token (also with trailing !?. removed, so "x7!" is out because
    "x7" is), or (held-out strings of 4+ characters) any substring."""
    t = " ".join(text.split()).casefold()
    cands = [t] + t.split()
    if any(c in _HELD or c.rstrip("!?.") in _HELD for c in cands):
        return True
    return any(len(h) >= 4 and h in t for h in _HELD)


def is_blocked(text):
    t = text.casefold()
    return any(b in t for b in BLOCKLIST)


def rand_case(rng, word, weights=(50, 25, 25)):
    mode = rng.choices(["upper", "cap", "lower"], weights)[0]
    return word.upper() if mode == "upper" else word.capitalize() if mode == "cap" else word.lower()


def gen_single(rng):
    r = rng.random()
    if r < 0.42:
        return rng.choice(UPPER), "letter"
    if r < 0.70:
        return rng.choice(LOWER.replace("l", "")), "letter"  # a lone "l" cannot be told from 1 or I
    if r < 0.94:
        return rng.choice(DIGITS), "digit"
    return rng.choice("!?+#$%"), "symbol"


def gen_number(rng):
    r = rng.random()
    if r < 0.40:
        n = rng.choice([rng.randint(10, 99), rng.randint(10, 99), rng.randint(100, 999), rng.randint(1000, 9999),
                        rng.randint(0, 99999)])
        s = str(n)
        if rng.random() < 0.08:
            s = s.zfill(rng.choice([2, 3]))
        return s, "number"
    n = rng.choice([rng.randint(0, 9), rng.randint(1, 20), rng.randint(1, 99), rng.randint(1, 99), rng.randint(100, 999)])
    forms = [
        lambda: f"x{n}", lambda: f"{n}%", lambda: f"${n}", lambda: f"#{n}", lambda: f"+{n}", lambda: f"-{n}",
        lambda: f"{n}x", lambda: f"{rng.randint(1, 12)}:{rng.randint(0, 59):02d}", lambda: f"{n % 10}UP",
        lambda: f"LV{n % 100}", lambda: f"Lv.{n % 100}", lambda: f"No.{n % 100}", lambda: f"{n % 100}/{rng.randint(1, 99)}",
        lambda: f"{n % 100}.{rng.randint(0, 9)}", lambda: rng.choice(["1st", "2nd", "3rd", "4th", "5th", "6th", "7th",
                                                                    "8th", "9th", "10th", "1ST", "2ND", "3RD", "4TH"]),
        lambda: f"{n}HP", lambda: f"{n}G", lambda: f"{n}pt", lambda: f"{n}K", lambda: f"v{n % 10}.{rng.randint(0, 9)}",
        lambda: f"{n}+", lambda: f"${n % 100}.{rng.choice(['99', '50', '00', '25', '95'])}", lambda: f"-{n % 100}%",
        lambda: f"{n}px", lambda: f"{n}m", lambda: f"{n}kg", lambda: f"P{n % 10}", lambda: f"{n % 10}P",
        lambda: f"{n}F", lambda: f"x{n % 100}!", lambda: f"+{n}%", lambda: f"{n}MP", lambda: f"{n}XP",
    ]
    return rng.choice(forms)(), "number_format"


def gen_random(rng):
    n = rng.choices([2, 3, 4, 5, 6], [20, 30, 25, 15, 10])[0]
    mode = rng.choices(["upper", "lower", "mixed", "alnum_u", "alnum_l", "digits_mixed"], [34, 22, 12, 20, 6, 6])[0]
    low = LOWER.replace("l", "")  # without a word around it, "l" reads as 1 or I
    alpha = {"upper": UPPER, "lower": low, "mixed": UPPER + low, "alnum_u": UPPER + DIGITS,
             "alnum_l": low + DIGITS, "digits_mixed": UPPER + low + DIGITS}[mode]
    s = "".join(rng.choice(alpha) for _ in range(n))
    if s.isdigit():
        return s, "number"
    return s, "random"


def gen_word(rng):
    return rand_case(rng, rng.choice(COMMON_WORDS)), "word"


def gen_ui(rng):
    w = rand_case(rng, rng.choice(UI_WORDS), (72, 16, 12))
    if rng.random() < 0.10 and len(w) <= 5:
        w += rng.choice("!!?")
    return w, "ui_word"


def gen_key(rng):
    r = rng.random()
    if r < 0.45:
        return rng.choice(UPPER + DIGITS), "letter"
    return rng.choice(KEY_LABELS), "key_label"


def gen_phrase(rng):
    if rng.random() < 0.2:
        p = f"{rng.choice(PHRASE_NUMBERED)} {rng.randint(1, 99)}"
    else:
        p = rng.choice(PHRASES)
    if rng.random() < 0.25:
        p = " ".join(w.capitalize() for w in p.split()) if rng.random() < 0.6 else p.lower()
    return p, "phrase"


def gen_price(rng):
    r = rng.random()
    n = rng.choice([rng.randint(1, 9), rng.randint(1, 99), rng.randint(1, 999)])
    if r < 0.35:
        return f"${n}", "number_format"
    if r < 0.50:
        return f"${n % 100}.{rng.choice(['99', '50', '00', '25', '95'])}", "number_format"
    if r < 0.70:
        return rng.choice([f"{n % 100}%", f"-{n % 100}%"]), "number_format"
    if r < 0.80:
        return str(n), "number"
    return rng.choice(["SALE", "FREE", "NEW", "HOT", "SOLD", "DEAL", "BUY", "OFF", "GIFT", "CHEAP", "PRICE", "Sale",
                       "Free", "New", "sale", "50% OFF", "BEST", "2 FOR 1", "BIG SALE", "HALF OFF"]), "ui_word"


STRING_GENS = {"single": gen_single, "number": gen_number, "random": gen_random, "word": gen_word, "ui": gen_ui,
               "key": gen_key, "phrase": gen_phrase, "price": gen_price}


# ------------------------------------------------------------------------------------------- colours

# caption name -> a few RGB shades (the caption uses the name; the shade adds pixel variety)
PALETTE = {
    "black": [(0, 0, 0), (17, 17, 17), (24, 20, 37)],
    "dark gray": [(64, 64, 64), (74, 74, 84), (51, 51, 51)],
    "gray": [(128, 128, 128), (140, 140, 150)],
    "light gray": [(200, 200, 200), (211, 215, 220)],
    "white": [(255, 255, 255), (248, 248, 248)],
    "red": [(220, 40, 40), (200, 30, 45), (235, 60, 50)],
    "dark red": [(140, 20, 30), (120, 25, 25)],
    "orange": [(240, 140, 30), (255, 150, 40)],
    "yellow": [(250, 215, 40), (255, 230, 60)],
    "green": [(50, 170, 70), (60, 190, 90)],
    "dark green": [(20, 100, 50), (30, 90, 40)],
    "lime green": [(150, 220, 50), (170, 230, 70)],
    "teal": [(20, 140, 140), (0, 150, 136)],
    "cyan": [(60, 210, 230), (0, 200, 220)],
    "sky blue": [(110, 180, 245), (135, 200, 250)],
    "blue": [(40, 100, 220), (50, 120, 235)],
    "navy blue": [(20, 35, 95), (25, 40, 110)],
    "purple": [(130, 60, 190), (150, 80, 200)],
    "magenta": [(210, 50, 170), (225, 60, 190)],
    "pink": [(250, 140, 180), (255, 160, 200)],
    "brown": [(130, 80, 40), (120, 72, 48)],
    "tan": [(210, 170, 120), (200, 160, 110)],
    "cream": [(250, 240, 200), (245, 235, 210)],
}
NAMES = list(PALETTE)
DARKS = ["black", "dark gray", "navy blue", "dark red", "dark green", "brown"]
LIGHTS = ["white", "cream", "light gray", "yellow", "sky blue", "pink", "cyan", "lime green", "tan"]
PAPER = ["white", "white", "white", "cream", "light gray"]


def lum(rgb):
    def f(c):
        c /= 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (f(float(v)) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    la, lb = lum(a), lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def shade(rng, name):
    return rng.choice(PALETTE[name])


def darken(rgb, f=0.7):
    return tuple(int(v * f) for v in rgb)


def lighten(rgb, f=0.5):
    return tuple(int(v + (255 - v) * f) for v in rgb)


def pick(rng, names, avoid=(), against=None, min_ratio=0.0, favour=(), favour_p=0.0):
    """A (name, rgb) from `names`, not in `avoid`, with contrast >= min_ratio against each rgb in `against`."""
    for _ in range(60):
        pool = favour if (favour and rng.random() < favour_p) else names
        name = rng.choice(pool)
        if name in avoid:
            continue
        rgb = shade(rng, name)
        if against and any(contrast(rgb, a) < min_ratio for a in against):
            continue
        return name, rgb
    return None


# ------------------------------------------------------------------------------------------- shapes

_INSET = {0: [], 1: [1], 2: [2, 1], 3: [3, 2, 1], 4: [4, 2, 1, 1], 5: [5, 3, 2, 1, 1]}


def rect(x0, y0, x1, y1):
    m = np.zeros((S, S), bool)
    m[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
    return m


def rrect(x0, y0, x1, y1, r):
    """Rounded rectangle mask [x0,x1) x [y0,y1) with pixel-art corners of radius r."""
    m = rect(x0, y0, x1, y1)
    r = min(r, (x1 - x0) // 2, (y1 - y0) // 2, 5)
    for i, ins in enumerate(_INSET[r]):
        for y in (y0 + i, y1 - 1 - i):
            m[y, x0:x0 + ins] = False
            m[y, x1 - ins:x1] = False
    return m


def disc(cx, cy, rad):
    """Disc mask; cx, cy, rad may be half-integers (pixel centres are at integer + 0.5)."""
    yy, xx = np.mgrid[0:S, 0:S]
    return (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= rad ** 2


def pill(x0, y0, x1, y1):
    h = y1 - y0
    m = rect(x0 + h // 2, y0, x1 - h // 2, y1)
    return m | (disc(x0 + h / 2, y0 + h / 2, h / 2) | disc(x1 - h / 2, y0 + h / 2, h / 2))


def erode(m, n=1):
    for _ in range(n):
        p = np.pad(m, 1)
        m = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m


def dilate(m, diag=True):
    p = np.pad(m, 1)
    out = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    if diag:
        out = out | p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
    return out


def blank():
    return np.zeros((S, S, 4), np.uint8)


def paint(img, mask, rgb):
    img[mask] = (*rgb, 255)


def filled(img, mask, fill, outline=None):
    """Paint `mask` with `fill` and its 1 px edge with `outline`; returns the inside (mask minus edge)."""
    paint(img, mask, fill)
    if outline is None:
        return mask
    inner = erode(mask)
    paint(img, mask & ~inner, outline)
    return inner


def centre(lo, hi, size):
    return lo + (hi - lo - size) // 2


# ------------------------------------------------------------------------------------------- layouts
# Each layout takes (rng, tw, th, big) for a text raster of tw x th px and returns None if it cannot hold
# it, else a dict: img (RGBA canvas with the container), x, y (text position), inside (mask the text must stay
# in), fill (rgb behind the text, None = transparent), c (colour names for the caption), f (other facts).

def L_plain(rng, tw, th, big):
    if tw > S - 2 or th > S - 2:
        return None
    x, y = centre(0, S, tw), centre(0, S, th)
    if rng.random() < 0.3:  # off-centre now and then
        x += rng.randint(-min(2, x - 1), min(2, S - 1 - tw - x)) if x > 1 else 0
        y += rng.randint(-min(3, y - 1), min(3, S - 1 - th - y)) if y > 1 else 0
    return dict(img=blank(), x=x, y=y, inside=rect(1, 1, S - 1, S - 1), fill=None, c={}, f={})


def L_solid(rng, tw, th, big):
    if tw > S - 6 or th > S - 6:
        return None
    img = blank()
    fname, fill = pick(rng, NAMES)
    c = {"fill": fname}
    inside = rect(2, 2, S - 2, S - 2)
    if rng.random() < 0.25:
        oname, out = pick(rng, NAMES, avoid=(fname,), against=[fill], min_ratio=1.8)
        filled(img, rect(0, 0, S, S), fill, out)
        c["border"] = oname
    else:
        paint(img, rect(0, 0, S, S), fill)
    return dict(img=img, x=centre(0, S, tw), y=centre(0, S, th), inside=inside, fill=fill, c=c, f={})


def _container_colours(rng, light_p=0.5, fills=None):
    """(fill name, fill rgb, outline name, outline rgb): outline contrasts with the fill."""
    if fills is None:
        fills = PAPER if rng.random() < light_p else NAMES
    fname, fill = pick(rng, fills)
    oname, out = pick(rng, NAMES, avoid=(fname,), against=[fill], min_ratio=2.2, favour=DARKS, favour_p=0.7)
    return fname, fill, oname, out


def L_speech_bubble(rng, tw, th, big):
    padx, pady, tail = rng.choice([2, 2, 3]), rng.choice([1, 2, 2]), rng.choice([3, 4, 4, 5])
    w, h = tw + 2 * padx + 2, th + 2 * pady + 2
    if rng.random() < 0.35:  # roomier bubble
        w, h = min(max(w, rng.randint(22, 30)), 30), min(max(h, rng.randint(14, 22)), S - tail - 1)
    if w > S or h + tail > S or h < 7:
        return None
    x0, y0 = centre(0, S, w), centre(0, S, h + tail)
    r = rng.choice([2, 3, 3, 4]) if h >= 11 else 2
    body = rrect(x0, y0, x0 + w, y0 + h, r)
    side = rng.choice(["left", "right"])
    m = body.copy()
    base = x0 + r + rng.randint(1, 3) if side == "left" else x0 + w - r - rng.randint(1, 3) - tail
    for k in range(tail):  # right-angled tail hanging from the bottom edge, pointing to its side
        if side == "left":
            m[y0 + h + k, base:base + tail - k] = True
        else:
            m[y0 + h + k, base + k:base + tail] = True
    fname, fill, oname, out = _container_colours(rng, 0.75)
    img = blank()
    filled(img, m, fill, out)
    return dict(img=img, x=centre(x0, x0 + w, tw), y=centre(y0, y0 + h, th), inside=erode(body, 2), fill=fill,
                c={"fill": fname, "outline": oname}, f={"tail": side})


def L_button(rng, tw, th, big):
    padx, pady, bev = rng.choice([2, 3, 3, 4]), rng.choice([2, 2, 3]), rng.choice([1, 2, 2, 3])
    w, h = tw + 2 * padx + 2, th + 2 * pady + 2 + bev
    if rng.random() < 0.3:
        w = min(max(w, rng.randint(24, 30)), 30)
    if w > S or h > S:
        return None
    x0, y0 = centre(0, S, w), centre(0, S, h)
    fname, fill = pick(rng, [n for n in NAMES if n not in ("black",)])
    oname, out = pick(rng, NAMES, avoid=(fname,), against=[fill], min_ratio=2.0, favour=DARKS, favour_p=0.75)
    img = blank()
    body = rrect(x0, y0, x0 + w, y0 + h, rng.choice([2, 2, 3]))
    inner = filled(img, body, fill, out)
    low = inner & rect(0, y0 + h - 1 - bev, S, S)
    paint(img, low, darken(fill, 0.68))
    f = {"bevel": True}
    face = inner & ~low
    if rng.random() < 0.35 and h - bev >= th + 6:
        paint(img, face & rect(0, 0, S, y0 + 2) & erode(body, 1), lighten(fill, 0.45))
        f["highlight"] = True
        face = face & rect(0, y0 + 2, S, S)
    return dict(img=img, x=centre(x0, x0 + w, tw), y=centre(y0 + 1 + (1 if f.get("highlight") else 0), y0 + h - 1 - bev, th),
                inside=face & rect(x0 + 2, 0, x0 + w - 2, S),
                fill=fill, c={"fill": fname, "outline": oname}, f=f)


def L_sign(rng, tw, th, big):
    padx, pady = rng.choice([2, 2, 3]), rng.choice([2, 2, 3])
    w, h = tw + 2 * padx + 2, th + 2 * pady + 2
    if rng.random() < 0.4:
        w = min(max(w, rng.randint(22, 30)), 30)
    post = S - 1 - h - rng.choice([0, 0, 1, 2])
    post = min(post, rng.randint(6, 14))
    if w > S or post < 4:
        return None
    x0, y0 = centre(0, S, w), centre(0, S, h + post)
    wood = rng.random() < 0.6
    fname, fill = pick(rng, ["brown", "tan", "tan", "cream"] if wood else ["white", "yellow", "green", "red", "blue",
                                                                             "orange", "light gray", "dark green"])
    oname, out = pick(rng, DARKS, avoid=(fname,), against=[fill], min_ratio=2.0)
    pname, pcol = pick(rng, ["brown", "brown", "gray", "dark gray"], avoid=(fname,))
    img = blank()
    pw = rng.choice([2, 3, 4])
    px = centre(0, S, pw)
    paint(img, rect(px, y0 + h, px + pw, y0 + h + post), pcol)
    if pw >= 3:
        paint(img, rect(px + pw - 1, y0 + h, px + pw, y0 + h + post), darken(pcol, 0.7))
    board = rrect(x0, y0, x0 + w, y0 + h, rng.choice([0, 1, 1]))
    filled(img, board, fill, out)
    return dict(img=img, x=centre(x0, x0 + w, tw), y=centre(y0, y0 + h, th), inside=erode(board, 2), fill=fill,
                c={"fill": fname, "outline": oname, "post": pname}, f={"wood": fname in ("brown", "tan")})


def L_banner(rng, tw, th, big):
    pady = rng.choice([2, 2, 3])
    h = th + 2 * pady + 2
    notch = min(rng.choice([2, 3, 3, 4]), h // 2)
    w = tw + 2 * (notch + 2) + 2
    w = min(max(w, rng.choice([w, 28, 30, 32])), S)
    if tw + 2 * (notch + 2) + 2 > S or h > 24:
        return None
    x0, y0 = centre(0, S, w), centre(0, S, h)
    m = rect(x0, y0, x0 + w, y0 + h)
    for j in range(h):  # swallowtail ends
        d = int(round(notch * (1 - abs(j - (h - 1) / 2) / ((h - 1) / 2))))
        m[y0 + j, x0:x0 + d] = False
        m[y0 + j, x0 + w - d:x0 + w] = False
    fname, fill = pick(rng, [n for n in NAMES if n != "black"])
    oname, out = pick(rng, NAMES, avoid=(fname,), against=[fill], min_ratio=2.0, favour=DARKS, favour_p=0.7)
    img = blank()
    filled(img, m, fill, out)
    return dict(img=img, x=centre(x0, x0 + w, tw), y=centre(y0, y0 + h, th),
                inside=erode(m, 2) & rect(x0 + notch + 2, 0, x0 + w - notch - 2, S), fill=fill,
                c={"fill": fname, "outline": oname}, f={})


def L_badge(rng, tw, th, big):
    need = math.hypot(tw / 2.0, th / 2.0) + 2.6  # disc radius that keeps the text corners inside the rim
    d = int(math.ceil(2 * need))
    if d > S:
        return None
    d = min(max(d, rng.choice([d, d + 2, 24, 28, 30])), S)
    c0 = (S - d) // 2
    m = disc(c0 + d / 2.0, c0 + d / 2.0, d / 2.0)
    fname, fill = pick(rng, NAMES)
    oname, out = pick(rng, NAMES, avoid=(fname,), against=[fill], min_ratio=1.8, favour=DARKS + ["white", "yellow"],
                      favour_p=0.7)
    img = blank()
    filled(img, m, fill, out)
    return dict(img=img, x=centre(0, S, tw), y=centre(0, S, th), inside=erode(m, 2), fill=fill,
                c={"fill": fname, "outline": oname}, f={})


def L_label(rng, tw, th, big):
    padx, pady = rng.choice([1, 2, 2, 3]), rng.choice([1, 2, 2])
    w, h = tw + 2 * padx + 2, th + 2 * pady + 2
    if w > S or h > S:
        return None
    x0, y0 = centre(0, S, w), centre(0, S, h)
    fname, fill, oname, out = _container_colours(rng, 0.4)
    img = blank()
    m = rrect(x0, y0, x0 + w, y0 + h, rng.choice([0, 0, 1]))
    filled(img, m, fill, out)
    return dict(img=img, x=x0 + 1 + padx, y=y0 + 1 + pady, inside=erode(m, 1), fill=fill,
                c={"fill": fname, "outline": oname}, f={})


def L_price_tag(rng, tw, th, big):
    pady, padr = rng.choice([2, 2, 3]), rng.choice([2, 2, 3])
    h = th + 2 * pady + 2
    cut = min(rng.choice([3, 4, 4, 5]), (h - 2) // 2)
    lead = cut + 4  # pointed end + hole
    w = lead + tw + padr + 1
    if w > S or h > S or cut < 2:
        return None
    x0, y0 = centre(0, S, w), centre(0, S, h)
    m = rect(x0, y0, x0 + w, y0 + h)
    for i in range(cut):  # chamfer the two left corners into a point
        m[y0 + i, x0:x0 + cut - i] = False
        m[y0 + h - 1 - i, x0:x0 + cut - i] = False
    fname, fill, oname, out = _container_colours(rng, 0.35, None)
    img = blank()
    filled(img, m, fill, out)
    hy = y0 + h // 2 - 1
    hole = rect(x0 + cut, hy, x0 + cut + 2, hy + 2)
    img[hole] = 0  # punched hole (transparent)
    paint(img, dilate(hole, diag=False) & ~hole & erode(m), out)
    return dict(img=img, x=x0 + lead, y=centre(y0, y0 + h, th), inside=erode(m, 2) & rect(x0 + lead, 0, S, S),
                fill=fill, c={"fill": fname, "outline": oname}, f={})


def L_key(rng, tw, th, big):
    bev = rng.choice([2, 2, 3])
    side = rng.choice([14, 16, 18, 20, 22, 24, 26])
    w, h = max(tw + 6, side), max(th + 5 + bev, side)
    if w > S or h > S:
        return None
    x0, y0 = centre(0, S, w), centre(0, S, h)
    fname, fill = pick(rng, ["light gray", "white", "dark gray", "black", "gray", "cream"] if rng.random() < 0.75 else NAMES)
    oname, out = pick(rng, NAMES, avoid=(fname,), against=[fill], min_ratio=2.0, favour=["black", "dark gray", "gray"],
                      favour_p=0.8)
    img = blank()
    body = rrect(x0, y0, x0 + w, y0 + h, 2)
    inner = filled(img, body, fill, out)
    low = inner & rect(0, y0 + h - 1 - bev, S, S)
    edge = darken(fill, 0.65) if lum(fill) > 0.05 else lighten(fill, 0.3)
    paint(img, low, edge)
    face = inner & ~low
    return dict(img=img, x=centre(x0, x0 + w, tw), y=centre(y0 + 1, y0 + h - 1 - bev, th),
                inside=face & rect(x0 + 2, y0 + 2, x0 + w - 2, S), fill=fill, c={"fill": fname, "outline": oname}, f={})


def L_dialog_box(rng, tw, th, big):
    double = rng.random() < 0.5
    b = 2 if double else 1
    padx, pady = rng.choice([1, 2, 2]), rng.choice([1, 2, 2, 3])
    w = rng.choice([30, 32, 32]) if tw + 2 * (padx + b) <= 30 else S
    h = th + 2 * (pady + b)
    if rng.random() < 0.3:
        h = max(h, rng.randint(14, 20))
    if tw + 2 * (padx + b) > w or h > S:
        return None
    x0 = centre(0, S, w)
    pos = rng.choice(["centre", "centre", "bottom"])
    y0 = centre(0, S, h) if pos == "centre" else S - h - (S - w) // 2
    fname, fill = pick(rng, ["navy blue", "navy blue", "black", "blue", "dark gray", "white", "cream", "purple", "dark green", "brown"])
    oname, out = pick(rng, ["white", "white", "light gray", "yellow", "cyan", "black", "tan"], avoid=(fname,), against=[fill], min_ratio=3.0)
    img = blank()
    m = rrect(x0, y0, x0 + w, y0 + h, rng.choice([0, 1, 2]))
    c = {"fill": fname, "border": oname}
    if double:  # dark edge outside a light frame line
        dn, dk = pick(rng, DARKS, avoid=(fname, oname), against=[out], min_ratio=2.5) or ("black", (0, 0, 0))
        inner = filled(img, m, fill, dk)
        filled(img, inner, fill, out)
        c["edge"] = dn
    else:
        filled(img, m, fill, out)
    align = rng.choice(["left", "centre"])
    x = x0 + b + padx if align == "left" else centre(x0, x0 + w, tw)
    return dict(img=img, x=x, y=centre(y0, y0 + h, th), inside=erode(m, b + 1), fill=fill, c=c,
                f={"double": double, "pos": pos})


def L_chip(rng, tw, th, big):
    pady = rng.choice([2, 2, 3])
    h = th + 2 * pady + (th + 1) % 2  # odd height gives a clean pill end
    padx = h // 2 + rng.choice([0, 1, 1, 2]) - 1
    w = tw + 2 * padx + 2
    if w > S or h > 20 or padx < 3:
        return None
    x0, y0 = centre(0, S, w), centre(0, S, h)
    m = pill(x0, y0, x0 + w, y0 + h)
    fname, fill = pick(rng, NAMES)
    img = blank()
    c = {"fill": fname}
    if rng.random() < 0.45:
        oname, out = pick(rng, NAMES, avoid=(fname,), against=[fill], min_ratio=2.0, favour=DARKS, favour_p=0.6)
        filled(img, m, fill, out)
        c["outline"] = oname
        inside = erode(m, 2)
    else:
        paint(img, m, fill)
        inside = erode(m, 1)
    return dict(img=img, x=centre(x0, x0 + w, tw), y=centre(y0, y0 + h, th), inside=inside, fill=fill, c=c, f={})


def L_tile(rng, tw, th, big):
    side = rng.choice([26, 28, 30, 30, 32, 32])
    if tw > side - 6 or th > side - 6:
        return None
    x0 = (S - side) // 2
    m = rrect(x0, x0, x0 + side, x0 + side, rng.choice([2, 3, 4, 5]))
    fname, fill = pick(rng, NAMES)
    img = blank()
    c = {"fill": fname}
    if rng.random() < 0.5:
        oname, out = pick(rng, NAMES, avoid=(fname,), against=[fill], min_ratio=1.8, favour=DARKS, favour_p=0.7)
        filled(img, m, fill, out)
        c["outline"] = oname
    else:
        paint(img, m, fill)
    return dict(img=img, x=centre(0, S, tw), y=centre(0, S, th), inside=erode(m, 2), fill=fill, c=c, f={"side": side})


# layout -> (function, category, share of the set, string-kind weights, max lines)
LAYOUTS = {
    "plain": (L_plain, "text", 0.235, dict(single=14, number=18, random=30, word=22, ui=10, phrase=6), 3),
    "solid": (L_solid, "text", 0.065, dict(single=14, number=18, random=28, word=22, ui=12, phrase=6), 3),
    "speech_bubble": (L_speech_bubble, "pixel_art", 0.12, dict(single=4, number=6, random=22, word=30, ui=28, phrase=10), 2),
    "button": (L_button, "ui_element", 0.10, dict(single=5, number=8, random=22, word=20, ui=40, phrase=5), 2),
    "sign": (L_sign, "pixel_art", 0.06, dict(single=4, number=10, random=22, word=30, ui=26, phrase=8), 2),
    "banner": (L_banner, "pixel_art", 0.05, dict(single=2, number=12, random=24, word=30, ui=32), 1),
    "badge": (L_badge, "pixel_art", 0.065, dict(single=30, number=34, random=22, word=6, ui=8), 1),
    "label": (L_label, "pixel_art", 0.06, dict(single=4, number=18, random=28, word=26, ui=20, phrase=4), 2),
    "price_tag": (L_price_tag, "pixel_art", 0.04, dict(price=85, number=10, random=5), 1),
    "key": (L_key, "ui_element", 0.05, dict(key=88, random=6, number=6), 1),
    "dialog_box": (L_dialog_box, "ui_element", 0.065, dict(number=6, random=22, word=24, ui=22, phrase=26), 3),
    "chip": (L_chip, "ui_element", 0.04, dict(number=18, random=26, word=24, ui=32), 1),
    "tile": (L_tile, "icon", 0.05, dict(single=45, number=20, random=22, word=5, ui=8), 2),
}

# ------------------------------------------------------------------------------------------- captions

SYMBOL_NAMES = {"!": "exclamation mark", "?": "question mark", "+": "plus sign", "#": "hash sign", "$": "dollar sign",
                "%": "percent sign", "&": "ampersand"}


def an(word):
    return ("an " if word[0].lower() in "aeiou" else "a ") + word


def cap(s):
    return s[0].upper() + s[1:]


def text_terms(rng, text, kind):
    """(Q, R, units): the string as quoted in a caption, a noun phrase naming it without an article
    ("word START", "number 42", "lowercase letter a"), and the glyph noun (letters/digits/characters)."""
    plain_ok = all(c.isalnum() or c == " " for c in text)
    style = rng.choices(["bare", "single", "double"], [70, 15, 15] if plain_ok else [45, 30, 25])[0]
    if "'" in text and style == "single":
        style = "double"
    q = text if style == "bare" else f"'{text}'" if style == "single" else f'"{text}"'
    letters = text.replace(" ", "")
    if letters.isalpha():
        units = "letters"
    elif letters.isdigit():
        units = "digits"
    else:
        units = "characters"
    if kind == "letter" and text.isalpha():
        if text.isupper():
            noun = rng.choice(["letter", "letter", "letter", "capital letter", "uppercase letter"])
        else:
            noun = rng.choice(["lowercase letter", "lowercase letter", "lowercase letter", "small letter", "letter"])
    elif kind in ("digit", "letter"):
        noun = rng.choice(["number", "number", "digit", "numeral"])
    elif kind == "symbol":
        noun = SYMBOL_NAMES[text] + " symbol"
    elif kind == "number":
        noun = rng.choice(["number", "number", "number", "digits"])
    elif kind == "phrase":
        noun = rng.choice(["words", "words", "text"])
    elif kind in ("word", "ui_word"):
        noun = rng.choice(["word", "word", "word", "text"])
    elif letters.isalpha():
        noun = rng.choice(["letters", "letters", "text"]) if len(letters) > 1 else "letter"
    else:
        noun = rng.choice(["text", "text", "characters"])
    return q, f"{noun} {q}", units


def style_word(rng, font, scale):
    h = font.cap_h * scale
    opts = []
    if h <= 5:
        opts += ["tiny ", "small ", ""]
    elif h >= 14:
        opts += ["large ", "big ", "large ", ""]
    if font.info.get("bold"):
        opts += ["bold ", "bold ", ""]
    if scale >= 2 and not font.info.get("bold"):
        opts += ["blocky ", ""]
    return rng.choice(opts) if opts else ""


def units_word(rng, text, units):
    letters = text.replace(" ", "")
    if units == "letters" and len(letters) >= 2:
        if letters.isupper():
            return rng.choice(["pixel letters", "pixel letters", "capital letters", "uppercase pixel letters"])
        if letters.islower():
            return rng.choice(["pixel letters", "lowercase letters", "lowercase pixel letters"])
    return "pixel " + units


BG_END = ["Pixel art style on a transparent background.", "Pixel art on a transparent background.",
          "The background is transparent.", "Crisp pixel art on a transparent background.", "Transparent background.",
          "It sits on a transparent background.", "Simple pixel art style on a transparent background."]

SHORT = {
    "plain": ["{tc} pixel {R}{eff}", "{R} in {tc} pixel {units}{eff}", "pixel text {Q} in {tc}{eff}", "{tc} {R}, pixel font{eff}",
              "{R} in {style}{tc} pixel {units}{eff}", "{style}{tc} {R}{eff}, pixel art"],
    "solid": ["{tc} {R} on {afc} background", "{tc} pixel text {Q} on {fc}", "{R} in {tc} pixel {units} on {afc} background",
              "{style}{tc} {R} on {fc} background, pixel font"],
    "speech_bubble": ["speech bubble saying {Q}, pixel art", "pixel speech bubble saying {Q}", "pixel art speech bubble saying {Q}",
                      "{fc} speech bubble with the {R}, {oc} outline", "speech bubble that reads {Q}, pixel art",
                      "speech bubble with the text {Q}", "speech bubble with {tc} {R}, pixel art"],
    "button": ["{fc} button with the {R}", "{fc} {Q} button, pixel art", "{fc} button labelled {Q}, pixel art",
               "pixel art button that reads {Q}, {fc}", "{fc} game button with {tc} text {Q}", "{Q} button, {fc} with {oc} outline",
               "{fc} {Q} button, game UI"],
    "sign": ["{sign} that reads {Q}, pixel art", "signpost with the {R}, pixel art", "{sign} on a post saying {Q}",
             "pixel art {sign} with the text {Q}", "{sign} labelled {Q}, pixel art"],
    "banner": ["{fc} ribbon banner with the {R}", "{fc} banner that reads {Q}, pixel art", "pixel art banner saying {Q}",
               "{fc} ribbon with the text {Q}", "{Q} banner, {fc} ribbon, pixel art"],
    "badge": ["round {fc} badge with the {R}", "the {R} in {tc} pixel {units} on {afc} badge", "{fc} circle badge labelled {Q}, pixel art",
              "{fc} round badge with {tc} {R}", "{Q} on a round {fc} badge, {oc} rim"],
    "label": ["{fc} label with the {R}", "{fc} label that reads {Q}, {oc} border", "rectangular {fc} label with the text {Q}",
              "pixel art label saying {Q}", "{tc} {R} on {afc} label"],
    "price_tag": ["{fc} price tag with {Q}", "price tag that reads {Q}, pixel art", "{fc} price tag labelled {Q}",
                  "pixel art price tag with the text {Q}", "{fc} tag with {tc} {R}"],
    "key": ["{fc} keyboard key with the {R}", "keyboard key labelled {Q}, pixel art", "{Q} key, {fc} keyboard key",
            "{fc} keyboard key with {tc} {R}", "pixel art {Q} keyboard key"],
    "dialog_box": ["{fc} dialog box with the text {Q}", "pixel art dialog box that reads {Q}", "game dialog box saying {Q}, {fc} with {bc} border",
                   "{fc} text box with {tc} {R}", "RPG dialog box with the {R}"],
    "chip": ["{fc} pill-shaped chip labelled {Q}", "{fc} UI chip with the {R}", "small {fc} tag chip that reads {Q}",
             "rounded {fc} pill with the text {Q}", "{Q} chip, {fc} pill, pixel art"],
    "tile": ["{fc} rounded square with {tc} {R}", "{fc} tile with the {R}", "{tc} {R} on {afc} rounded square",
             "{fc} square icon with the text {Q}", "{fc} rounded square tile labelled {Q}"],
}

LONG = {
    "plain": ["{TP}{effl}{lines}.", "{TP}{effl}{lines}, with nothing else in the frame."],
    "solid": ["{TP} on {afc} background{border}{lines}.", "{TP}{lines}, set against a solid {fc} background{border}."],
    "speech_bubble": [
        "{Afc} speech bubble with {aoc} outline and a small tail at the lower {side}, containing {tp}{lines}.",
        "{Afc} speech bubble with {aoc} outline that says {QP}{lines}. Its tail points to the lower {side}.",
        "A pixel art speech bubble, {fc} with {aoc} outline, containing {tp}{lines}."],
    "button": [
        "{Afc} rounded button with {aoc} outline and a darker bottom edge, labelled {QP}{lines}.",
        "A rounded {fc} game button with {tp}{lines} and {aoc} outline.",
        "{TP} on {afc} button with {aoc} outline{hl}{lines}."],
    "sign": [
        "{Asign} on {apc} post with {aoc} outline, reading {QP}{lines}.",
        "A signpost with {afc} board and {apc} post, showing {tp}{lines}."],
    "banner": [
        "{Afc} ribbon banner with notched ends and {aoc} outline, bearing {tp}.",
        "{TP} across {afc} ribbon banner with swallowtail ends and {aoc} outline."],
    "badge": [
        "A round {fc} badge with {aoc} rim showing {tp}.",
        "{TP} centered on {afc} circular badge with {aoc} outline."],
    "label": [
        "{Afc} rectangular label with {aoc} border that reads {QP}{lines}.",
        "{TP} on a small {fc} label with {aoc} border{lines}."],
    "price_tag": [
        "{Afc} price tag with a pointed left end, a punched hole and {aoc} outline, marked {QP}.",
        "{TP} on {afc} price tag with a small hole near its pointed end and {aoc} outline."],
    "key": [
        "{Afc} keyboard key with {aoc} outline and a shaded lower edge, labelled {QP}.",
        "A square {fc} keyboard key showing {tp}, with {aoc} outline."],
    "dialog_box": [
        "{Afc} dialog box with {frame} containing {tp}{lines}.{pos}",
        "A game dialog box, {fc} with {frame}, that reads {QP}{lines}.{pos}"],
    "chip": [
        "{Afc} pill-shaped chip{outl} labelled {QP}.",
        "{TP} on a small {fc} pill-shaped UI chip{outl}."],
    "tile": [
        "{Afc} rounded square tile{outl} with {tp}{effl}{lines}.",
        "{TP}{effl} centered on {afc} rounded square{outl}{lines}."],
}


def make_captions(rng, rec):
    """Short and long caption + subject for one rendered record."""
    text, kind, layout, c, f = rec["text"], rec["kind"], rec["layout"], rec["c"], rec["f"]
    font, scale, eff = rec["font_obj"], rec["scale"], rec["effect"]
    q, r, units = text_terms(rng, text, kind)
    tc, fc, oc = rec["tc"], c.get("fill", ""), c.get("outline", "")
    style = style_word(rng, font, scale)
    uw = units_word(rng, text, units)
    single = len(text) == 1
    nl = rec["n_lines"]
    lines = "" if nl == 1 else f" on {('two', 'three')[nl - 2]} lines"
    if single:
        tp = f"{an(style + tc)} pixel {r}"
        qp = f"{q} in {style}{tc}"
    else:
        tp = f"the {r} in {style}{tc} {uw}"
        qp = f"{q} in {style}{tc} {uw}"
    effs = effl = ""
    if eff:
        ename = "outline" if eff[0] == "outline" else "drop shadow"
        effs = f", {eff[1]} outline" if eff[0] == "outline" else f" with {eff[1]} shadow"
        effl = f" with {an(eff[1])} {ename}"
    wood = f.get("wood")
    sign = rng.choice(["wooden sign", f"{fc} wooden sign"]) if wood else f"{fc} sign"
    if "edge" in c:
        frame = rng.choice([f"{an(c['border'])} border and {an(c['edge'])} outer edge", f"a double {c['border']} and {c['edge']} border"])
    else:
        frame = f"{an(c.get('border', 'x'))} border"
    slots = dict(
        Q=q, R=r, units=units, tc=tc, fc=fc, oc=oc, bc=c.get("border", ""), style=style, eff=effs, effl=effl,
        afc=an(fc) if fc else "", Afc=cap(an(fc)) if fc else "", aoc=an(oc) if oc else "", apc=an(c.get("post", "x")),
        TP=cap(tp), tp=tp, QP=qp, lines=lines, side=f.get("tail", ""), sign=sign, Asign=cap(an(sign)), frame=frame,
        hl=" and a lighter top highlight" if f.get("highlight") else "",
        border=f" with {an(c['border'])} border" if layout == "solid" and "border" in c else "",
        outl=f" with {an(oc)} outline" if oc else "",
        pos=" The box sits at the bottom of the frame." if f.get("pos") == "bottom" else "",
    )
    short = rng.choice(SHORT[layout]).format(**slots)
    if nl > 1 and rng.random() < 0.5:
        short += f", {('two', 'three')[nl - 2]} lines"
    long_ = rng.choice(LONG[layout]).format(**slots)
    if layout == "solid":
        long_ += rng.choice([" The background fills the whole frame.", " Pixel art style, filling the frame.", " Pixel font style."])
    elif layout == "tile" and f.get("side") == S:
        long_ += rng.choice([" The tile fills the frame.", " Pixel art style, filling the frame."])
    else:
        long_ += " " + rng.choice(BG_END)
    container = {"plain": "", "solid": "", "speech_bubble": " speech bubble", "button": " button", "sign": " sign",
                 "banner": " banner", "badge": " badge", "label": " label", "price_tag": " price tag", "key": " key",
                 "dialog_box": " dialog box", "chip": " chip", "tile": " tile"}[layout]
    subject = (r if not container else text + container) if layout not in ("plain", "solid") else \
        (r if kind in ("letter", "digit", "number", "symbol") else f"{text} text")
    return " ".join(short.split()), " ".join(long_.split()), subject


# ------------------------------------------------------------------------------------------- rendering

EFFECT_LAYOUTS = ("plain", "solid", "tile")


def place(mask, x, y):
    """Text mask on the canvas, or None if any of it would fall outside (never clip)."""
    h, w = mask.shape
    if x < 0 or y < 0 or x + w > S or y + h > S:
        return None
    m = np.zeros((S, S), bool)
    m[y:y + h, x:x + w] = mask
    return m


def effect_ok(kind, font, scale, text):
    """Outlines and drop shadows are only used where the glyphs stay readable (judged on the contact sheets):
    never on fonts with a cap height under 7 px, at any scale (their 1-cell counters and gaps fill up and the
    glyph turns into stripes or a blob), and never on strings with symbols. Outlines: bold fonts, or 1 px-stroke
    fonts from 13 px. Shadows: text of 9+ px."""
    if font.cap_h < 7 or any(ch in "#%&@$*" for ch in text):
        return False
    if kind == "shadow":
        return font.cap_h * scale >= 9
    return bool(font.info.get("bold")) or (scale == 1 and font.cap_h >= 13)


def render_one(rng, layout, text, kind, fonts):
    """Try to draw `text` in `layout`; returns a record dict (with img) or None if nothing legible fits."""
    fn, category, _, _, max_lines = LAYOUTS[layout]
    words = text.split()
    options = [[text]] + ([words] if 1 < len(words) <= max_lines else [])
    if len(options) == 2 and rng.random() < 0.6:
        options.reverse()
    effect_kind = None
    if layout in EFFECT_LAYOUTS:
        effect_kind = rng.choices([None, "outline", "shadow"], [66, 17, 17])[0]
    names = list(fonts)
    rng.shuffle(names)
    align = "left" if (len(options[0]) > 1 and rng.random() < 0.25) else "center"
    for lines in options:
        for name in names:
            font = fonts[name]
            if not all(font.can_render(t) for t in lines):
                continue
            scales = [s for n, s in FONT_VARIANTS if n == name]
            if rng.random() < 0.6:
                scales.sort(reverse=True)
            else:
                rng.shuffle(scales)
            for scale in scales:
                ek = effect_kind
                if ek and not effect_ok(ek, font, scale, text):
                    ek = None
                extra = 1 if (ek == "outline" and scale == 1) else 0
                gap = scale * (1 if font.cap_h < 7 else rng.choice([1, 2]))
                mask = font.render(lines, scale, extra, gap, align)
                if mask is None:
                    continue
                th, tw = mask.shape
                mx = my = 0
                if ek == "outline":
                    mx = my = 2
                elif ek == "shadow":
                    mx = my = 1 if scale <= 2 else 2
                if tw + mx > S or th + my > S:
                    continue
                lay = fn(rng, tw + mx, th + my, scale > 1)
                if lay is None:
                    continue
                rec = compose(rng, lay, mask, ek, font, scale)
                if rec is None:
                    continue
                rec.update(text=text, lines=lines, kind=kind, layout=layout, category=category, font=name,
                           font_obj=font, scale=scale, n_lines=len(lines), align=align if len(lines) > 1 else None)
                return rec
    return None


def compose(rng, lay, mask, effect_kind, font, scale):
    """Draw the text (and outline/shadow) into the layout and verify it: fully on the canvas, inside the
    container's interior, enough contrast, and the final pixels of the text colour are exactly the glyph mask."""
    img, fill = lay["img"].copy(), lay["fill"]
    height = font.cap_h * scale
    # thin 1 px strokes need more contrast than bold or upscaled glyphs
    min_ratio = 4.5 if height < 7 else (3.5 if (scale > 1 or font.info.get("bold")) else 4.0)
    against = [fill] if fill is not None else None
    got = pick(rng, NAMES, against=against, min_ratio=min_ratio if against else 0.0,
               favour=["black", "white"], favour_p=0.45 if against else 0.35)
    if got is None:
        return None
    tc, trgb = got
    x, y = lay["x"], lay["y"]
    effect = None
    emask = None
    if effect_kind == "outline":
        tm = place(mask, x + 1, y + 1)
        if tm is None:
            return None
        emask = dilate(tm) & ~tm
    elif effect_kind == "shadow":
        tm = place(mask, x, y)
        off = 1 if scale <= 2 else 2  # keep the shadow thinner than the strokes
        sm = place(mask, x + off, y + off)
        if tm is None or sm is None:
            return None
        emask = sm & ~tm
    else:
        tm = place(mask, x, y)
        if tm is None:
            return None
    if emask is not None:
        avoid = (tc,) + tuple(lay["c"].values())
        got = pick(rng, NAMES, avoid=avoid, against=[trgb], min_ratio=5.5 if effect_kind == "outline" else 4.0,
                   favour=DARKS + ["white", "gray"], favour_p=0.6)
        if got is None:
            return None
        if effect_kind == "shadow" and lum(got[1]) >= lum(trgb):
            return None  # a drop shadow lighter than the text reads as a second copy of the text
        effect = (effect_kind, got[0])
        paint(img, emask, got[1])
        if not (emask <= lay["inside"]).all():
            return None
    if not (tm <= lay["inside"]).all():
        return None  # would touch or cross the container edge
    paint(img, tm, trgb)
    # final check on the composited pixels: text colour appears exactly where the glyphs are
    same = (img[..., :3] == np.array(trgb, np.uint8)).all(-1) & (img[..., 3] == 255)
    if not np.array_equal(same & lay["inside"], tm) or int(tm.sum()) != int(mask.sum()):
        return None
    if fill is not None and contrast(trgb, fill) < min_ratio:
        return None
    img[img[..., 3] == 0] = 0
    ys, xs = np.where(tm)
    return dict(img=img, tc=tc, c=lay["c"], f=lay["f"], effect=effect, opaque=fill is not None and bool((img[..., 3] == 255).all()),
                bbox=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], text_px=int(tm.sum()))


def sidecar(rec, short, long_, subject, seed):
    info = rec["font_obj"].info
    fl = info["licence"]
    return {
        "source": SOURCE,
        "license": "CC0-1.0",
        "attribution": f"Generated by scripts/gen_rendered_text.py (image: CC0-1.0). Font: {info['credit']}; "
                       f"font licence: {fl}; {info['url']}",
        "origin": f"generated:gen_rendered_text.py:seed={seed}",
        "name": rec["text"],
        "tags": ["rendered_text", rec["layout"], rec["font"]],
        "text": rec["text"],
        "lines": rec["lines"],
        "layout": rec["layout"],
        "string_kind": rec["kind"],
        "font": rec["font"],
        "font_scale": rec["scale"],
        "font_license": fl,
        "font_source": info["url"],
        "text_color": rec["tc"],
        "colors": rec["c"],
        "effect": list(rec["effect"]) if rec["effect"] else None,
        "text_bbox": rec["bbox"],
        "description": long_,
        "verbose_description": short,
        "quality": 5 if rec["font_obj"].cap_h * rec["scale"] >= 7 else 4,
        "category": rec["category"],
        "subject": subject,
        "has_text": True,
        "fragment": False,
        "caption_source": "generated",
    }


def generate(args, fonts):
    rng = random.Random(args.seed)
    quotas = {k: int(round(args.count * v[2])) for k, v in LAYOUTS.items()}
    quotas["plain"] += args.count - sum(quotas.values())
    seen_keys, seen_pixels, records = set(), set(), []
    rejects = Counter()
    for layout, quota in quotas.items():
        kinds, weights = zip(*LAYOUTS[layout][3].items())
        made = tries = 0
        while made < quota:
            tries += 1
            if tries > 400 * quota + 2000:
                sys.exit(f"{layout}: gave up after {tries} tries ({made}/{quota})")
            text, kind = STRING_GENS[rng.choices(kinds, weights)[0]](rng)
            if is_heldout(text):
                rejects["heldout"] += 1
                continue
            if kind == "random" and is_blocked(text):
                rejects["blocklist"] += 1
                continue
            rec = render_one(rng, layout, text, kind, fonts)
            if rec is None:
                rejects["no_fit"] += 1
                continue
            key = (text, layout, rec["font"], rec["tc"], tuple(sorted(rec["c"].items())))
            ph = hashlib.blake2b(rec["img"].tobytes(), digest_size=16).digest()
            if key in seen_keys or ph in seen_pixels:
                rejects["duplicate"] += 1
                continue
            seen_keys.add(key)
            seen_pixels.add(ph)
            short, long_, subject = make_captions(rng, rec)
            assert text in short and text in long_, (text, short, long_)
            rec["side"] = sidecar(rec, short, long_, subject, args.seed)
            rec["rel"] = f"{layout}/{len(records):06d}.png"
            records.append(rec)
            made += 1
        print(f"  {layout:14s} {made:6d} images ({tries} tries)", flush=True)
    return records, rejects


# ------------------------------------------------------------------------------------------- output

MARKER = ".gen_rendered_text"
INDEX_COLS = ["dst", "src", "group", "source", "caption_source", "category", "quality",
              "text", "layout", "font", "font_scale", "n_chars", "n_lines", "short_caption"]


def contact_sheet(recs, path, cols=12, scale=4, title=None):
    """Nearest-neighbour upscaled grid on a checkerboard, with the expected string under each cell."""
    cell, lab = S * scale, 13
    rows = (len(recs) + cols - 1) // cols
    top = 14 if title else 0
    sheet = Image.new("RGB", (cols * (cell + 2) + 2, top + rows * (cell + lab + 2) + 2), (30, 30, 34))
    d = ImageDraw.Draw(sheet)
    if title:
        d.text((3, 1), title, fill=(255, 255, 160))
    yy, xx = np.mgrid[0:cell, 0:cell]
    even = ((yy // 8 + xx // 8) % 2 == 0)[..., None]
    bgs = {k: Image.fromarray(np.where(even, a, b).astype(np.uint8).repeat(3, 2), "RGB")
           for k, (a, b) in {"mid": (150, 118), "dark": (58, 40), "light": (236, 208)}.items()}
    for i, rec in enumerate(recs):
        x, y = 2 + (i % cols) * (cell + 2), top + 2 + (i // cols) * (cell + lab + 2)
        arr = rec["img"]
        im = Image.fromarray(arr, "RGBA").resize((cell, cell), Image.NEAREST)
        # Sprites whose silhouette is neither near-black nor near-white are hard to see on the grey
        # checkerboard: use a dark or light one for those cells (the sprite itself is unchanged).
        opaque = arr[..., 3] > 0
        edge = opaque & dilate(~opaque)
        key = "mid"
        if edge.any():
            grey = arr[..., :3][edge].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
            tone = float(np.median(grey))
            if 40 < tone < 225:
                key = "dark" if tone >= 110 else "light"
        tile = bgs[key].copy()
        tile.paste(im, (0, 0), im)
        sheet.paste(tile, (x, y))
        d.text((x + 1, y + cell + 1), " / ".join(rec["lines"])[:20], fill=(235, 235, 235))
    sheet.save(path)


def write_sheets(records, out, seed):
    rng = random.Random(seed + 1)  # separate stream: sheets never change the dataset
    sheets = out / "_sheets"
    sheets.mkdir(exist_ok=True)

    def sample(recs, n=96):
        recs = list(recs)
        return recs if len(recs) <= n else rng.sample(recs, n)

    contact_sheet(sample(records), sheets / "overview.png", title="overview: random sample of the whole set")
    for layout in LAYOUTS:
        contact_sheet(sample(r for r in records if r["layout"] == layout), sheets / f"layout_{layout}.png",
                      title=f"layout: {layout}")
    for name in FONT_SOURCES:
        recs = [r for r in records if r["font"] == name]
        contact_sheet(sample(recs), sheets / f"font_{name}.png", title=f"font: {name} (all scales)")
        small = [r for r in recs if r["scale"] == 1]
        if small:
            contact_sheet(sample(small), sheets / f"font_{name}_scale1.png", title=f"font: {name}, scale 1 only")
    contact_sheet(sample(r for r in records if r["n_lines"] > 1), sheets / "multiline.png", title="two and three lines")
    contact_sheet(sample(r for r in records if r["effect"]), sheets / "effects.png", title="outline / shadow")
    contact_sheet(sample(r for r in records if len(r["text"]) >= 6), sheets / "long_strings.png", title="6+ characters")
    contact_sheet(sample(r for r in records if r["text"] != r["text"].upper()), sheets / "lowercase.png",
                  title="strings containing lowercase")


def table(counter, head, order=None):
    keys = order or [k for k, _ in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))]
    rows = [f"| {head} | images |", "|---|---:|"] + [f"| {k} | {counter[k]:,} |" for k in keys if k in counter]
    return "\n".join(rows)


def write_readme(out, records, stats, args, fonts):
    by_len = Counter(len(r["text"]) for r in records)
    used = [n for n in FONT_SOURCES if stats["font"].get(n)]
    font_rows = ["| font | licence | source | images | notes |", "|---|---|---|---:|---|"]
    for n in used:
        i, f = FONT_SOURCES[n], fonts[n]
        note = f"cap height {f.cap_h} px" + ("" if f.has_lower else ", uppercase only")
        if f.dropped:
            note += f"; ambiguous glyphs not used: {''.join(sorted(set(f.dropped) - set(LOWER))) or 'lowercase (small caps)'}"
        if f.avoided:
            av = "".join(c for c in f.avoided if c not in LOWER or f.has_lower)
            note += f"; hard-to-read glyphs not used: {av}" + ("" if f.has_lower else " and all lowercase")
        font_rows.append(f"| {n} | {i['licence']} | {i['url']} | {stats['font'][n]:,} | {i['credit']}; {note} |")
    ex = "\n".join(f"- `{r['side']['verbose_description']}`  \n  {r['side']['description']}"
                   for r in random.Random(args.seed + 2).sample(records, min(8, len(records))))
    text = f"""# rendered_text: generated text renders for sprite32

{len(records):,} images, 32x32 RGBA PNG (alpha is 0 or 255, transparent pixels are RGB 0), each a short string drawn
with a bitmap/pixel font without anti-aliasing, alone or inside a simple container. Every caption (short and long)
contains the exact string, with exact case and punctuation, because the strings are rendered by the generator.
{len(set(r['text'] for r in records)):,} distinct strings. Generated by `scripts/gen_rendered_text.py --seed {args.seed} --count {args.count}`.

## Files

- `images/<layout>/<nnnnnn>.png` and `<nnnnnn>.png.json`: image + sidecar. Sidecar keys follow data_v12:
  `description` = LONG caption, `verbose_description` = SHORT caption (the inverted names hf_data.py expects),
  `quality`, `category`, `subject`, `has_text` (true), `fragment` (false), `caption_source` ("generated"),
  `source` ("rendered_text"), `license` (CC0-1.0 for the generated image), `attribution`. Extra keys: `text` (exact
  string; lines joined by one space), `lines`, `layout`, `string_kind`, `font`, `font_scale`, `font_license`,
  `font_source`, `text_color`, `colors`, `effect`, `text_bbox`.
- `_index.tsv`: one row per image; the first seven columns are those of `data_v12/_index.tsv`
  (`dst` is the path the image would have inside a training tree: `rendered_text/<layout>/<nnnnnn>.png`).
- `_stats.json`: the counts below, machine-readable. `heldout_strings.txt`: strings that never appear.
- `_sheets/`: contact sheets (4x nearest-neighbour, expected string under each cell; the checkerboard behind a
  cell is dark, grey or light, whichever contrasts with the sprite's edge colour).
- `_fonts/`: the downloaded fonts with their licence files and `SHA256SUMS`.

## Counts

{table(stats['layout'], 'layout', list(LAYOUTS))}

{table(stats['font_variant'], 'font x scale')}

{table(Counter({str(k): v for k, v in by_len.items()}), 'string length (characters, spaces included)', [str(k) for k in sorted(by_len)])}

{table(stats['string_kind'], 'string kind')}

{table(stats['n_lines'], 'lines')}

{table(stats['category'], 'category')}

{table(stats['quality'], 'quality (5 = cap height >= 7 px, 4 = smaller text)')}

{table(stats['effect'], 'text effect')}

## Fonts

{chr(10).join(font_rows)}

The generated images are CC0-1.0. The OFL fonts are used only to rasterise text; the font files in `_fonts/`
keep their own licence (`OFL.txt` next to each), and are not part of the training images.

## Held-out strings

`heldout_strings.txt` lists {len(HELDOUT)} strings, one per line with no header or comment lines (including
`HELP!`), that are never rendered, compared
case-insensitively: not as the whole string, not as a whitespace-separated token (also not with `!`, `?` or `.`
appended), and (for held-out strings of 4+ characters) not as a substring of any rendered string. Note that a
held-out string with punctuation, such as `YES!` or `WOW!`, does not hold out the bare word.

## Checks done by the generator

A render is kept only if the text mask lies completely on the canvas (never clipped), every text pixel is inside the
container's interior (not touching its outline), the text/background contrast ratio (WCAG) is at least 4.5 for text
under 7 px tall, 4.0 for thin text of 7+ px at scale 1, and 3.5 for bold or upscaled text, and in the final
composited image the pixels of the text colour inside the container are exactly the glyph mask. Outlines and drop
shadows are never used with fonts whose cap height is under 7 px (at any scale) or on strings containing `#%&@$*`.
Outlines (1 px): bold fonts or 1 px-stroke fonts of 13+ px, contrast with the text at least 5.5. Drop shadows: text
of 9+ px, darker than the text, contrast at least 4.0. Characters whose glyph is identical to another character's in
a font (e.g. small-caps lowercase and the zero in Silkscreen) are not used with that font, nor are the glyphs listed
as hard to read in the font table; a lone lowercase `l` is never rendered. No two images share (string, layout,
font, colour names), and no two images are pixel-identical. There is no OCR read-back: legibility was judged by eye
on the contact sheets.

## Example captions

{ex}

## Regenerate

```sh
python3 scripts/gen_rendered_text.py --download-fonts     # only if _fonts/ is missing
python3 scripts/gen_rendered_text.py --seed {args.seed} --count {args.count} --force
```

Same seed and count give the same images and captions (Pillow {Image.__version__}, numpy {np.__version__} here).
`--force` replaces `images/`, `_sheets/`, `_index.tsv`, `_stats.json`, `README.md` and `heldout_strings.txt` in
this directory only.

## Add to a training set

Copy (or hardlink) `images/` into the training tree as a source directory named `rendered_text`, e.g.
`cp -R data_gen/rendered_text/images <data_dir>/rendered_text`. Do not link this whole directory: `hf_data.py` scans
every PNG under `data_dir`, and `_sheets/` must stay out of it. A symlinked directory is not followed by
`Path.rglob` on older Pythons, so copy rather than symlink. Notes:

- Append the rows of `_index.tsv` to that tree's `_index.tsv` if it has one.
- `hf_data.split_dataset_paths` shuffles all paths with a fixed seed, so adding a source changes which images are in
  the validation split.
- Captions are new, so a text cache (`scripts/build_text_cache.py`) has to be rebuilt or extended.
- The source name for `source_weights` / source balancing is `rendered_text`.
"""
    (out / "README.md").write_text(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--count", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--font-dir", default=None, help="default: <out>/_fonts, else data_gen/rendered_text/_fonts")
    ap.add_argument("--download-fonts", action="store_true", help="fetch the fonts into --font-dir and exit")
    ap.add_argument("--force", action="store_true", help="replace an earlier run in --out")
    ap.add_argument("--no-sheets", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    font_dir = Path(args.font_dir) if args.font_dir else (out / "_fonts" if (out / "_fonts").exists() or args.download_fonts
                                                         else DEFAULT_OUT / "_fonts")
    if args.download_fonts:
        download_fonts(font_dir)
        return
    fonts = load_fonts(font_dir)
    images = out / "images"
    if images.exists() and any(images.iterdir()):
        if not args.force:
            sys.exit(f"{images} is not empty; pass --force to replace it")
        if not (out / MARKER).exists():
            sys.exit(f"{out} was not made by this script (no {MARKER}); refusing to replace anything in it")
        shutil.rmtree(images)
        if (out / "_sheets").exists():
            shutil.rmtree(out / "_sheets")
    out.mkdir(parents=True, exist_ok=True)
    (out / MARKER).write_text("made by scripts/gen_rendered_text.py; images/ and _sheets/ here are replaced by --force\n")

    print(f"generating {args.count:,} images (seed {args.seed})")
    records, rejects = generate(args, fonts)
    assert not any(is_heldout(r["text"]) for r in records)

    stats = defaultdict(Counter)
    try:
        src_root = images.resolve().relative_to(REPO)
    except ValueError:
        src_root = images
    with open(out / "_index.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(INDEX_COLS)
        for rec in records:
            dst = images / rec["rel"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rec["img"], "RGBA").save(dst, format="PNG")
            side = rec["side"]
            Path(f"{dst}.json").write_text(json.dumps(side, ensure_ascii=False))
            w.writerow([f"{SOURCE}/{rec['rel']}", f"{src_root}/{rec['rel']}", "generated", SOURCE, "generated",
                        side["category"], side["quality"], rec["text"], rec["layout"], rec["font"], rec["scale"],
                        len(rec["text"]), rec["n_lines"], side["verbose_description"]])
            stats["layout"][rec["layout"]] += 1
            stats["font"][rec["font"]] += 1
            stats["font_variant"][f"{rec['font']} x{rec['scale']}"] += 1
            stats["string_length"][len(rec["text"])] += 1
            stats["string_kind"][rec["kind"]] += 1
            stats["n_lines"][rec["n_lines"]] += 1
            stats["category"][side["category"]] += 1
            stats["quality"][side["quality"]] += 1
            stats["effect"][rec["effect"][0] if rec["effect"] else "none"] += 1
    stats["rejected_tries"] = rejects
    summary = {"total": len(records), "distinct_strings": len({r["text"] for r in records}), "seed": args.seed,
               "count": args.count}
    summary.update({k: {str(a): b for a, b in sorted(v.items(), key=lambda kv: str(kv[0]))} for k, v in stats.items()})
    (out / "_stats.json").write_text(json.dumps(summary, indent=1) + "\n")
    # One string per line and no header: some entries (e.g. "#8") start with "#", so there are no comment lines.
    (out / "heldout_strings.txt").write_text("\n".join(HELDOUT) + "\n")
    if not args.no_sheets:
        write_sheets(records, out, args.seed)
    write_readme(out, records, stats, args, fonts)
    print(f"{len(records):,} images -> {images}  (rejected tries: {dict(rejects)})")


if __name__ == "__main__":
    main()
