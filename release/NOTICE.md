# Notice on the training data's licences

This says what the model was trained on and which conditions those works carry. It is not legal advice, and
licence checking was best effort (`DATASHEET.md`).

The model weights do not contain the training images, and the images are not distributed with the model.
Whether training on a work makes the weights or the outputs a derivative of it is not settled, and differs
by country. This file therefore states the conditions the sources carry and leaves the conclusion to the user.

## Attribution

149,929 training images are under attribution licences (CC BY 3.0 / 4.0, OGA-BY) and 155,562 under permissive
licences that ask for their notice to be kept. `attribution.tsv.gz` gives the licence, author, title and URL
of every training image, and `SOURCES.md` lists each source with its licence. Together they are the credit
for these works.

## Share-alike and copyleft

118,466 training images (17.9%) are under share-alike or copyleft licences. The larger groups:

| Source | Licence | Images |
|---|---|---:|
| Space Station 13 codebases (tgstation, vgstation13, paradise, shiptest) | CC BY-SA 3.0 | 34,223 |
| OpenGameArt entries under CC BY-SA 3.0 / 4.0 | CC BY-SA | 15,358 |
| Cataclysm: DDA tilesets | CC BY-SA 3.0 / 4.0 | 12,734 |
| Papirus icon theme | GPL-3.0 | 6,140 |
| OpenMoji | CC BY-SA 4.0 | 3,906 |
| Mulberry Symbols | CC BY-SA 4.0 | 3,239 |
| Luanti (Minetest) games and mods | CC BY-SA, LGPL, GPL | 2,699 |
| OpenGameArt entries under GPL-2.0 / GPL-3.0 | GPL | 2,258 |
| Crossfire | GPL-2.0-or-later | 2,039 |
| OpenTTD OpenGFX | GPL-2.0 | 1,364 |
| KDE Oxygen | LGPL-3.0-or-later | 1,265 |

The remaining sources with share-alike or copyleft images are in `SOURCES.md` (176 sources in all).

What this means in practice:

- If a sample closely reproduces one of these images, treat that sample as the original work under its
  licence. About 2% of samples from training captions are near-copies of the caption's own training image
  (`MODEL_CARD.md`); `attribution.tsv.gz` can be searched by source and title.
- If you hold the view that share-alike or copyleft terms reach a model trained on such works, then the
  compatible choices for the weights are the share-alike licences themselves, and the mix of CC BY-SA and
  GPL-family sources has no single licence that satisfies both. A model without that question would be one
  retrained without the 118,466 images.
- The weights are released under Apache-2.0. That does not settle the question above.

## Licences stated but not checked upstream

117,598 images (17.8%) carry the licence that the source or its mirror states, without the upstream page
having been fetched. Nearly all are OpenGameArt entries (CC BY 3.0 / 4.0, CC0, OGA-BY) taken from a mirror
that records each entry's licence, author and URL.

## Trademarks and characters

Some icon sets include product and company logos, and some game art depicts existing characters. The
licences above cover the drawings only.
