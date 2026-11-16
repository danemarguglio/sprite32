# The sample sheets

`samples.png` is sprite32 1.0 at 35 steps, `samples_fast.png` is sprite32 1.0 fast at 12 steps. Both are shown
at 4x over white, with guidance 3 and quality 5, which are the defaults of `hf_sample.py`.

These are picked, not typical. Each prompt was drawn with seeds 1 to 16 on both models and one seed was
chosen by eye. 167 prompts were tried and 48 are on the sheet. None of the picks is a near-copy of a
training image: the closest is at RMSE 24 on the 0-255 scale, and the benchmark counts 10 or less as a copy.

To redraw one:

```sh
python hf_sample.py weights/sprite32-1.0 --n 1 --seed 9 --out wizard.png \
    --prompts "pixel art wizard in a purple robe holding a staff"
```

For the fast model add `--steps 12`.

| Row | Column | Prompt | Seed, 1.0 | Seed, 1.0 fast |
|---:|---:|---|---:|---:|
| 1 | 1 | pixel art wizard in a purple robe holding a staff | 9 | 9 |
| 1 | 2 | pixel art treasure chest with gold coins | 5 | 16 |
| 1 | 3 | glossy 3D rocket icon | 3 | 3 |
| 1 | 4 | pixel art green frog | 11 | 11 |
| 1 | 5 | red stop sign that reads "STOP", pixel art | 12 | 8 |
| 1 | 6 | pixel art green slime monster | 4 | 11 |
| 1 | 7 | blue diamond gem, game icon | 6 | 6 |
| 1 | 8 | face with tears of joy emoji | 6 | 9 |
| 2 | 1 | pixel art battle axe with a double blade | 2 | 12 |
| 2 | 2 | pixel art floating eyeball monster with tentacles | 12 | 3 |
| 2 | 3 | pixel art cobblestone tile | 6 | 6 |
| 2 | 4 | pixel art button that reads PLAY, navy blue | 6 | 7 |
| 2 | 5 | cute orange fox, flat cartoon | 8 | 8 |
| 2 | 6 | pixel art gold crown with jewels | 11 | 11 |
| 2 | 7 | glossy blue globe icon | 2 | 2 |
| 2 | 8 | green rounded play button | 10 | 6 |
| 3 | 1 | A pixel art sprite of a dwarf with a long orange beard and a horned iron helmet, holding a hammer. He has a black outline and faces forward on a transparent background. | 10 | 4 |
| 3 | 2 | red health potion in a round glass flask, game item | 8 | 4 |
| 3 | 3 | A stylized image depicts a sunflower against a dark green background. The sunflower has bright yellow petals forming a roughly circular shape around a dark brown circular center. | 7 | 7 |
| 3 | 4 | gold circle badge labelled 777, pixel art | 13 | 8 |
| 3 | 5 | pixel art robot, gray with red eyes | 13 | 3 |
| 3 | 6 | pixel art hamburger | 13 | 3 |
| 3 | 7 | flat camera icon | 7 | 3 |
| 3 | 8 | pixel art lava tile | 9 | 15 |
| 4 | 1 | butterfly with blue wings | 3 | 3 |
| 4 | 2 | pixel art treasure chest monster with teeth | 1 | 15 |
| 4 | 3 | golden key, game item | 11 | 5 |
| 4 | 4 | wooden sign on a post saying 'SHOP' | 3 | 3 |
| 4 | 5 | smiling sun emoji, glossy 3D style | 11 | 11 |
| 4 | 6 | flat cloud icon with rain drops | 16 | 16 |
| 4 | 7 | pixel art vampire with a black cape | 3 | 3 |
| 4 | 8 | pixel art checkbox with a green check mark | 10 | 16 |
| 5 | 1 | pixel art bomb with a lit fuse | 4 | 3 |
| 5 | 2 | green baby dragon sprite, pixel art | 15 | 3 |
| 5 | 3 | tiny black text HELP on cyan background, pixel font | 5 | |
| 5 | 3 | The word YES in white pixel letters, set against a solid green background. Pixel art style, filling the frame. | | 10 |
| 5 | 4 | brown owl, cartoon clip art | 3 | 3 |
| 5 | 5 | pixel art spell book with a purple cover | 2 | 3 |
| 5 | 6 | green alien face emoji | 2 | 13 |
| 5 | 7 | A glossy 3D-rendered orange basketball with black seams, on a transparent background. | 11 | 11 |
| 5 | 8 | pixel art fire elemental monster | 6 | 3 |
| 6 | 1 | pixel text 'FIRE' in orange | 7 | 12 |
| 6 | 2 | pixel art cactus in a clay pot | 3 | 10 |
| 6 | 3 | pixel art cherries | 7 | 7 |
| 6 | 4 | A top-down pixel art tile of green grass with small yellow flowers. | 10 | 16 |
| 6 | 5 | party popper emoji | 13 | 11 |
| 6 | 6 | "START" button, cream with brown outline | 5 | 8 |
| 6 | 7 | black line icon of a shopping cart | 3 | 3 |
| 6 | 8 | pixel art hourglass | 10 | 10 |

Row 5, column 3 is the one place where the sheets use different prompts. The fast model wrote "HEEP"
for HELP on all 16 seeds, so its sheet has another word there.
