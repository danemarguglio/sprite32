# Benchmark

The holdout and the prompt grid were built on 2026-11-04 and are **frozen**, so scores stay comparable
between runs. The training set was built in two rounds whose folders keep their working names: `data_v12/` is
the first collection, which the holdout is drawn from, and `data_v13/` is what sprite32 1.0 was trained on.
Neither the images nor the lists under `curation/`, which name them, are in the repository.

## Files

| File | What |
|------|------|
| `curation/holdout.txt` | 5,000 held-out images (`data_v12/` paths), the reference |
| `outputs/benchmark/eval/real_stats_holdout.npz` | Inception statistics of the holdout (RGBA over white) |
| `curation/prompt_grid.json` / `.txt` | 50-prompt grid (`.txt`: one prompt per line) |
| `curation/text_prompts.jsonl` | held-out strings and layouts for the text metric |
| `scripts/benchmark.sh` | runs everything below for a list of run directories |
| `scripts/bench_extra.py` | text accuracy, seed diversity, near copies |

## Held-out set

- **Drawn from** the captioned images of `data_v12/` that existed on 2026-11-04 (351,872 labels: all
  sliced sprites and Openclipart, 56k of the 238k named-set images). Sources added later are not represented.
- **Stratified** by (source, category). A stratum's share is its kept-label count times candidates/labeled
  for its source (`data_v12/_stats.json`), so the named sets, labeled only about a quarter at the time, get
  their expected final share. Strata of 12 or more images get at least 3; no stratum gives up more than half
  its images.
- **Deterministic**: within a stratum the images with the lowest salted hash of their path are taken, with
  no RNG state.
- **Size 5,000.** Measured against a second disjoint stratified 5k set (same rule, other salt): real-vs-real
  FID is 8.88 at 5k vs 5k, 13.4 at 5k vs 2.5k and 17.4 at 2.5k vs 2.5k, a finite-sample bias of about
  21,700 x (1/N1 + 1/N2). The spread over 3 resamples was 0.03-0.08. So against this reference with 10k
  generated samples, **a perfect model scores about 6.5**. The offset is the same for every model and does
  not change rankings.
- Composition (28 sources; 127 of 200 (source, category) strata represented): iconify_mono 1,581,
  iconify_color 647, oga_ccby3 547, opengameart_cc0 410, openclipart 326, oga_ccby4 311, oga_mixed 277,
  itch_cc0_more 168, oga_ogaby3 151, itch_cc0 141, kenney_extra 127, dcss 65, and 3-45 each from the 16
  small sets. By category: icon 2,809, pixel_art 857, game_sprite 816, emoji 214, tile_or_texture 75,
  ui_element 59, text 55, other 48, abstract 43, photo 12, logo 6, scene 6.

The training tree `data_v13/` was built without the holdout, and checked to hold no holdout path and no
pixel-identical copy of a holdout image, so the training config needs no exclude list.

## Reference statistics (done once)

```sh
python hf_eval.py --precompute --holdout_list curation/holdout.txt \
    --stats_path outputs/benchmark/eval/real_stats_holdout.npz
```

The holdout images go through the training loader (premultiplied RGBA, resize and centre crop to 32x32), are
composited over white and rounded to 8 bits, the same as generated samples.

## Running it

```sh
scripts/benchmark.sh full=outputs_diffusers/sprite32/exports/step_0500000
# results: outputs/benchmark/runs/results.txt
# env: EMA=0.9999 N=10000 STEPS=20 R=<output dir> STATS=<npz> PY=<interpreter> QUICK=1 EXTRA=1
```

Per run directory:

```sh
X="--holdout_list curation/holdout.txt --stats_path $STATS --num_samples 10000 --num_steps 20 --amp"
# prompted FID with the long and short held-out captions, guidance 1 and 3 (QUICK=1: guidance 3 only)
python hf_eval.py RUN_DIR $X --caption_key description         --guidance_scale 3.0 --keep_dir R/name_long_g3.0
python hf_eval.py RUN_DIR $X --caption_key verbose_description --guidance_scale 3.0 --keep_dir R/name_short_g3.0
# unconditional FID (empty caption; skipped with QUICK=1)
python hf_eval.py RUN_DIR $X
# grid: 50 prompts as columns, 6 seeds as rows, the same noise for every model
python hf_sample.py RUN_DIR --sampler heun --steps 35 --guidance 3.0 --prompts ... --n 6 --seed 0 --out R/grid_name.png
# CLIP score of the kept prompted samples
python scripts/clip_prompt_score.py R/name_long_g3.0 R/name_short_g3.0
# EXTRA=1
python scripts/bench_extra.py run RUN_DIR --name name --out R/extra_name
```

The key names are inverted in the sidecars: `description` is the long caption, `verbose_description` the short
one. FID samples use the Heun sampler and each caption's own quality and category labels. `hf_train.py` runs
the script with `QUICK=1` at every export when the config has `benchmark_exports: true` (`benchmark_ema`,
`benchmark_stats`, `benchmark_extra`).

## Prompt grid

`curation/prompt_grid.json` lists, per prompt, `kind` (short/long), `category`, `origin` (hand/holdout)
and, for held-out captions, the real `image`. It holds 30 hand-written prompts and the short and long captions
of 10 held-out images (quality >= 4), one per group: 3D emoji, flat emoji, colour icon, mono icon, 3D icon,
clip art, roguelike tile (dcss), game sprite, pixel art and tile/texture. The hand-written prompts cover the
same groups plus UI.

## Text, seeds and copies (`scripts/bench_extra.py`)

`scripts/bench_extra.py run RUN_DIR --out DIR` prints `<name> <metric> <value>` lines and writes `summary.json`
plus a sheet per part.

- **Text** (`text_exact`, `text_exact_short`, `text_exact_long`, per layout and length, `text_any_seed`): the
  60 held-out strings of the rendered-text set in 172 layouts (`curation/text_prompts.jsonl`, frozen;
  `bench_extra.py prompts` wrote it), 4 seeds each, read by `scripts/text_reader.py`. The reader matches the
  sample against the target, the other held-out strings and one-character edits of the target, drawn in the
  generator's own fonts; a sample is exact when the target is the best match with score >= 0.8. On 1,000 real
  renders it scores 95.4% (plain 99%, label 82%, dialog_box 83%), which is the ceiling of the metric. It only
  knows the generator's fonts: correct text in another letter shape counts as wrong.
- **Seed diversity** (`seed_rmse`, `seed_rmse_min_prompt`, `seed_pairs_near`): the prompt grid at 8 seeds,
  quality 5, no category; mean pairwise RMSE between seeds of a prompt (premultiplied RGBA, 0-255), the lowest
  prompt, and the share of seed pairs within RMSE 10.
- **Near copies** (`copy_train_near`, `copy_train_near5`, `copy_train_own_image`, `copy_holdout_near`,
  `copy_real_holdout_near`): 2,000 training captions and 2,000 holdout captions sampled with their own labels;
  share of samples within RMSE 10 (and 5) of any training view, using the pixel cache written by
  `scripts/cluster.py` (`outputs/cluster/pixels.npy`). `copy_real_holdout_near` is the same share for
  the real holdout images (`curation/holdout_near.tsv`), i.e. what an ideal model would score on the
  holdout captions: 0.1634.
