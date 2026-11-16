#!/bin/bash
# Benchmark for one or more run dirs of hf_train.py (see BENCHMARK.md). Run when the GPU is free.
#   scripts/benchmark.sh full=outputs_diffusers/sprite32/exports/step_0500000
# Per NAME=RUN_DIR: prompted FID with held-out long and short captions (guidance 1 and 3), unconditional FID,
# and the 50-prompt grid; all against the frozen holdout reference.
# Env: EMA (e.g. 0.9999), STEPS, N, PY (interpreter).
# QUICK=1 (hf_train.py benchmark_exports): guidance 3 only, no unconditional FID, CLIP score of these runs only.
# STATS=<npz>: the reference file (hf_eval.py --precompute; see BENCHMARK.md).
# EXTRA=1: also scripts/bench_extra.py (held-out text accuracy, seed diversity, near-copy rate).
set -euo pipefail
cd "$(dirname "$0")/.."
R=${R:-outputs/benchmark/runs}; mkdir -p "$R"
OUT=$R/results.txt; LOG=$R/eval.log
STATS=${STATS:-outputs/benchmark/eval/real_stats_holdout.npz}
HOLD=curation/holdout.txt
N=${N:-10000}; STEPS=${STEPS:-20}
X="--holdout_list $HOLD --stats_path $STATS --num_samples $N --num_steps $STEPS --amp"
echo "started $(date) N=$N steps=$STEPS ema=${EMA:-default}" >> "$OUT"
PY=${PY:-python}
QUICK=${QUICK:-}
if [ -n "$QUICK" ]; then GUIDANCE=3.0; else GUIDANCE="1.0 3.0"; fi
fid() {  # RUN_DIR [extra args]
  local ck=$1; shift
  $PY hf_eval.py "$ck" $X ${EMA:+--weights ema$EMA} "$@" 2>>"$LOG" | grep "Final FID"
}
for arg in "$@"; do
  name=${arg%%=*}; ck=${arg#*=}
  for g in $GUIDANCE; do
    for key in description verbose_description; do
      tag=$([ $key = description ] && echo long || echo short)
      f=$(fid "$ck" --caption_key $key --guidance_scale $g --keep_dir "$R/${name}_${tag}_g$g")
      echo "$name ${tag}_g$g $f" >> "$OUT"
    done
  done
  if [ -z "$QUICK" ]; then
    f=$(fid "$ck")
    echo "$name uncond $f" >> "$OUT"
  fi
  mapfile -t GRID < curation/prompt_grid.txt
  $PY hf_sample.py "$ck" ${EMA:+--weights ema$EMA} --sampler heun --steps 35 --guidance 3.0 \
    --prompts "${GRID[@]}" --n 6 --seed 0 --out "$R/grid_${name}.png" >>"$LOG" 2>&1 || true
  if [ -n "${EXTRA:-}" ]; then
    $PY scripts/bench_extra.py run "$ck" --name "$name" --out "$R/extra_${name}" ${EMA:+--weights ema$EMA} \
      >>"$OUT" 2>>"$LOG" || true
  fi
done
if [ -n "$QUICK" ]; then
  DIRS=(); for arg in "$@"; do DIRS+=("$R/${arg%%=*}"_long_g* "$R/${arg%%=*}"_short_g*); done
else
  DIRS=("$R"/*_long_g* "$R"/*_short_g*)
fi
$PY scripts/clip_prompt_score.py "${DIRS[@]}" >> "$OUT" 2>>"$LOG" || true
echo "done $(date)" >> "$OUT"
