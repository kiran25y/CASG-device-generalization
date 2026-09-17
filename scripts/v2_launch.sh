#!/usr/bin/env bash
# v2 experiment — Phase 1: CF-only, CASG-Identity, reference CASG under the
# corrected common pipeline. 3 conditions x 6 fits x 3 seeds = 54 runs.
#
#   nohup bash scripts/v2_launch.sh > runs_v2/launch.log 2>&1 &
#   tail -f runs_v2/launch.log
#
# Resumable: a run whose ckpt already exists is skipped; a run interrupted
# mid-way resumes from its resume_*.pt via the trainer.
#
# Order: seed-major so that after ~18 runs every condition x fit has one
# seed and an interim comparison is possible.
#
# PRE-CONDITIONS (the script refuses to start otherwise):
#   * python scripts/v2_gates.py printed ALL V2 GATES PASSED
#   * git tree is committed (so the recorded hash identifies this code)
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=runs_v2
mkdir -p "$OUT"
GATE_STAMP="$OUT/.gates_passed"

echo "=== v2 Phase 1 launcher  $(date -u +%FT%TZ) ==="
if [ ! -f "$GATE_STAMP" ]; then
  echo "running gates..."
  python scripts/v2_gates.py 2>&1 | tee "$OUT/gates.log"
  grep -q "ALL V2 GATES PASSED" "$OUT/gates.log" || { echo "GATES FAILED — not launching"; exit 1; }
  date -u +%FT%TZ > "$GATE_STAMP"
fi
if [ -n "$(git status --porcelain -- src scripts configs)" ]; then
  echo "src/scripts/configs have uncommitted changes — commit first:"; git status --short -- src scripts configs
  exit 1
fi
git rev-parse HEAD > "$OUT/code_commit.txt"
echo "code commit: $(cat $OUT/code_commit.txt)"

FOLDS=(AKGC417L Meditron LittC2SE Litt3200 smartphone none)
SEEDS=(0 1 2)
# condition -> config, variant, tag suffix
declare -A CFG=( [cf]=configs/cf_physics.yaml [id]=configs/casg_identity.yaml [lite]=configs/casg_lite.yaml )
declare -A VAR=( [cf]=cf_physics [id]=casg_identity [lite]=casg_lite )
CONDS=(cf id lite)

n_done=0; n_skip=0; n_total=$(( ${#CONDS[@]} * ${#FOLDS[@]} * ${#SEEDS[@]} ))
for s in "${SEEDS[@]}"; do
  for d in "${FOLDS[@]}"; do
    for c in "${CONDS[@]}"; do
      tag="casg_ast_${d}_seed${s}_${VAR[$c]}_v2"
      ck="$OUT/casg/ckpt_${tag}.pt"
      if [ -f "$ck" ]; then n_skip=$((n_skip+1)); echo "[skip] $tag (exists)"; continue; fi
      echo; echo "=== [$((n_done+n_skip+1))/$n_total] $tag  $(date -u +%FT%TZ) ==="
      python scripts/train_contrastive.py \
          --config "${CFG[$c]}" --held_out "$d" --seed "$s" \
          --tag_suffix "_${VAR[$c]}_v2" \
          --set output_dir="$OUT" model.variant="${VAR[$c]}" \
        2>&1 | tee "$OUT/log_${tag}.txt"
      [ -f "$ck" ] || { echo "*** run produced no checkpoint: $tag — stopping"; exit 1; }
      n_done=$((n_done+1))
    done
  done
  echo; echo "=== seed $s complete: $n_done trained, $n_skip skipped  $(date -u +%FT%TZ) ==="
done

echo
echo "=== ALL 54 RUNS PRESENT  $(date -u +%FT%TZ) ==="
echo "Freeze + alignment gate + pre-declared contrasts (raw selection = primary):"
echo "  python scripts/v2_analyze.py"
echo "  python scripts/v2_analyze.py --selection ema     # secondary"