#!/usr/bin/env bash
# Fit and analyse the primary model plus the robustness variants.
# Usage: bash scripts/run_robustness_models.sh DATA_DIR OUT_DIR PRIMARY_K "rationale"
set -euo pipefail
D="$1"; O="$2"; K="$3"; WHY="${4:-}"
cd "$(dirname "$0")/.."

run () {  # name, extra args for 04, extra args for 05
  local name="$1"; shift
  # Skip only if a finished run exists AND it was fit on the current chunks.csv
  # (a run from before 01b_clean_chunks.py has a different number of chunks).
  if [ -f "$O/$name/analysis/administration_comparison.csv" ] && \
     [ "$(wc -l < "$O/$name/topics.csv")" -eq "$(wc -l < "$D/chunks.csv")" ]; then
    echo "== $name already done"; return
  fi
  if [ -d "$O/$name" ]; then
    echo "== $name exists but was fit on a different corpus; moving it to ${name}_stale"
    rm -rf "$O/${name}_stale"; mv "$O/$name" "$O/${name}_stale"
  fi
  echo "== $name"
  python scripts/04_train_model.py --data-dir "$D" --out-dir "$O/$name" "$@"
  python scripts/05_run_analysis.py --data-dir "$D" --model-dir "$O/$name" $SEP
}

SEP=""                    run "final_k${K}_seeded"   --k "$K" --k-rationale "$WHY"
SEP="--skip-separability" run "final_k30_seeded"     --k 30
SEP="--skip-separability" run "final_k60_seeded"     --k 60
SEP="--skip-separability" run "final_k${K}_seed1"    --k "$K" --seed 1
SEP="--skip-separability" run "final_k${K}_seed2"    --k "$K" --seed 2
SEP="--skip-separability" run "final_k${K}_unseeded" --k "$K" --no-seeds

python scripts/06_robustness.py --primary "$O/final_k${K}_seeded" \
  --runs "$O/final_k30_seeded" "$O/final_k60_seeded" "$O/final_k${K}_seed1" "$O/final_k${K}_seed2" "$O/final_k${K}_unseeded"
