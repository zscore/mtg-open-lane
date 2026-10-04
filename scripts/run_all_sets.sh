#!/usr/bin/env bash
# Build features, fit baselines + lgbm_all, train WheelNet and summarise each set in turn.
#   scripts/run_all_sets.sh [SET ...]    (default: every PremierDraft set in the inventory)
# Bigger sets get fewer WheelNet epochs so each set costs about the same: round(8 * 48k / train
# drafts), clamped to 3..8. Logs go to results/<SET>.PremierDraft/run.log.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONPATH=src
SETS=("$@")
if [ ${#SETS[@]} -eq 0 ]; then
  SETS=($(awk -F, '$2=="PremierDraft"{print $3" "$1}' data/parquet/inventory.csv | sort -n | awk '{print $2}'))
fi
for s in "${SETS[@]}"; do
  out=results/$s.PremierDraft; mkdir -p "$out"; log=$out/run.log
  echo "=== $s $(date '+%H:%M:%S')" | tee -a "$log"
  { $PY scripts/build_dataset.py "$s" &&
    $PY scripts/train.py "$s" PremierDraft quick=1 &&
    n=$($PY -c "import pickle;print(len(pickle.load(open('data/features/$s.PremierDraft/meta.pkl','rb'))['splits']['train']))") &&
    ep=$($PY -c "print(min(8, max(3, round(8 * 48000 / $n))))") &&
    echo "wheelnet epochs=$ep" &&
    $PY scripts/train_nn.py "$s" "$ep" &&
    $PY scripts/report.py "$s" PremierDraft readme=0; } >> "$log" 2>&1
  status=$?
  echo "=== $s done status=$status $(date '+%H:%M:%S')" | tee -a "$log"
done
