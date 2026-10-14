#!/bin/zsh
# TableNet step 1 + FRA/HOB WheelNet baselines. Run from the repo root: caffeinate -i zsh results/tablenet/run.sh
# A failed baseline is logged and skipped so TableNet still runs.
source .venv/bin/activate
for s in FRA HOB; do
  python scripts/train_nn.py $s 8 PremierDraft rank=0 skill=0 types=0 name=wheelnet_time \
    > results/tablenet/wheelnet_time_$s.log 2>&1 || echo "baseline $s failed" >> results/tablenet/wheelnet_time_$s.log
done
python scripts/train_tablenet.py > results/tablenet/tablenet.log 2>&1
