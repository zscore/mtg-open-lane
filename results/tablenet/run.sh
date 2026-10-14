#!/bin/zsh
# TableNet step 1 + FRA/HOB WheelNet baselines. Run from the repo root: caffeinate -i zsh results/tablenet/run.sh
source .venv/bin/activate
set -e
for s in FRA HOB; do
  python scripts/train_nn.py $s 8 PremierDraft rank=0 skill=0 types=0 name=wheelnet_time > results/tablenet/wheelnet_time_$s.log 2>&1
done
python scripts/train_tablenet.py > results/tablenet/tablenet.log 2>&1
