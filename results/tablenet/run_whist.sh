#!/bin/zsh
# Wheel-specific history encoder (gap-diagnosis.md, next step 1), 2026-10-15
# (detached: nohup caffeinate -i zsh results/tablenet/run_whist.sh &):
#   A. 2-layer wheel history, wheel loss kept out of PickNet entirely (picks train as in the w=0 control), w=1
#   B. 2-layer wheel history, wheel loss kept out of PickNet's history encoder only, w=3
source .venv/bin/activate
python scripts/train_tablenet_joint.py whist_layers=2 wheel_block=all wheel_w=1 name=tablenet_whist_all \
    > results/tablenet/tablenet_whist_all.log 2>&1
python scripts/train_tablenet_joint.py whist_layers=2 wheel_block=hist wheel_w=3 name=tablenet_whist_hist_w3 \
    > results/tablenet/tablenet_whist_hist_w3.log 2>&1
