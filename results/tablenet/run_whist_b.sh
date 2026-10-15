#!/bin/zsh
# Run B of run_whist.sh, resumed after a pause (2026-10-15): 2-layer wheel history encoder, wheel loss kept out of
# PickNet's history encoder only, w=3 (nohup caffeinate -i zsh results/tablenet/run_whist_b.sh &)
source .venv/bin/activate
python scripts/train_tablenet_joint.py whist_layers=2 wheel_block=hist wheel_w=3 name=tablenet_whist_hist_w3 \
    > results/tablenet/tablenet_whist_hist_w3.log 2>&1
