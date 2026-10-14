#!/bin/zsh
# TableNet step 2 variants, run from the repo root (detached: nohup caffeinate -i zsh results/tablenet/run_w1_control.sh &):
#   1. wheel weight 1.0 (step 2 used 0.3)
#   2. pick-only control: wheel weight 0, the same fine-tune without the wheel loss (its wheel numbers are meaningless)
source .venv/bin/activate
python scripts/train_tablenet_joint.py wheel_w=1 name=tablenet_joint_w1 > results/tablenet/tablenet_joint_w1.log 2>&1
python scripts/train_tablenet_joint.py wheel_w=0 name=tablenet_joint_w0 > results/tablenet/tablenet_joint_w0.log 2>&1
