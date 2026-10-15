#!/bin/zsh
# Added at 04:50 on 2026-10-15, after w=3 (wheel log loss past WheelNet, pick top-1 -0.03 pt vs the w=0 control):
# the same w=3 run with the wheel gradient kept out of the history encoder and its inputs (wheel_block=hist).
# If picks return to the w=0 control and wheels stay near w=3, the pick cost lives in the history encoder (D1).
source .venv/bin/activate
while pgrep -f "run_overnight.sh" > /dev/null; do sleep 60; done
python scripts/train_tablenet_joint.py wheel_w=3 wheel_block=hist name=tablenet_joint_w3_hblock \
    > results/tablenet/tablenet_joint_w3_hblock.log 2>&1
