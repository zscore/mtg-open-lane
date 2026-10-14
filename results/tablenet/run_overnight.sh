#!/bin/zsh
# Overnight queue, 2026-10-15 (detached: nohup caffeinate -i zsh results/tablenet/run_overnight.sh &), after the
# wheel_w=0 control from run_w1_control.sh finishes:
#   1. wheel weight 3.0 from step 1 (step 2 used 0.3, then 1.0)
#   2. a second pass continuing tablenet_joint_w1 at wheel weight 1.0 (is one pass under-training?)
#   3. gradient cosines on both new checkpoints
source .venv/bin/activate
while pgrep -f "train_tablenet_joint.py wheel_w=0" > /dev/null; do sleep 60; done
python scripts/train_tablenet_joint.py wheel_w=3 name=tablenet_joint_w3 > results/tablenet/tablenet_joint_w3.log 2>&1
python scripts/train_tablenet_joint.py wheel_w=1 resume=results/tablenet/models/tablenet_joint_w1.pt \
    name=tablenet_joint_w1_ep2 > results/tablenet/tablenet_joint_w1_ep2.log 2>&1
python scripts/grad_cosine.py batches=1000 name=grad_cosine_late \
    ckpts=w3=results/tablenet/models/tablenet_joint_w3.pt,w1_ep2=results/tablenet/models/tablenet_joint_w1_ep2.pt \
    > results/tablenet/grad_cosine_late.log 2>&1
