#!/bin/sh
# Wait until 1:00 AM, then train epoch 3 from epoch 2's best weights.
cd /Users/zblanton/Documents/repos/mtg-open-lane
target=$(date -v+1d -v1H -v0M -v0S +%s)
echo "$(date) waiting until $(date -r $target)" >> results/picknet_all/queue.log
while [ "$(date +%s)" -lt "$target" ]; do sleep 60; done
echo "$(date) starting epoch 3" >> results/picknet_all/queue.log
.venv/bin/python scripts/train_picknet_all.py name=all_d256_ep3 d=256 heads=8 hist_layers=4 pack_layers=3 batch=32 \
  steps=53791 val_every=2500 pack_drop=0 deck_w=0.3 lr=1e-4 warmup=0.02 shuffle_seed=2 \
  init=results/picknet_all/models/all_d256_ep2.pt > results/picknet_all/all_d256_ep3.log 2>&1
echo "$(date) epoch 3 finished (exit $?)" >> results/picknet_all/queue.log
