#!/bin/sh
# Wait for all_d256 (epoch 1) to finish, resume it if it died, then train epoch 2 from its best weights.
cd /Users/zblanton/Documents/repos/mtg-open-lane
ARCH="d=256 heads=8 hist_layers=4 pack_layers=3 batch=32 steps=53791 val_every=2500 pack_drop=0 deck_w=0.3"
while pgrep -f "name=all_d256 " >/dev/null; do sleep 60; done
if [ ! -f results/picknet_all/all_d256.json ]; then
  echo "$(date) epoch 1 has no results: resuming it" >> results/picknet_all/queue.log
  .venv/bin/python scripts/train_picknet_all.py name=all_d256 $ARCH lr=5e-4 >> results/picknet_all/all_d256.log 2>&1
fi
echo "$(date) starting epoch 2" >> results/picknet_all/queue.log
.venv/bin/python scripts/train_picknet_all.py name=all_d256_ep2 $ARCH lr=2e-4 warmup=0.02 shuffle_seed=1 \
  init=results/picknet_all/models/all_d256.pt > results/picknet_all/all_d256_ep2.log 2>&1
echo "$(date) epoch 2 finished (exit $?)" >> results/picknet_all/queue.log
