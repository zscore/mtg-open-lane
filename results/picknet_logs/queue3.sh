while pgrep -f queue2.sh > /dev/null; do sleep 20; done
set -x
.venv/bin/python -u scripts/deck_multiset.py 200000 name=shared_df > results/deck_multiset.log 2>&1
.venv/bin/python scripts/train_picknet.py MSH picknet 8 batch=128 d=192 heads=6 hist_layers=4 pack_layers=3 name=picknet_big
.venv/bin/python scripts/simulate_draft.py MSH picknet_big 1000
for m in picknet pool_d32 rating; do .venv/bin/python scripts/simulate_draft.py MSH $m 1000; done
.venv/bin/python scripts/train_picknet_multiset.py MSH 3 20000 with_target=1
.venv/bin/python -u scripts/deck_multiset.py 200000 deck_feats=0 name=shared_nodf > results/deck_multiset_nodf.log 2>&1
