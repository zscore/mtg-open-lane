set -x
.venv/bin/python scripts/train_picknet.py MSH picknet 6 batch=64 lr=7e-4 d=192 heads=6 hist_layers=4 pack_layers=3 name=picknet_big
for m in picknet_big picknet pool_d32 rating; do .venv/bin/python scripts/simulate_draft.py MSH $m 1000; done
.venv/bin/python -u scripts/deck_multiset.py 200000 deck_feats=0 name=shared_nodf > results/deck_multiset_nodf.log 2>&1
.venv/bin/python scripts/train_picknet_multiset.py MSH 3 20000 with_target=1
