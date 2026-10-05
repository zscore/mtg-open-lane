set -x
.venv/bin/python -u scripts/train_wheelnet_multiset.py MSH 4 20000 with_target=1 > results/wheel_logs/multiset_plus_target.log 2>&1
.venv/bin/python -u scripts/train_wheelnet_multiset.py MSH 4 20000 > results/wheel_logs/multiset_zeroshot.log 2>&1
