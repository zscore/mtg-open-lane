set -x
P=".venv/bin/python scripts/train_picknet.py MSH"
$P rating 6 lr=0.05 batch=128
$P pool 8 lr=0.01 batch=128 d=32 name=pool_d32
$P pool 8 lr=0.01 batch=128 d=64 ctx=1 name=pool_d64
$P picknet 6 batch=128 name=picknet
