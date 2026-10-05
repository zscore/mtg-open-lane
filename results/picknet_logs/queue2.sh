# waits for queue 1, then: per-set rating/pool pick models, simulations, ablation, zero-shot multi-set
while pgrep -f msh_queue1.sh > /dev/null; do sleep 20; done
set -x
for s in MSH BLB DFT DSK FDN FIN LCI MKM OTJ EOE SOS TDM ECL TLA; do
  [ $s = MSH ] || .venv/bin/python scripts/train_picknet.py $s rating 6 lr=0.05 batch=128
  [ $s = MSH ] || .venv/bin/python scripts/train_picknet.py $s pool 8 lr=0.01 batch=128 d=32 name=pool_d32
done
touch results/picknet_logs/pool_models.done
for m in picknet pool_d32 rating; do .venv/bin/python scripts/simulate_draft.py MSH $m 1000; done
.venv/bin/python scripts/train_picknet.py MSH picknet 6 batch=128 history=0 name=picknet_nohist
.venv/bin/python scripts/train_picknet_multiset.py MSH 3 20000
