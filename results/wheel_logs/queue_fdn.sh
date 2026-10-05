set -x
.venv/bin/python scripts/train_picknet.py FDN picknet 6 batch=128 > results/wheel_logs/picknet_FDN.log 2>&1
.venv/bin/python -u scripts/force_vs_follow.py FDN 2000 picknet > results/wheel_logs/force_vs_follow_FDN.log 2>&1
echo "FDN DONE" >> results/wheel_logs/force_vs_follow_FDN.log
