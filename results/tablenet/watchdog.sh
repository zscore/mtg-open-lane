#!/bin/zsh
# Resume (SIGCONT) any stopped TableNet job once a minute; w0 sat stopped (state T) for 75 min on 2026-10-15.
# Logs each resume to results/tablenet/watchdog.log; exits when the overnight queue is gone.
while pgrep -f "run_overnight.sh|run_w1_control.sh" > /dev/null; do
  for p in $(pgrep -f "train_tablenet_joint.py|grad_cosine.py"); do
    if ps -o stat= -p $p | grep -q T; then kill -CONT $p; echo "$(date) resumed $p" >> results/tablenet/watchdog.log; fi
  done
  sleep 60
done
