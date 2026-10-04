#!/bin/bash
# WheelNet input ablation on MSH validation (val_only=1, nothing written besides this log)
cd "$(dirname "$0")/.."
OFF="types=0 rank=0 skill=0 time=0"
run() { echo "=== $1"; .venv/bin/python -u scripts/train_nn.py MSH 8 PremierDraft val_only=1 $2 2>&1 | grep --line-buffered -v -i warn; }
run baseline "$OFF"
run +types "types=1 rank=0 skill=0 time=0"
run +rank+skill "types=0 rank=1 skill=1 time=0"
run +time "types=0 rank=0 skill=0 time=1"
run all ""
