#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PY=/home/rojan/anaconda3/envs/myenv/bin/python
EPOCHS=10
BATCH=64
echo "=== baseline (no SSM) ==="
$PY scripts/train.py --epochs $EPOCHS --batch $BATCH --tag oct_baseline_s0 --seed 0
echo "=== SSM hybrid ==="
$PY scripts/train.py --epochs $EPOCHS --batch $BATCH --use_ssm --tag oct_ssm_s0 --seed 0
echo "=== both done ==="
