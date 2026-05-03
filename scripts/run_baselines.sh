#!/bin/bash
# Queue baselines after the current run_both.sh finishes
set -e
cd "$(dirname "$0")/.."
PY=/home/rojan/anaconda3/envs/myenv/bin/python
EPOCHS=10
echo "=== ResNet-50 baseline ==="
$PY scripts/train.py --epochs $EPOCHS --batch 48 --backbone resnet50 --tag oct_resnet50_s0 --seed 0
echo "=== EfficientNet-B0 baseline ==="
$PY scripts/train.py --epochs $EPOCHS --batch 64 --backbone efficientnet_b0 --tag oct_effnet_b0_s0 --seed 0
echo "=== ViT-B/16 baseline ==="
$PY scripts/train.py --epochs $EPOCHS --batch 32 --backbone vit_b_16 --tag oct_vit_b16_s0 --seed 0
echo "=== all baselines done ==="
