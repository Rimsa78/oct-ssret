#!/bin/bash
# Runs after the entire training queue finishes.
# Eval all 5 ckpts on Kermany test + OCTID zero-shot, build comparison table,
# Grad-CAM on best SSM model, ekacare qualitative figure.
set +e  # don't bail if one step fails - we want everything else to still run
cd "$(dirname "$0")/.."
PY=/home/rojan/anaconda3/envs/myenv/bin/python
LOG=/tmp/oct_overnight.log

echo "=== overnight wrap started at $(date) ===" > $LOG

for tag in oct_baseline_s0 oct_ssm_s0 oct_resnet50_s0 oct_effnet_b0_s0 oct_vit_b16_s0; do
    if [ -f "artifacts/${tag}/best.pt" ]; then
        echo "=== eval ${tag} on Kermany test ===" >> $LOG
        $PY scripts/eval.py --ckpt artifacts/${tag}/best.pt >> $LOG 2>&1
        echo "=== eval ${tag} on OCTID zero-shot ===" >> $LOG
        $PY scripts/eval_octid.py --ckpt artifacts/${tag}/best.pt >> $LOG 2>&1
    else
        echo "[skip] artifacts/${tag}/best.pt missing" >> $LOG
    fi
done

echo "=== build comparison table ===" >> $LOG
$PY scripts/build_comparison.py >> $LOG 2>&1

echo "=== Grad-CAM on best SSM model ===" >> $LOG
$PY scripts/figure_gradcam.py --ckpt artifacts/oct_ssm_s0/best.pt >> $LOG 2>&1

echo "=== ekacare qualitative ===" >> $LOG
$PY scripts/figure_ekacare_qualitative.py --ckpt artifacts/oct_ssm_s0/best.pt >> $LOG 2>&1

echo "=== OVERNIGHT_DONE at $(date) ===" >> $LOG
