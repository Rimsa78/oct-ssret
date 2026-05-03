#!/bin/bash
# Full publication pipeline. Runs after the training queue completes.
# Outputs:
#   artifacts/<tag>/{eval_test.json, eval_octid.json, mcnemar*.json}
#   artifacts/comparison.{tsv,tex}
#   paper/paper_latex/figures/fig_*.{pdf,png}
#   paper/oct_ssret_submission.zip
set +e
cd "$(dirname "$0")/.."
PY=/home/rojan/anaconda3/envs/myenv/bin/python
LOG=/tmp/oct_make_paper.log
> $LOG

ALL_TAGS="oct_baseline_s0 oct_ssm_s0 oct_resnet50_s0 oct_effnet_b0_s0 oct_vit_b16_s0 oct_vit_b16_ssm_s0 oct_swin_s0 oct_swin_s1 oct_swin_s2 oct_swin_ssm_s0 oct_swin_ssm_s1 oct_swin_ssm_s2 oct_swin_uni_s0 oct_swin_mlp_s0"

echo "=== eval all checkpoints on Kermany test + OCTID ===" | tee -a $LOG
for tag in $ALL_TAGS; do
    if [ -f "artifacts/${tag}/best.pt" ]; then
        echo "[$tag] Kermany test" | tee -a $LOG
        $PY scripts/eval.py --ckpt artifacts/${tag}/best.pt 2>&1 | tail -5 | tee -a $LOG
        echo "[$tag] OCTID zero-shot" | tee -a $LOG
        $PY scripts/eval_octid.py --ckpt artifacts/${tag}/best.pt 2>&1 | tail -5 | tee -a $LOG
    fi
done

echo "=== build comparison table ===" | tee -a $LOG
# update comparison ROWS to include the new Swin and ViT-SSM entries
$PY -c "
import sys
fp = 'scripts/build_comparison.py'
src = open(fp).read()
new_rows = '''ROWS = [
    (\"oct_resnet50_s0\",       \"ResNet-50 (CNN baseline)\"),
    (\"oct_effnet_b0_s0\",      \"EfficientNet-B0 (CNN baseline)\"),
    (\"oct_vit_b16_s0\",        \"ViT-B/16 (Transformer baseline)\"),
    (\"oct_vit_b16_ssm_s0\",    \"ViT-B/16 + SSM (honest comparison)\"),
    (\"oct_baseline_s0\",       \"OCT-SSRet/ResNet-18 baseline\"),
    (\"oct_ssm_s0\",            \"OCT-SSRet/ResNet-18 + SSM\"),
    (\"oct_swin_s0\",           \"OCT-SSRet/Swin-V2-T baseline\"),
    (\"oct_swin_ssm_s0\",       \"OCT-SSRet/Swin-V2-T + SSM (ours, full)\"),
]'''
import re
src = re.sub(r'ROWS = \\[.*?\\]', new_rows, src, count=1, flags=re.S)
open(fp, 'w').write(src)
print('build_comparison.py rows updated')
" 2>&1 | tee -a $LOG
$PY scripts/build_comparison.py 2>&1 | tee -a $LOG

echo "=== McNemar tests ===" | tee -a $LOG
# Swin + SSM vs Swin baseline
if [ -f artifacts/oct_swin_ssm_s0/best.pt ] && [ -f artifacts/oct_swin_s0/best.pt ]; then
    $PY scripts/mcnemar_test.py --a artifacts/oct_swin_s0/best.pt --b artifacts/oct_swin_ssm_s0/best.pt \
        --name_a "Swin-V2-T baseline" --name_b "Swin-V2-T + SSM" 2>&1 | tail -10 | tee -a $LOG
fi
# Swin + SSM vs ResNet-50 (best CNN)
if [ -f artifacts/oct_swin_ssm_s0/best.pt ] && [ -f artifacts/oct_resnet50_s0/best.pt ]; then
    $PY scripts/mcnemar_test.py --a artifacts/oct_resnet50_s0/best.pt --b artifacts/oct_swin_ssm_s0/best.pt \
        --name_a "ResNet-50 (best CNN)" --name_b "Swin-V2-T + SSM (ours)" 2>&1 | tail -10 | tee -a $LOG
fi

echo "=== qualitative + statistical figures ===" | tee -a $LOG
WINNER=artifacts/oct_swin_ssm_s0/best.pt
if [ ! -f "$WINNER" ]; then WINNER=artifacts/oct_swin_s0/best.pt; fi
if [ ! -f "$WINNER" ]; then WINNER=artifacts/oct_ssm_s0/best.pt; fi

echo "[fig] Grad-CAM on winner" | tee -a $LOG
$PY scripts/figure_gradcam.py --ckpt $WINNER 2>&1 | tail -3 | tee -a $LOG
echo "[fig] confusion+ROC (Kermany + OCTID)" | tee -a $LOG
$PY scripts/figure_confusion_roc.py --ckpt $WINNER 2>&1 | tail -3 | tee -a $LOG
echo "[fig] training curves" | tee -a $LOG
$PY scripts/figure_curves.py --runs oct_baseline_s0 oct_ssm_s0 oct_swin_s0 oct_swin_ssm_s0 \
    --labels "ResNet-18 baseline" "ResNet-18 + SSM" "Swin-V2-T baseline" "Swin-V2-T + SSM" 2>&1 | tail -3 | tee -a $LOG
echo "[fig] ekacare qualitative (paired-modality demo)" | tee -a $LOG
$PY scripts/figure_ekacare_qualitative.py --ckpt $WINNER 2>&1 | tail -3 | tee -a $LOG
echo "[fig] architecture (matplotlib backup)" | tee -a $LOG
$PY scripts/figure_architecture.py 2>&1 | tail -3 | tee -a $LOG

echo "=== B. 5-crop TTA on winner ===" | tee -a $LOG
$PY scripts/eval_tta.py --ckpt $WINNER 2>&1 | tail -8 | tee -a $LOG

echo "=== C. Compute / FLOPs / latency analysis ===" | tee -a $LOG
$PY scripts/compute_analysis.py --batch 16 --image_size 256 2>&1 | tail -15 | tee -a $LOG

echo "=== D. OOD detection (Kermany ID vs OCTID OOD) for baseline + SSM ===" | tee -a $LOG
$PY scripts/ood_analysis.py --ckpt artifacts/oct_swin_s0/best.pt 2>&1 | tail -10 | tee -a $LOG
$PY scripts/ood_analysis.py --ckpt artifacts/oct_swin_ssm_s0/best.pt 2>&1 | tail -10 | tee -a $LOG

echo "=== Per-class stratified OCTID transfer (well-mapped vs poorly-mapped) ===" | tee -a $LOG
$PY scripts/stratified_octid.py --baseline artifacts/oct_swin_s0/best.pt --ssm artifacts/oct_swin_ssm_s0/best.pt 2>&1 | tail -15 | tee -a $LOG

echo "=== MAKE_PAPER_COMPLETE at $(date) ===" | tee -a $LOG
