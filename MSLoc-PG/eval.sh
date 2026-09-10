#!/bin/bash
# Evaluate the trained DeMamba (small) checkpoints on Tasle-CoT-10K.
# Run from MSLoc/DeMamba directory.
set -e

cd "$(dirname "$0")"

# Point to the pre-trained XCLIP backbone in MSLoc_assets
export XCLIP_WEIGHTS_PATH="../../MSLoc_assets/DeMamba/pretrained_weights/xclip-base-patch16"

# ---- four-class on test set ----
python eval.py \
  --config ./configs/XCLIP_Tasle.yaml \
  --model_path ../../MSLoc_assets/DeMamba/results/all_Class_4/best_acc.pth \
  --output_dir ../../MSLoc_assets/DeMamba/results/all_Class_4/eval

# ---- four-class on train set (used as proposals for Trace.ref2 training) ----
python eval.py \
  --config ./configs/XCLIP_Tasle_train.yaml \
  --model_path ../../MSLoc_assets/DeMamba/results/all_Class_4/best_acc.pth \
  --output_dir ../../MSLoc_assets/DeMamba/results/all_Class_4/eval_train

# ---- binary on test set ----
python eval.py \
  --config ./configs/03.yaml \
  --model_path ../../MSLoc_assets/DeMamba/results/03/best_acc.pth \
  --output_dir ../../MSLoc_assets/DeMamba/results/03/eval


