#!/bin/bash
set -e

cd "$(dirname "$0")"

export XCLIP_WEIGHTS_PATH="../../MSLoc_assets/DeMamba/pretrained_weights/xclip-base-patch16"

python eval.py \
  --config ./configs/XCLIP_Tasle.yaml \
  --model_path ../../MSLoc_assets/DeMamba/results/all_Class_4/best_acc.pth \
  --output_dir ../../MSLoc_assets/DeMamba/results/all_Class_4/eval
