#!/bin/bash
# Train the four-class MSLoc-PG model.

set -euo pipefail

cd "$(dirname "$0")"

export XCLIP_WEIGHTS_PATH=${XCLIP_WEIGHTS_PATH:-"../../MSLoc_assets/DeMamba/pretrained_weights/xclip-base-patch16"}

python train.py \
  --config ./configs/MSLoc_train.yaml \
  --device-ids "${DEVICE_IDS:-0}" \
  --seed 42 \
  "$@"
