#!/usr/bin/env bash
# Train MSLoc-PG, evaluate the test set, and export PR training proposals.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Default to single-GPU execution.
devices="0"
mode=()
while (( $# )); do
    case "$1" in
        --device-ids) devices="${2:?Missing device IDs}"; shift 2 ;;
        --resume|--clean) mode+=("$1"); shift ;;
        *) echo "Usage: bash MSLoc-PG/run_baseline.sh [--device-ids IDS] [--resume|--clean]" >&2; exit 2 ;;
    esac
done

# Use full logical batches when multiple GPUs are explicitly selected.
forward_options=()
eval_forward_options=()
if [[ "$devices" != *,* ]]; then
    forward_options=(--micro-batch-size 4 --eval-micro-batch-size 4)
    eval_forward_options=(--eval-micro-batch-size 4)
fi

config="MSLoc-PG/configs/xclip_baseline_full.yaml"
output="../MSLoc_assets/DeMamba/full/baseline"
mkdir -p "$output/logs"

python -u MSLoc-PG/train.py \
    --config "$config" --device-ids "$devices" \
    --train-batch-size 32 --val-batch-size 32 --max-epoch 10 \
    --train-videos 0 --val-videos 0 --seed 42 "${forward_options[@]}" "${mode[@]}" \
    2>&1 | tee -a "$output/logs/train.log"

for split in test train; do
    eval_dir="$output/eval"
    if [[ "$split" == train ]]; then eval_dir="$output/eval_train"; fi
    eval_mode=("${mode[@]}")
    # A resumed pipeline may not have reached this evaluation stage yet.
    if [[ " ${mode[*]} " == *" --resume "* && ! -f "$eval_dir/run_config.json" ]]; then
        eval_mode=()
    fi
    python -u MSLoc-PG/eval.py \
        --config "$config" --model_path "$output/results/best_acc.pth" \
        --output_dir "$eval_dir" --device-ids "$devices" \
        --val-batch-size 16 --max-videos 0 --seed 42 --eval-split "$split" "${eval_forward_options[@]}" "${eval_mode[@]}" \
        2>&1 | tee -a "$output/logs/eval_${split}.log"
done
echo "Finished: $output/eval/predictions.json and $output/eval_train/predictions.json"
