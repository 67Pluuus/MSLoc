#!/bin/bash
# Train MSLoc-PR with DAM, EAM and anomaly classification.

set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH=".:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1

# Model and data paths.
MSLOC_ASSETS=${MSLOC_ASSETS:-"../../MSLoc_assets"}
DATA_ROOT=${DATA_ROOT:-"$MSLOC_ASSETS/data/Tasle-CoT-10K"}
MODEL_DIR=${MODEL_DIR:-"$MSLOC_ASSETS/Trace/ckpts/trace-uni"}
PROPOSAL_PATH=${PROPOSAL_PATH:-"$MSLOC_ASSETS/DeMamba/results/all_Class_4/eval_train/predictions.json"}
CLASS_FEATURE_PATH=${CLASS_FEATURE_PATH:-"$MSLOC_ASSETS/Trace/class_features_bge.pt"}

# Keep the effective batch size at 16 across GPU counts.
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
GLOBAL_BATCH_SIZE=16
if (( NPROC_PER_NODE <= 0 || GRADIENT_ACCUMULATION_STEPS <= 0 )); then
    echo "GPU count and accumulation steps must be positive" >&2
    exit 1
fi
DENOMINATOR=$((NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS))
if (( GLOBAL_BATCH_SIZE % DENOMINATOR != 0 )); then
    echo "16 must be divisible by NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS" >&2
    exit 1
fi
LOCAL_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / DENOMINATOR))
if (( NPROC_PER_NODE == 1 )); then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
fi

torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" trace/train_mt.py \
    --model_name_or_path "$MODEL_DIR" \
    --vision_tower "$MSLOC_ASSETS/Trace/ckpts/clip-vit-large-patch14-336" \
    --class_feature_path "$CLASS_FEATURE_PATH" \
    --data_path "$DATA_ROOT/annos/msloc_train.json" \
    --eval_data_path "$DATA_ROOT/annos/msloc_val.json" \
    --data_folder "$DATA_ROOT/videos" \
    --proposal_path "$PROPOSAL_PATH" \
    --mm_projector_type ref_projector \
    --closs True \
    --num_frames 40 \
    --bnd_frames 16 \
    --seg_frames 8 \
    --bnd_ratio 0.2 \
    --freeze_backbone True \
    --tune_mm_embed_head True \
    --tune_lm_embed_head True \
    --output_dir "$MSLOC_ASSETS/Trace/output/trace_vllava/ref2" \
    --num_train_epochs 2 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size "$LOCAL_BATCH_SIZE" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --model_max_length 4096 \
    --dataloader_num_workers 4 \
    --bf16 True \
    --tf32 False \
    --report_to none \
    --seed 42 \
    --data_seed 42 \
    "$@"
