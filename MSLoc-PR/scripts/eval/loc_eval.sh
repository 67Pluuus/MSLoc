#!/bin/bash

# =============================================================================
# loc_eval.sh - run inference of the `loc` model.
# Single machine, NUM_GPUS-way data-parallel inference.
# =============================================================================

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TRACE_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
MSLOC_ROOT=$(cd "$TRACE_DIR/.." && pwd)
MSLOC_ASSETS=${MSLOC_ASSETS:-"$(cd "$MSLOC_ROOT/../MSLoc_assets" && pwd)"}
DATA_ROOT=${DATA_ROOT:-"$MSLOC_ASSETS/data/Tasle-CoT-10K"}

# ============================ Configurable parameters ============================
DIR="$TRACE_DIR"
MODEL_DIR=${MODEL_DIR:-"$MSLOC_ASSETS/Trace/output/trace_vllava/loc"}

TASK=${TASK:-'dvc'}
DATASET='aigc'
SPLIT='test'

TEST_ANNO_FILE=${TEST_ANNO_FILE:-"$DATA_ROOT/annos/test_all_1209.json"}

VIDEO_DIR=${VIDEO_DIR:-"$DATA_ROOT/videos"}
PROMPT_FILE="${DIR}/trace/prompts/dvc.txt"

NUM_FRAME=32
MAX_NEW_TOKENS=512

NUM_GPUS=8

OUTPUT_DIR="${MSLOC_ASSETS}/Trace/inference_results/loc_${DATASET}_${SPLIT}"
mkdir -p "${OUTPUT_DIR}"

# ============================ Parameter checks ============================
if [ ! -f "${TEST_ANNO_FILE}" ]; then
  echo "Error: TEST_ANNO_FILE does not exist: ${TEST_ANNO_FILE}"
  exit 1
fi

if [ ! -d "${MODEL_DIR}" ]; then
  echo "Error: MODEL_DIR does not exist: ${MODEL_DIR}"
  exit 1
fi

echo "=========================================="
echo "Inference (single-machine ${NUM_GPUS}-GPU data parallel; no metric computation)"
echo "=========================================="
echo "Model dir:    ${MODEL_DIR}"
echo "Test anno:    ${TEST_ANNO_FILE}"
echo "Video dir:    ${VIDEO_DIR}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Num frames:   ${NUM_FRAME}"
echo "=========================================="

echo "Starting inference..."

# ============================ Parallel inference ============================
for i in $(seq 0 $(($NUM_GPUS - 1))); do
  echo "Starting process on GPU ${i} (chunk ${i}/${NUM_GPUS})..."

  if [ "$i" -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=$i python -u ${DIR}/trace/eval/evaluate_loc.py \
      --anno_file "${TEST_ANNO_FILE}" \
      --video_path "${VIDEO_DIR}" \
      --gpu_id 0 \
      --task "${TASK}" \
      --dataset "${DATASET}" \
      --output_dir "${OUTPUT_DIR}" \
      --split "${SPLIT}" \
      --num_frames ${NUM_FRAME} \
      --batch_size 1 \
      --prompt_file "${PROMPT_FILE}" \
      --model_path "${MODEL_DIR}" \
      --max_new_tokens ${MAX_NEW_TOKENS} \
      --sample_num -1 \
      --num_chunks ${NUM_GPUS} \
      --chunk_idx $i \
      --quiet_non_master \
      --tqdm_position $i &
  else
    LOG_FILE="${OUTPUT_DIR}/gpu${i}.stdout.log"
    CUDA_VISIBLE_DEVICES=$i python -u ${DIR}/trace/eval/evaluate_loc.py \
      --anno_file "${TEST_ANNO_FILE}" \
      --video_path "${VIDEO_DIR}" \
      --gpu_id 0 \
      --task "${TASK}" \
      --dataset "${DATASET}" \
      --output_dir "${OUTPUT_DIR}" \
      --split "${SPLIT}" \
      --num_frames ${NUM_FRAME} \
      --batch_size 1 \
      --prompt_file "${PROMPT_FILE}" \
      --model_path "${MODEL_DIR}" \
      --max_new_tokens ${MAX_NEW_TOKENS} \
      --sample_num -1 \
      --num_chunks ${NUM_GPUS} \
      --chunk_idx $i \
      --quiet_non_master \
      --tqdm_position $i 1>"${LOG_FILE}" &
  fi
done

wait

echo "All inference processes finished."

echo "Merging results..."

export OUTPUT_DIR
export DATASET
export SPLIT
export NUM_FRAME="${NUM_FRAME}"

MERGE_SCRIPT="${OUTPUT_DIR}/merge_results.py"

cat << 'EOF' > "${MERGE_SCRIPT}"
import json
import glob
import os

output_dir = os.environ['OUTPUT_DIR']
dataset = os.environ['DATASET']
split = os.environ['SPLIT']
num_frames = int(os.environ['NUM_FRAME'])

pattern = f'fmt_{dataset}_{split}_f{num_frames}_result_chunk*.json'
files = sorted(glob.glob(os.path.join(output_dir, pattern)))
print(f'Found {len(files)} chunk files to merge.')

if not files:
    print('No chunk files found, merge aborted.')
    import sys
    sys.exit(1)

with open(files[0], 'r', encoding='utf-8') as f:
    first = json.load(f)

if isinstance(first, list):
    out = []
    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f'Warning: {fp} is not list, skipped')
            continue
        out.extend(data)

elif isinstance(first, dict):
    if 'annotations' not in first or not isinstance(first['annotations'], list):
        raise SystemExit('Dict root json but missing annotations list')
    
    # For dict-rooted json, only the `annotations` list is merged across chunks.
    # Other fields are taken from the first chunk.
    merged_anns = []
    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict) or 'annotations' not in data:
            print(f'Warning: {fp} is not dict-with-annotations, skipped')
            continue
        merged_anns.extend(data.get('annotations', []))
    
    first['annotations'] = merged_anns
    out = first
else:
    raise SystemExit(f'Unsupported root type: {type(first)}')

out_file = os.path.join(output_dir, f'fmt_{dataset}_{split}_f{num_frames}_result.json')
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False)

print(f'Merged results saved to {out_file}')
print(f'Total items: {len(out) if isinstance(out, list) else len(out["annotations"])}')
EOF

python "${MERGE_SCRIPT}"
rm -f "${MERGE_SCRIPT}"

echo "=========================================="
echo "Inference finished."
echo "Result file: ${OUTPUT_DIR}/fmt_${DATASET}_${SPLIT}_f${NUM_FRAME}_result.json"
echo "- Per-GPU stdout logs: ${OUTPUT_DIR}/gpu*.stdout.log"
echo "=========================================="


