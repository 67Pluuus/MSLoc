#!/bin/bash


set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PR_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
export PYTHONPATH="$PR_DIR:${PYTHONPATH:-}"
MSLOC_ROOT=$(cd "$PR_DIR/.." && pwd)
MSLOC_ASSETS=${MSLOC_ASSETS:-"$(cd "$MSLOC_ROOT/../MSLoc_assets" && pwd)"}
DATA_ROOT=${DATA_ROOT:-"$MSLOC_ASSETS/data/Tasle-CoT-10K"}

MODEL_DIR=${MODEL_DIR:-"$MSLOC_ASSETS/Trace/output/trace_vllava/ref2"}


DATASET='aigc'
SPLIT='test'

TEST_ANNO_FILE=${TEST_ANNO_FILE:-"$MSLOC_ASSETS/DeMamba/results/all_Class_4/eval/predictions.json"}

VIDEO_DIR=${VIDEO_DIR:-"$DATA_ROOT/videos"}
PROMPT_FILE="${PR_DIR}/trace/prompts/dvc.txt"

NUM_FRAME=40
MAX_NEW_TOKENS=512

# The released ref2 checkpoint expects 16 boundary, 8 event, and 16 boundary frames.
BND_RATIO=0.2
BND_FRAMES=16
SEG_FRAMES=8

NUM_GPUS=${NUM_GPUS:-1}
if [ "${NUM_GPUS}" -lt 1 ]; then
  echo "Error: NUM_GPUS must be at least 1."
  exit 1
fi

OUTPUT_DIR="${MSLOC_ASSETS}/Trace/inference_results/ref2_${DATASET}_${SPLIT}"
mkdir -p "${OUTPUT_DIR}"

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

PIDS=()

cleanup() {
  trap - INT TERM
  echo
  echo "Stopping inference processes..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  sleep 2
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  exit 130
}

trap cleanup INT TERM

for i in $(seq 0 $(($NUM_GPUS - 1))); do
  echo "Starting process on GPU ${i} (chunk ${i}/${NUM_GPUS})..."

  if [ "$i" -eq 0 ]; then
    setsid env CUDA_VISIBLE_DEVICES=$i python -u ${PR_DIR}/trace/eval/evaluate_ref.py \
      --anno_file "${TEST_ANNO_FILE}" \
      --video_path "${VIDEO_DIR}" \
      --gpu_id 0 \
      --dataset "${DATASET}" \
      --output_dir "${OUTPUT_DIR}" \
      --split "${SPLIT}" \
      --num_frames ${NUM_FRAME} \
      --prompt_file "${PROMPT_FILE}" \
      --model_path "${MODEL_DIR}" \
      --max_new_tokens ${MAX_NEW_TOKENS} \
      --sample_num -1 \
      --num_chunks ${NUM_GPUS} \
      --chunk_idx $i \
      --quiet_non_master \
      --tqdm_position $i \
      --bnd_ratio ${BND_RATIO} \
      --bnd_frames ${BND_FRAMES} \
      --seg_frames ${SEG_FRAMES} &
    PIDS+=($!)
  else
    LOG_FILE="${OUTPUT_DIR}/gpu${i}.stdout.log"
    setsid env CUDA_VISIBLE_DEVICES=$i python -u ${PR_DIR}/trace/eval/evaluate_ref.py \
      --anno_file "${TEST_ANNO_FILE}" \
      --video_path "${VIDEO_DIR}" \
      --gpu_id 0 \
      --dataset "${DATASET}" \
      --output_dir "${OUTPUT_DIR}" \
      --split "${SPLIT}" \
      --num_frames ${NUM_FRAME} \
      --prompt_file "${PROMPT_FILE}" \
      --model_path "${MODEL_DIR}" \
      --max_new_tokens ${MAX_NEW_TOKENS} \
      --sample_num -1 \
      --num_chunks ${NUM_GPUS} \
      --chunk_idx $i \
      --quiet_non_master \
      --tqdm_position $i \
      --bnd_ratio ${BND_RATIO} \
      --bnd_frames ${BND_FRAMES} \
      --seg_frames ${SEG_FRAMES} 1>"${LOG_FILE}" &
    PIDS+=($!)
  fi
done

FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAILED=1
  fi
done

trap - INT TERM

if [ "${FAILED}" -ne 0 ]; then
  echo "Error: one or more inference processes failed. Merge skipped."
  exit 1
fi

echo "All inference processes finished."

if [ "${NUM_GPUS}" -eq 1 ]; then
  echo "Single-GPU result already generated; merge skipped."
else
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
fi

echo "=========================================="
echo "Inference finished."
echo "Result file: ${OUTPUT_DIR}/fmt_${DATASET}_${SPLIT}_f${NUM_FRAME}_result.json"
echo "- Per-GPU stdout logs: ${OUTPUT_DIR}/gpu*.stdout.log"
echo "=========================================="
