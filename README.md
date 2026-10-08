# MSLoc

MSLoc is a two-stage pipeline for localizing AIGC-tampered segments in long videos.
The inference entry points and their behavior are synchronized with `MSLoc_ori`:

1. **MSLoc-PG** uses a DeMamba backbone with boundary-aware four-class prediction to generate coarse proposals.
2. **MSLoc-PR** uses a TRACE-based MLLM to refine proposal boundaries and generate forensic outputs.
3. The long-video evaluator reports the final paper metrics.

[![Project Page](https://img.shields.io/badge/Project-Page-3273DC)](https://debby-0527.github.io/TASLE/)
[![TASLE Dataset](https://img.shields.io/badge/TASLE%20Dataset-ModelScope-624AFF)](https://www.modelscope.cn/datasets/L67plus/TASLE)

The restored inference code includes the original `loc`, `ref`, `ref2`, and `det` paths. The original implementation behavior is retained as-is.

## Layout

The repository contains code only. Download TASLE and the released checkpoints from [ModelScope](https://www.modelscope.cn/datasets/L67plus/TASLE), then use this sibling-directory layout:

```text
<workspace>/
├── MSLoc/
│   ├── MSLoc-PG/              # proposal generation
│   ├── MSLoc-PR/              # proposal refinement
│   ├── scripts/
│   └── README.md
└── MSLoc_assets/
    ├── data/
    │   └── Tasle-CoT-10K/
    │       ├── annos/test.json
    │       ├── videos/
    │       └── video_frames/
    ├── DeMamba/               # released backbone/checkpoint assets
    │   ├── pretrained_weights/xclip-base-patch16/
    │   └── results/all_Class_4/
    │       └── best_acc.pth
    └── Trace/                 # released backbone/checkpoint assets
        ├── ckpts/clip-vit-large-patch14-336/
        ├── output/trace_vllava/ref2/
        └── inference_results/
```

The code directories follow the paper names `MSLoc-PG` and `MSLoc-PR`. The asset directories retain the released `DeMamba` and `Trace` names for checkpoint compatibility. The default paths assume `MSLoc_assets/` is beside this repository. MSLoc-PR paths can also be overridden with `MSLOC_ASSETS`, `DATA_ROOT`, `MODEL_DIR`, `TEST_ANNO_FILE`, and `VIDEO_DIR`.

## Environments

MSLoc-PG and MSLoc-PR use separate environments.

### Stage 1: MSLoc-PG

```bash
conda create -n msloc-pg python=3.10 -y
conda activate msloc-pg
pip install -r MSLoc-PG/requirements.txt
```

### Stage 2: MSLoc-PR

MSLoc-PR inference requires CUDA/Linux.

```bash
conda create -n msloc-pr python=3.10 -y
conda activate msloc-pr
pip install -r MSLoc-PR/requirements.txt
```

Install PyTorch and torchvision wheels matching the local CUDA driver when necessary.

## Training

Use the environments and asset layout above.
Place the Mistral-based `trace-uni` model under `MSLoc_assets/Trace/ckpts/` and BGE under `MSLoc_assets/Trace/bge-large-en-v1.5/`.

### 1. Prepare data

Extract 8-fps frames, then reserve 10% of the training videos for validation. Skip extraction if the frames are already prepared.

```bash
cd MSLoc/MSLoc-PG
conda activate msloc-pg
python Preprocess/video2frame.py \
    --input_root ../../MSLoc_assets/data/Tasle-CoT-10K/videos \
    --output_root ../../MSLoc_assets/data/Tasle-CoT-10K/video_frames
python ../scripts/prepare_train.py \
    --anno_file ../../MSLoc_assets/data/Tasle-CoT-10K/annos/train_all_1209.json \
    --output_dir ../../MSLoc_assets/data/Tasle-CoT-10K/annos \
    --val_ratio 0.1 --seed 42
```

Outputs: frames in `../../MSLoc_assets/data/Tasle-CoT-10K/video_frames/`, and `msloc_train.json` / `msloc_val.json` in the annotation directory.

### 2. Train MSLoc-PG and generate proposals

In the `msloc-pg` environment:

```bash
cd MSLoc/MSLoc-PG
bash train.sh
python eval.py \
    --config ./configs/XCLIP_Tasle_train.yaml \
    --model_path ../../MSLoc_assets/DeMamba/results/all_Class_4/best_acc.pth \
    --output_dir ../../MSLoc_assets/DeMamba/results/all_Class_4/eval_train
```

Settings are in `configs/MSLoc_train.yaml`. Outputs go to
`../../MSLoc_assets/DeMamba/results/all_Class_4/`, including `eval_train/predictions.json`.

### 3. Train MSLoc-PR

Switch to the `msloc-pr` environment, prepare anomaly-class features, then train:

```bash
cd MSLoc/MSLoc-PR
python trace/scripts/extract_class_features.py \
    --data_path ../../MSLoc_assets/data/Tasle-CoT-10K/annos/train_all_1209.json \
    --bge_model_path ../../MSLoc_assets/Trace/bge-large-en-v1.5 \
    --output_path ../../MSLoc_assets/Trace/class_features_bge.pt \
    --device cuda --batch_size 32
bash scripts/train/ref2.sh
```

This enables DAM, EAM and anomaly-aware loss with 40 frames, 2 epochs, effective batch size 16 and learning rate `3e-5`.
Training outputs go to `../../MSLoc_assets/Trace/output/trace_vllava/ref2/`.

## Best-result inference

### 1. Extract 8-fps frames

Run once before Stage 1:

```bash
cd MSLoc/MSLoc-PG
python Preprocess/video2frame.py \
    --input_root  ../../MSLoc_assets/data/Tasle-CoT-10K/videos \
    --output_root ../../MSLoc_assets/data/Tasle-CoT-10K/video_frames
```

The extractor resolves both roots before starting workers and reprocesses incomplete frame directories.

### 2. Generate four-class proposals

```bash
cd MSLoc/MSLoc-PG
bash eval.sh
```

The released classifier uses 2-second windows with 8 sampled frames. Its classes are `0=real`, `1=fake`, `2=real-to-fake`, and `3=fake-to-real`.

The proposal file is written to:

```text
MSLoc_assets/DeMamba/results/all_Class_4/eval/predictions.json
```

### 3. Run TRACE/MSLoc-PR inference

Use the entry point matching the checkpoint and experiment:

```bash
cd MSLoc/MSLoc-PR
bash scripts/eval/loc_eval.sh    # plain end-to-end Trace, full-video input
bash scripts/eval/ref_eval.sh    # ref checkpoint, proposal-conditioned input
bash scripts/eval/ref2_eval.sh   # ref2/MSLoc-PR, proposal-conditioned input
bash scripts/eval/det_eval.sh    # binary detection path
```

These launchers retain the original `MSLoc_ori` configuration and use eight GPUs.

Multi-GPU chunks are merged automatically. The final prediction file is:

```text
MSLoc_assets/Trace/inference_results/ref2_aigc_test/fmt_aigc_test_f40_result.json
```

An explicit-path example is:

```bash
MSLOC_ASSETS=/abs/path/to/MSLoc_assets \
DATA_ROOT=/abs/path/to/Tasle-CoT-10K \
MODEL_DIR=/abs/path/to/MSLoc_assets/Trace/output/trace_vllava/ref2 \
TEST_ANNO_FILE=/abs/path/to/predictions.json \
VIDEO_DIR=/abs/path/to/Tasle-CoT-10K/videos \
bash scripts/eval/ref2_eval.sh
```

### 4. Compute the paper metrics

```bash
cd MSLoc
python scripts/evaluate_long.py \
    --gt_file    ../MSLoc_assets/data/Tasle-CoT-10K/annos/test.json \
    --infer_file ../MSLoc_assets/Trace/inference_results/ref2_aigc_test/fmt_aigc_test_f40_result.json
```

The evaluator requires predictions for exactly the videos in `test.json`. It reports:

- `Det_Acc`: video-level detection accuracy.
- `Loc_F1`: temporal-grid F1 at the released 0.01-second resolution.
- `Loc_IoU`: mean segment-matching F1 over IoU thresholds `{0.1, 0.3, 0.5, 0.7}`.

## Inference data flow

| Artifact | Producer | Consumer |
| --- | --- | --- |
| `video_frames/` | `MSLoc-PG/Preprocess/video2frame.py` | `MSLoc-PG/eval.py` |
| `all_Class_4/eval/predictions.json` | `MSLoc-PG/eval.py` | `MSLoc-PR/trace/eval/evaluate_ref.py` |
| `ref2_aigc_test/fmt_aigc_test_f40_result.json` | `MSLoc-PR/scripts/eval/ref2_eval.sh` | `scripts/evaluate_long.py` |

The official inference split is `MSLoc_assets/data/Tasle-CoT-10K/annos/test.json`. Training annotations, `test_add.json`, and segment-classification annotations are not used by this retained path.

## License

- MSLoc-PR is built on TRACE and [VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2); see `MSLoc-PR/VideoLLaMA2-LICENSE`.
- MSLoc-PG uses the [DeMamba](https://github.com/chenhaoxing/DeMamba) backbone (Apache-2.0).
