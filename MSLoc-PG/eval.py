"""Evaluate MSLoc-PG and export proposals compatible with MSLoc-PR."""
import os
from pathlib import Path
from runtime import prepare_run, seed_all
import json
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from dataloader import generate_dataset_loader_from_json

import models
from util import build_model
from util import print_evaluation_metrics, eval_model

def parse_device_ids(value):
    """Parse explicit CUDA device IDs, e.g. ``0`` or ``0,1,...,7``."""
    try:
        device_ids = [int(item.strip()) for item in value.split(',') if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError('--device-ids must be comma-separated integers') from error
    if not device_ids or len(set(device_ids)) != len(device_ids) or min(device_ids) < 0:
        raise argparse.ArgumentTypeError('--device-ids must contain unique non-negative integers')
    return device_ids


def load_trained_model(cfg, model_path, device='cuda', device_ids=None):
    """Load a trained checkpoint."""
    model = build_model(
        cfg['model'],
        neuron_indices_path=cfg.get('neuron_indices_path'),
        xclip_model_path=cfg.get('xclip_model_path'),
        xclip_checkpoint_path=cfg.get('xclip_checkpoint_path'),
    )
    model = model.to(device)

    if device_ids is not None and len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # Adapt between single-GPU and multi-GPU formats.
        if not any(key.startswith('module.') for key in state_dict.keys()) and device_ids is not None and len(device_ids) > 1:
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif any(key.startswith('module.') for key in state_dict.keys()) and (device_ids is None or len(device_ids) == 1):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"[ok] Loaded model: {model_path}")
    return model

def save_predictions_json(predictions_data, output_path):
    """Save predictions to a JSON file."""
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return obj

    serializable_data = convert_to_serializable(predictions_data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)

    print(f"[ok] Predictions saved to: {output_path}")

def convert_to_serializable(obj):
    """Recursively convert NumPy values to JSON-compatible Python values."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(value) for value in obj]
    return obj


def analyze_fake_video_proposals(per_video_results, low_recall, high_confidence):
    """Measure the quality of Stage-1 binary proposals against fake-video GT.

    Recall and FPR use the existing 0.01-second temporal masks (the same
    TP/FP/FN/TN definition used by F1Det).  FPR is aggregated over every
    ground-truth-real moment, including the real portions of fake videos and
    the full duration of real videos.
    """
    fake_rows = []
    classification_errors = []

    for video_name, result in per_video_results.items():
        video_correct = bool(result.get('video_correct', False))
        video_confidence = float(result.get('video_decision_confidence', 0.0))
        classification_errors.append({
            'video_name': video_name,
            'is_error': not video_correct,
            'decision_confidence': video_confidence,
            'is_high_confidence_error': (not video_correct and video_confidence >= high_confidence),
        })

        if result.get('video_true_type') != 'fake':
            continue

        recall = float(result.get('F1Det_recall', 0.0))

        proposal_metrics = {
            'proposal_recall': recall,
            'proposal_confidence_mean': float(result.get('proposal_confidence_mean', 0.0)),
            'proposal_confidence_max': float(result.get('proposal_confidence_max', 0.0)),
            'video_decision_confidence': video_confidence,
            'video_classification_correct': video_correct,
            'low_recall_proposal': recall < low_recall,
        }
        result['proposal_quality'] = proposal_metrics
        fake_rows.append({'video_name': video_name, **proposal_metrics})

    total_videos = len(classification_errors)
    classification_error_count = sum(row['is_error'] for row in classification_errors)
    high_confidence_error_count = sum(row['is_high_confidence_error'] for row in classification_errors)
    low_recall_rows = [row for row in fake_rows if row['low_recall_proposal']]
    total_fp = sum(int(result.get('F1Det_fp', 0)) for result in per_video_results.values())
    total_tn = sum(int(result.get('F1Det_tn', 0)) for result in per_video_results.values())

    summary = {
        'definition': {
            'proposal_metrics': 'Per-fake-video temporal overlap at 0.01s resolution.',
            'low_recall_threshold': low_recall,
            'high_confidence_threshold': high_confidence,
            'classification': 'Video real/fake decision derived from the existing four-class window model.',
        },
        'counts': {
            'total_videos': total_videos,
            'fake_videos': len(fake_rows),
            'video_classification_errors': classification_error_count,
            'low_recall_fake_videos': len(low_recall_rows),
            'high_confidence_classification_errors': high_confidence_error_count,
            'temporal_false_positives_0.01s': total_fp,
            'temporal_true_negatives_0.01s': total_tn,
        },
        'fake_video_macro_average': {
            'proposal_recall': (
                float(np.mean([row['proposal_recall'] for row in fake_rows])) if fake_rows else 0.0
            ),
        },
        'ratios': {
            'video_classification_error_ratio': (
                classification_error_count / total_videos if total_videos else 0.0
            ),
            'high_confidence_classification_error_ratio_all_videos': (
                high_confidence_error_count / total_videos if total_videos else 0.0
            ),
            'high_confidence_classification_error_ratio_among_errors': (
                high_confidence_error_count / classification_error_count
                if classification_error_count else 0.0
            ),
            'low_recall_ratio_among_fake_videos': (
                len(low_recall_rows) / len(fake_rows) if fake_rows else 0.0
            ),
            'temporal_false_positive_rate': (
                total_fp / (total_fp + total_tn) if (total_fp + total_tn) else 0.0
            ),
        },
    }
    return summary, fake_rows


def print_proposal_quality_summary(summary):
    averages = summary['fake_video_macro_average']
    ratios = summary['ratios']
    counts = summary['counts']
    print("\n" + "=" * 60)
    print("Stage-1 proposal quality (fake-video macro average)")
    print("=" * 60)
    print(
        f"Average proposal Recall: {averages['proposal_recall']:.4f}"
    )
    print(
        "Video classification errors: "
        f"{counts['video_classification_errors']}/{counts['total_videos']} "
        f"({ratios['video_classification_error_ratio']:.2%})"
    )
    print(
        "Fake videos with low Recall: "
        f"{counts['low_recall_fake_videos']}/{counts['fake_videos']} "
        f"({ratios['low_recall_ratio_among_fake_videos']:.2%})"
    )
    print(
        "High-confidence classification errors: "
        f"{counts['high_confidence_classification_errors']}/{counts['total_videos']} "
        f"({ratios['high_confidence_classification_error_ratio_all_videos']:.2%})"
    )
    print(
        "Temporal FPR (GT-real time predicted fake): "
        f"{ratios['temporal_false_positive_rate']:.2%} "
        f"({counts['temporal_false_positives_0.01s']}/"
        f"{counts['temporal_false_positives_0.01s'] + counts['temporal_true_negatives_0.01s']})"
    )


def evaluate_trained_model(
    cfg,
    model_path,
    output_dir,
    device='cuda',
    device_ids=None,
    low_recall=0.5,
    high_confidence=0.8,
):
    """Evaluate a trained model and persist the results."""

    # Create the output directory.
    os.makedirs(output_dir, exist_ok=True)

    # Load the model.
    model = load_trained_model(cfg, model_path, device, device_ids)

    # Build the dataloaders.
    print("******* Building dataloaders *******")
    _, val_loader, test_fake_segments = generate_dataset_loader_from_json(cfg)

    # Detect binary mode.
    is_binary = cfg.get('mode', '') == 'binary'

    # Loss used by `eval_model`.
    if cfg['mode'] == 'binary':
        loss_ce = torch.nn.BCEWithLogitsLoss()
    else:
        loss_ce = torch.nn.CrossEntropyLoss()

    # Run evaluation.
    print("******* Running evaluation *******")
    pred_accuracy, video_list, pred_labels, true_labels, outpred, all_videos_results = eval_model(
        cfg, model, val_loader, loss_ce, cfg.get('val_batch_size', 16), test_fake_segments
    )
    proposal_quality_summary, fake_proposal_rows = analyze_fake_video_proposals(
        all_videos_results['per_video_results'],
        low_recall=low_recall,
        high_confidence=high_confidence,
    )

    # Read the original test JSON.
    with open(cfg['test_json_path'], 'r', encoding='utf-8') as f:
        test_json_data = json.load(f)
    if cfg.get('val_videos', 0):
        test_json_data = test_json_data[:cfg['val_videos']]

    final_results = []

    for video_item in test_json_data:
        # Copy the original record.
        new_item = video_item.copy()

        # Build the matching key.
        video_path = video_item["video_path"]
        formatted_video_name = video_path[:-4].replace('/', '_')

        # Look up the prediction.
        if formatted_video_name in all_videos_results['per_video_results']:
            video_results = all_videos_results['per_video_results'][formatted_video_name]
            pred_segments = video_results.get('pred_segments', [])

            # Build `model_inference`.
            model_inference = {}
            if pred_segments:
                model_inference['type'] = 'fake'
                # Make sure each segment is a list (not a tuple).
                model_inference['segment'] = [list(seg) for seg in pred_segments]
            else:
                model_inference['type'] = 'real'
                model_inference['segment'] = []

            new_item['model_inference'] = model_inference
        else:
            raise ValueError(f'Missing prediction for {video_path}')

        final_results.append(new_item)

    # Save the new predictions JSON.
    predictions_json_path = os.path.join(output_dir, 'predictions.json')
    save_predictions_json(final_results, predictions_json_path)

    predictions_data = final_results

    # Optional: keep saving the legacy detailed results too.
    results_to_save = {
        'config': cfg,
        'model_path': model_path,
        'evaluation_results': {
            'overall_metrics': all_videos_results['overall_metrics'],
            'proposal_quality_summary': proposal_quality_summary,
            'per_video_results': {}
        }
    }

    # Make per-video results JSON-serializable.
    for video_name, video_results in all_videos_results['per_video_results'].items():
        # Cast numpy scalars/arrays to native Python types.
        serializable_results = {}
        for key, value in video_results.items():
            if isinstance(value, (np.integer, np.int64, np.int32)):
                serializable_results[key] = int(value)
            elif isinstance(value, (np.floating, np.float32, np.float64)):
                serializable_results[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, list):
                # Convert tuple segments to lists.
                if key == 'pred_segments' and value and isinstance(value[0], tuple):
                    serializable_results[key] = [list(seg) for seg in value]
                else:
                    serializable_results[key] = value
            else:
                serializable_results[key] = value
        results_to_save['evaluation_results']['per_video_results'][video_name] = serializable_results

    # Save detailed results JSON.
    output_json_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(results_to_save), f, indent=2, ensure_ascii=False)

    # Save the summary metrics.
    summary_results = {
        'model_path': model_path,
        'evaluation_time': pd.Timestamp.now().isoformat(),
        'overall_metrics': all_videos_results['overall_metrics'],
        'proposal_quality_summary': proposal_quality_summary,
    }

    summary_json_path = os.path.join(output_dir, 'summary_results.json')
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(summary_results), f, indent=2, ensure_ascii=False)

    # Save the per-video CSV.
    csv_data = []
    for video_name, video_results in all_videos_results['per_video_results'].items():
        row = {
            'video_name': video_name,
            'video_true_type': video_results['video_true_type'],
            'video_pred_type': video_results['video_pred_type'],
            'video_correct': video_results['video_correct'],
            'F1Det': video_results.get('F1Det', 0),
            'F1Det_precision': video_results.get('F1Det_precision', 0),
            'F1Det_recall': video_results.get('F1Det_recall', 0),
            'F1Loc': video_results.get('F1Loc', 0),
            'F1Loc@0.1': video_results.get('F1Loc@0.1', 0),
            'F1Loc@0.3': video_results.get('F1Loc@0.3', 0),
            'F1Loc@0.5': video_results.get('F1Loc@0.5', 0),
            'F1Loc@0.7': video_results.get('F1Loc@0.7', 0),
            'frame_f1': video_results.get('f1', 0),
            'frame_precision': video_results.get('precision', 0),
            'frame_recall': video_results.get('recall', 0),
            'loc_f1_0.1': video_results.get('loc_f1_0.1', 0),
            'loc_f1_0.3': video_results.get('loc_f1_0.3', 0),
            'loc_f1_0.5': video_results.get('loc_f1_0.5', 0),
            'loc_f1_0.7': video_results.get('loc_f1_0.7', 0),
            'avg_loc_f1': video_results.get('avg_loc_f1', 0)
        }
        proposal_quality = video_results.get('proposal_quality')
        if proposal_quality:
            row.update(proposal_quality)
        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, 'detailed_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')

    fake_proposal_csv_path = os.path.join(output_dir, 'fake_proposal_quality.csv')
    pd.DataFrame(fake_proposal_rows).to_csv(fake_proposal_csv_path, index=False, encoding='utf-8')

    print(f"[ok] Evaluation done. Results saved to: {output_dir}")
    print(f"   - predictions JSON: {predictions_json_path}")
    print(f"   - detailed JSON:    {output_json_path}")
    print(f"   - summary JSON:     {summary_json_path}")
    print(f"   - per-video CSV:    {csv_path}")
    print(f"   - fake proposal CSV:{fake_proposal_csv_path}")

    # Print the overall metrics.
    print("\n" + "=" * 60)
    print("Evaluation summary")
    print("=" * 60)
    print_evaluation_metrics(all_videos_results['overall_metrics'])
    print_proposal_quality_summary(proposal_quality_summary)

    return results_to_save, predictions_data


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Evaluate a trained temporal-segmentation model.')
    parser.add_argument('--config', required=True, help='Path to the config file.')
    parser.add_argument('--model_path', required=True, help='Path to the trained checkpoint.')
    parser.add_argument('--output_dir', required=True, help='Directory to write evaluation results to.')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda only for the current evaluator).')
    parser.add_argument('--device-ids', default='0,1,2,3,4,5,6,7', help='CUDA device IDs: 0 for one GPU, or 0,1,2,3,4,5,6,7 for eight GPUs')
    parser.add_argument('--val-batch-size', type=int, default=None, help='Optional override for cfg.val_batch_size')
    parser.add_argument('--eval-micro-batch-size', type=int, default=0,
                        help='Windows per forward; 0 uses the complete logical batch.')
    parser.add_argument('--low-proposal-recall', type=float, default=0.5,
                        help='Proposal Recall below this is reported as low Recall (default: 0.5).')
    parser.add_argument('--high-confidence', type=float, default=0.8,
                        help='Decision/proposal confidence at or above this is high confidence (default: 0.8).')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-videos', type=int, default=0, help='0 evaluates every input video.')
    parser.add_argument('--eval-split', choices=['test', 'train'], default='test')
    parser.add_argument('--resume', action='store_true', help='Reuse completed inference batches.')
    parser.add_argument('--clean', action='store_true', help='Remove this evaluation run\'s outputs.')
    args = parser.parse_args()
    seed_all(args.seed)
    args.device_ids = parse_device_ids(args.device_ids)
    if args.device != 'cuda':
        raise ValueError('The current evaluation loop uses CUDA tensors; pass --device cuda.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by the current DeMamba evaluation loop')
    available = torch.cuda.device_count()
    if max(args.device_ids) >= available:
        raise ValueError(f'Requested --device-ids {args.device_ids}, but only {available} CUDA device(s) are visible')
    primary_device = args.device_ids[0]
    torch.cuda.set_device(primary_device)
    args.device = f'cuda:{primary_device}'

    # Sanity-check input paths.
    assert os.path.exists(args.config), f"Config file not found: {args.config}"
    assert os.path.exists(args.model_path), f"Checkpoint file not found: {args.model_path}"

    # Load the config.
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    if args.val_batch_size is not None:
        if args.val_batch_size <= 0:
            raise ValueError('--val-batch-size must be positive')
        cfg['val_batch_size'] = args.val_batch_size
    for name, value in (
        ('--low-proposal-recall', args.low_proposal_recall),
        ('--high-confidence', args.high_confidence),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f'{name} must be in [0, 1]')

    print("Temporal-segmentation model evaluation")
    print("=" * 50)
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.model_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device:     {args.device}  IDs: {args.device_ids}")
    print("=" * 50)

    if args.max_videos < 0:
        raise ValueError('--max-videos must be non-negative')
    cfg.update(val_videos=args.max_videos, device_ids=args.device_ids)
    if args.eval_micro_batch_size < 0:
        raise ValueError('--eval-micro-batch-size must be non-negative')
    cfg['eval_micro_batch_size'] = args.eval_micro_batch_size
    if args.eval_split == 'train':
        cfg['test_json_path'] = cfg['train_json_path']
    prepare_run(args.output_dir, cfg, args.seed, args.resume, args.clean, args.model_path)
    cfg['_evaluation_cache'] = str(Path(args.output_dir) / 'inference_cache.pth')
    # Run evaluation.
    results, predictions_data = evaluate_trained_model(
        cfg,
        args.model_path,
        args.output_dir,
        args.device,
        args.device_ids,
        low_recall=args.low_proposal_recall,
        high_confidence=args.high_confidence,
    )

    print("[ok] Evaluation pipeline finished.")


if __name__ == '__main__':
    main()
