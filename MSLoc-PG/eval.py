import os
import json
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from dataloader import generate_dataset_loader_from_json, select_video_records

import models
from util import build_model
from util import print_evaluation_metrics, eval_model

def load_trained_model(cfg, model_path, device='cuda'):
    """Load a trained checkpoint."""
    model = build_model(cfg['model'])
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # Adapt between single-GPU and multi-GPU formats.
        if not any(key.startswith('module.') for key in state_dict.keys()) and torch.cuda.device_count() > 1:
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif any(key.startswith('module.') for key in state_dict.keys()) and torch.cuda.device_count() == 1:
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

def evaluate_trained_model(cfg, model_path, output_dir, device='cuda'):
    """Evaluate a trained model and persist the results."""
    
    # Create the output directory.
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model.
    model = load_trained_model(cfg, model_path, device)
    
    # Build the dataloaders.
    print("******* Building dataloaders *******")
    _, val_loader, test_fake_segments = generate_dataset_loader_from_json(cfg, include_train=False)
    if len(val_loader) == 0:
        raise ValueError('No evaluation windows; check annotations and frame paths')
    
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
    
    # Read the original test JSON.
    with open(cfg['test_json_path'], 'r', encoding='utf-8') as f:
        test_json_data = json.load(f)
    test_json_data = select_video_records(test_json_data, cfg.get('val_videos', 0))
        
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
            if cfg.get('val_videos', 0) > 0:
                raise ValueError(f'No prediction produced for selected video: {video_path}')
            # Preserve the original full-evaluation output behavior.
            new_item['model_inference'] = {
                'type': 'real',
                'segment': []
            }
            
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
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    
    # Save the summary metrics.
    summary_results = {
        'model_path': model_path,
        'evaluation_time': pd.Timestamp.now().isoformat(),
        'overall_metrics': all_videos_results['overall_metrics']
    }
    
    summary_json_path = os.path.join(output_dir, 'summary_results.json')
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    
    # Save the per-video CSV.
    csv_data = []
    for video_name, video_results in all_videos_results['per_video_results'].items():
        row = {
            'video_name': video_name,
            'video_true_type': video_results['video_true_type'],
            'video_pred_type': video_results['video_pred_type'],
            'video_correct': video_results['video_correct'],
            'frame_f1': video_results.get('f1', 0),
            'frame_precision': video_results.get('precision', 0),
            'frame_recall': video_results.get('recall', 0),
            'loc_f1_0.1': video_results.get('loc_f1_0.1', 0),
            'loc_f1_0.3': video_results.get('loc_f1_0.3', 0),
            'loc_f1_0.5': video_results.get('loc_f1_0.5', 0),
            'loc_f1_0.7': video_results.get('loc_f1_0.7', 0),
            'avg_loc_f1': video_results.get('avg_loc_f1', 0)
        }
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, 'detailed_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    print(f"[ok] Evaluation done. Results saved to: {output_dir}")
    print(f"   - predictions JSON: {predictions_json_path}")
    print(f"   - detailed JSON:    {output_json_path}")
    print(f"   - summary JSON:     {summary_json_path}")
    print(f"   - per-video CSV:    {csv_path}")
    
    # Print the overall metrics.
    print("\n" + "=" * 60)
    print("Evaluation summary")
    print("=" * 60)
    print_evaluation_metrics(all_videos_results['overall_metrics'])
    
    return results_to_save, predictions_data


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Evaluate a trained temporal-segmentation model.')
    parser.add_argument('--config', required=True, help='Path to the config file.')
    parser.add_argument('--model_path', required=True, help='Path to the trained checkpoint.')
    parser.add_argument('--output_dir', required=True, help='Directory to write evaluation results to.')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu).')
    parser.add_argument('--max-videos', '--val-videos', dest='max_videos', type=int, default=0,
                        help='Evaluate the first N complete videos; 0 uses all videos')
    parser.add_argument('--batch-size', '--val-batch-size', dest='batch_size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    
    args = parser.parse_args()
    
    # Sanity-check input paths.
    assert os.path.exists(args.config), f"Config file not found: {args.config}"
    assert os.path.exists(args.model_path), f"Checkpoint file not found: {args.model_path}"
    
    # Load the config.
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    cfg['val_videos'] = args.max_videos
    cfg['show_progress'] = True
    if args.batch_size is not None:
        cfg['val_batch_size'] = args.batch_size
    if args.num_workers is not None:
        cfg['num_workers'] = args.num_workers
    if cfg['val_videos'] < 0 or cfg['val_batch_size'] <= 0 or cfg['num_workers'] < 0:
        raise ValueError('Video limit and workers must be nonnegative; batch size must be positive')
    
    print("Temporal-segmentation model evaluation")
    print("=" * 50)
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.model_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device:     {args.device}")
    print("=" * 50)
    
    # Run evaluation.
    results, predictions_data = evaluate_trained_model(cfg, args.model_path, args.output_dir, args.device)
    
    print("[ok] Evaluation pipeline finished.")


if __name__ == '__main__':
    main()

