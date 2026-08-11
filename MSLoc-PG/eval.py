import os
import json
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from dataloader import build_inference_loader
from util import build_model
from util import print_evaluation_metrics, eval_model

def load_checkpoint(cfg, model_path, device='cuda'):
    model = build_model(cfg['model'])
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
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


def build_predictions_json(test_json_data, per_video_results):
    final_results = []
    missing_predictions = []

    for video_item in test_json_data:
        video_path = video_item["video_path"]
        formatted_video_name = video_path[:-4].replace('/', '_')
        if formatted_video_name not in per_video_results:
            missing_predictions.append(video_path)
            continue

        video_results = per_video_results[formatted_video_name]
        pred_segments = video_results.get('pred_segments', [])
        model_inference = {
            'type': 'fake' if pred_segments else 'real',
            'segment': [list(segment) for segment in pred_segments],
        }
        new_item = video_item.copy()
        new_item['model_inference'] = model_inference
        final_results.append(new_item)

    if missing_predictions:
        preview = '\n'.join(missing_predictions[:20])
        raise RuntimeError(
            f"No prediction was produced for {len(missing_predictions)} video(s):\n{preview}"
        )
    return final_results

def run_inference(cfg, model_path, output_dir, device='cuda'):
    os.makedirs(output_dir, exist_ok=True)

    model = load_checkpoint(cfg, model_path, device)

    print("******* Building inference loader *******")
    val_loader, test_fake_segments = build_inference_loader(cfg)

    print("******* Running evaluation *******")
    pred_accuracy, video_list, pred_labels, true_labels, outpred, all_videos_results = eval_model(
        model, val_loader, test_fake_segments
    )

    with open(cfg['test_json_path'], 'r', encoding='utf-8') as f:
        test_json_data = json.load(f)

    final_results = build_predictions_json(
        test_json_data,
        all_videos_results['per_video_results'],
    )

    predictions_json_path = os.path.join(output_dir, 'predictions.json')
    save_predictions_json(final_results, predictions_json_path)

    predictions_data = final_results

    results_to_save = {
        'config': cfg,
        'model_path': model_path,
        'evaluation_results': {
            'overall_metrics': all_videos_results['overall_metrics'],
            'per_video_results': {}
        }
    }

    for video_name, video_results in all_videos_results['per_video_results'].items():
        serializable_results = {}
        for key, value in video_results.items():
            if isinstance(value, (np.integer, np.int64, np.int32)):
                serializable_results[key] = int(value)
            elif isinstance(value, (np.floating, np.float32, np.float64)):
                serializable_results[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, list):
                if key == 'pred_segments' and value and isinstance(value[0], tuple):
                    serializable_results[key] = [list(seg) for seg in value]
                else:
                    serializable_results[key] = value
            else:
                serializable_results[key] = value
        results_to_save['evaluation_results']['per_video_results'][video_name] = serializable_results

    output_json_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)

    summary_results = {
        'model_path': model_path,
        'evaluation_time': pd.Timestamp.now().isoformat(),
        'overall_metrics': all_videos_results['overall_metrics']
    }

    summary_json_path = os.path.join(output_dir, 'summary_results.json')
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)

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
        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, 'detailed_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')

    print(f"[ok] Evaluation done. Results saved to: {output_dir}")
    print(f"   - predictions JSON: {predictions_json_path}")
    print(f"   - detailed JSON:    {output_json_path}")
    print(f"   - summary JSON:     {summary_json_path}")
    print(f"   - per-video CSV:    {csv_path}")

    print("\n" + "=" * 60)
    print("Evaluation summary")
    print("=" * 60)
    print_evaluation_metrics(all_videos_results['overall_metrics'])

    return results_to_save, predictions_data


def main():
    parser = argparse.ArgumentParser(description='Run the released Stage-1 proposal model.')
    parser.add_argument('--config', required=True, help='Path to the config file.')
    parser.add_argument('--model_path', required=True, help='Path to the released checkpoint.')
    parser.add_argument('--output_dir', required=True, help='Directory to write evaluation results to.')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu).')

    args = parser.parse_args()

    assert os.path.exists(args.config), f"Config file not found: {args.config}"
    assert os.path.exists(args.model_path), f"Checkpoint file not found: {args.model_path}"

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    print("Temporal-segmentation model evaluation")
    print("=" * 50)
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.model_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device:     {args.device}")
    print("=" * 50)

    results, predictions_data = run_inference(cfg, args.model_path, args.output_dir, args.device)

    print("[ok] Evaluation pipeline finished.")


if __name__ == '__main__':
    main()
