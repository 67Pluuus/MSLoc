import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def is_fake_annotation(video_entry, annotation):
    """Support both TASLE annotation schemas without accepting explicit real segments."""
    if video_entry.get("type") != "fake" or "segment" not in annotation:
        return False
    segment_label = annotation.get("segment_label")
    return segment_label is None or segment_label == "fake"

def calculate_iou(seg1, seg2):
    start1, end1 = seg1
    start2, end2 = seg2

    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)

    intersection = max(0, intersection_end - intersection_start)
    union = (end1 - start1) + (end2 - start2) - intersection

    if union == 0:
        return 0
    return intersection / union

def get_matches(pred_segs, gt_segs, threshold):
    if not pred_segs and not gt_segs:
        return 0, 0, 0
    if not pred_segs:
        return 0, 0, len(gt_segs)
    if not gt_segs:
        return 0, len(pred_segs), 0

    ious = np.zeros((len(pred_segs), len(gt_segs)))
    for i, p in enumerate(pred_segs):
        for j, g in enumerate(gt_segs):
            ious[i, j] = calculate_iou(p, g)

    tp = 0
    matched_gt = set()
    matched_pred = set()

    pairs = []
    for i in range(len(pred_segs)):
        for j in range(len(gt_segs)):
            if ious[i, j] >= threshold:
                pairs.append((ious[i, j], i, j))
    pairs.sort(key=lambda x: x[0], reverse=True)

    for _, p_idx, g_idx in pairs:
        if p_idx not in matched_pred and g_idx not in matched_gt:
            matched_pred.add(p_idx)
            matched_gt.add(g_idx)
            tp += 1

    fp = len(pred_segs) - len(matched_pred)
    fn = len(gt_segs) - len(matched_gt)

    return tp, fp, fn

def evaluate(gt_json_file, infer_json_file, window_size=0.01):
    print(f"Loading GT: {gt_json_file}...")
    with open(gt_json_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    print(f"Loading Pred: {infer_json_file}...")
    with open(infer_json_file, 'r', encoding='utf-8') as f:
        infer_data = json.load(f)

    if not isinstance(gt_data, list) or not all(isinstance(item, dict) for item in gt_data):
        raise ValueError("Ground-truth JSON must be a list of objects")
    if not isinstance(infer_data, list) or not all(isinstance(item, dict) for item in infer_data):
        raise ValueError("Inference JSON must be a list of objects")
    if any(not isinstance(item.get('video_path'), str) for item in gt_data):
        raise ValueError("Every ground-truth record must contain a string video_path")
    if any(not isinstance(item.get('video_path'), str) for item in infer_data):
        raise ValueError("Every inference record must contain a string video_path")

    print(f"Total GT videos: {len(gt_data)}")

    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    gt_paths = [item['video_path'] for item in gt_data]
    pred_paths = [item['video_path'] for item in infer_data]
    if len(gt_paths) != len(set(gt_paths)):
        raise ValueError("Ground-truth JSON contains duplicate video_path entries")
    if len(pred_paths) != len(set(pred_paths)):
        raise ValueError("Inference JSON contains duplicate video_path entries")
    missing_predictions = sorted(set(gt_paths) - set(pred_paths))
    extra_predictions = sorted(set(pred_paths) - set(gt_paths))
    if missing_predictions or extra_predictions:
        raise ValueError(
            "Prediction split does not match ground truth: "
            f"missing={len(missing_predictions)}, extra={len(extra_predictions)}"
        )

    pred_map = {item['video_path']: item for item in infer_data}

    domains = set()
    for entry in gt_data:
        d = entry.get('tool_domain')
        if not d and 'annotations' in entry and entry['annotations']:
            d = entry['annotations'][0].get('tool_domain')
        domains.add(d if d else 'unknown')

    sorted_domains = sorted(list(domains)) + ['Total']
    thresholds = [0.1, 0.3, 0.5, 0.7]

    domain_stats = {}
    for d in sorted_domains:
        domain_stats[d] = {
            'y_true_vid': [],
            'y_pred_vid': [],
            'y_true_win': [],
            'y_pred_win': [],
            'stats_per_threshold': {th: {'tp': 0, 'fp': 0, 'fn': 0} for th in thresholds}
        }

    for gt_entry in gt_data:
        video_path = gt_entry['video_path']

        current_domain = gt_entry.get('tool_domain')
        if not current_domain and 'annotations' in gt_entry and gt_entry['annotations']:
            current_domain = gt_entry['annotations'][0].get('tool_domain')
        if not current_domain:
            current_domain = 'unknown'

        target_domains = [current_domain, 'Total']

        pred_entry = pred_map.get(video_path)

        if gt_entry['type'] == 'real':
            vid_gt = 0
        else:
            vid_gt = 1

        vid_pred = 0
        pred_segments = []

        if pred_entry:
            model_infer = pred_entry.get('model_inference', {})
            if model_infer.get('type') == 'fake':
                vid_pred = 1
            elif model_infer.get('type') == 'real':
                vid_pred = 0
            else:
                segs = model_infer.get('segment', [])
                if segs:
                    vid_pred = 1

            pred_segments = model_infer.get('segment', [])
            if pred_segments and isinstance(pred_segments[0], (int, float)):
                 pred_segments = [pred_segments]

        for d in target_domains:
            domain_stats[d]['y_true_vid'].append(vid_gt)
            domain_stats[d]['y_pred_vid'].append(vid_pred)

        gt_segments = []
        if gt_entry['type'] == 'fake':
            if 'annotations' in gt_entry:
                for ann in gt_entry['annotations']:
                    if is_fake_annotation(gt_entry, ann):
                        gt_segments.append(ann['segment'])

        curr_y_true_win = []
        curr_y_pred_win = []

        duration = gt_entry.get('duration', 0)
        if duration > 0:
            for start in np.arange(0, duration, window_size):
                end = min(start + window_size, duration)
                if end <= start: continue

                w_gt_label = 0
                for g_seg in gt_segments:
                    inter_start = max(start, g_seg[0])
                    inter_end = min(end, g_seg[1])
                    if inter_end > inter_start:
                        w_gt_label = 1
                        break

                w_pred_label = 0
                for p_seg in pred_segments:
                    inter_start = max(start, p_seg[0])
                    inter_end = min(end, p_seg[1])
                    if inter_end > inter_start:
                        w_pred_label = 1
                        break

                curr_y_true_win.append(w_gt_label)
                curr_y_pred_win.append(w_pred_label)

        for d in target_domains:
            domain_stats[d]['y_true_win'].extend(curr_y_true_win)
            domain_stats[d]['y_pred_win'].extend(curr_y_pred_win)

        for th in thresholds:
            tp, fp, fn = get_matches(pred_segments, gt_segments, th)
            for d in target_domains:
                domain_stats[d]['stats_per_threshold'][th]['tp'] += tp
                domain_stats[d]['stats_per_threshold'][th]['fp'] += fp
                domain_stats[d]['stats_per_threshold'][th]['fn'] += fn

    final_results = {}

    for domain in sorted_domains:
        stats = domain_stats[domain]
        if not stats['y_true_vid']:
            continue

        if domain == 'Total':
            continue

        print(f"\n--- Domain: {domain} ---")

        det_acc = accuracy_score(stats['y_true_vid'], stats['y_pred_vid'])

        loc_f1 = f1_score(stats['y_true_win'], stats['y_pred_win'], pos_label=1, zero_division=0)

        iou_f1s = []
        print(f"Segment-level metrics per threshold ({domain}):")
        for th in thresholds:
            tp = stats['stats_per_threshold'][th]['tp']
            fp = stats['stats_per_threshold'][th]['fp']
            fn = stats['stats_per_threshold'][th]['fn']

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            iou_f1s.append(f1)
            print(f"  IoU@{th}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        loc_iou = np.mean(iou_f1s)

        results = {
            "Det_Acc": det_acc,
            "Loc_F1": loc_f1,
            "Loc_IoU": loc_iou
        }
        final_results[domain] = results
        print(f"Results for {domain}: {json.dumps(results, indent=4)}")

    if len(final_results) > 0:
        total_det_acc = np.mean([final_results[d]["Det_Acc"] for d in final_results])
        total_loc_f1 = np.mean([final_results[d]["Loc_F1"] for d in final_results])
        total_loc_iou = np.mean([final_results[d]["Loc_IoU"] for d in final_results])

        final_results["Total"] = {
            "Det_Acc": total_det_acc,
            "Loc_F1": total_loc_f1,
            "Loc_IoU": total_loc_iou
        }
        print(f"\n--- Domain: Total (Simple Average) ---")
        print(f"Results for Total: {json.dumps(final_results['Total'], indent=4)}")

    print("\nFinal Results Summary (All Domains):")
    print(json.dumps(final_results, indent=4))
    return final_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", type=str, required=True, help="Path to GT JSON file")
    parser.add_argument("--infer_file", type=str, required=True, help="Path to Inference JSON file")
    parser.add_argument(
        "--window_size",
        type=float,
        default=0.01,
        help="Temporal grid in seconds for Loc_F1 (default: 0.01, matching the released evaluation).",
    )
    args = parser.parse_args()

    evaluate(args.gt_file, args.infer_file, args.window_size)
