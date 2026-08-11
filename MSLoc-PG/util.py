import models
import torch
import math
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

from typing import List, Dict, Tuple, Any
import warnings

class TemporalSegmentationEvaluator:
    def __init__(self, window_duration: float = 2.0, fps: int = 8):
        self.window_duration = window_duration
        self.fps = fps

    def calculate_segment_iou(self, seg1: Tuple[float, float], seg2: Tuple[float, float]) -> float:
        start1, end1 = seg1
        start2, end2 = seg2

        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap = max(0.0, overlap_end - overlap_start)

        union = (end1 - start1) + (end2 - start2) - overlap
        return overlap / union if union > 0 else 0.0

    def calculate_loc_f1_for_iou_threshold(self, pred_segments: List[Tuple[float, float]],
                                         gt_segments: List[Tuple[float, float]],
                                         iou_threshold: float = 0.5) -> float:
        if not gt_segments and not pred_segments:
            return 1.0
        elif not gt_segments:
            return 0.0
        elif not pred_segments:
            return 0.0

        matched_gt = set()
        matched_pred = set()

        tp = 0
        for pred_idx, pred_seg in enumerate(pred_segments):
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt_seg in enumerate(gt_segments):
                if gt_idx not in matched_gt:
                    iou = self.calculate_segment_iou(pred_seg, gt_seg)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

            if best_gt_idx != -1 and best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)

        fp = len(pred_segments) - len(matched_pred)
        fn = len(gt_segments) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1

    def calculate_frame_level_f1(self, pred_scores: List,
                           gt_labels: List,
                           threshold: float = 0.5) -> Dict[str, float]:
        if len(pred_scores) != len(gt_labels):
            warnings.warn(f"Predicted count ({len(pred_scores)}) does not match GT count ({len(gt_labels)})")
            min_len = min(len(pred_scores), len(gt_labels))
            pred_scores = pred_scores[:min_len]
            gt_labels = gt_labels[:min_len]

        y_pred = []
        y_true = []

        for pred, gt in zip(pred_scores, gt_labels):
            pred_fake_score = float(pred)
            if isinstance(gt, (list, np.ndarray)) and len(gt) >= 4:
                true_fake = 0 if gt[0] > 0.5 else 1
            else:
                true_fake = 1 if gt > 0.5 else 0

            pred_fake = 1 if pred_fake_score > threshold else 0
            y_pred.append(pred_fake)
            y_true.append(true_fake)

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        window_results = [
            {
                "index": i,
                "prediction": "fake" if y_pred[i] == 1 else "real",
                "feature_distance": pred_scores[i],
                "start_time": i * 2,
                "end_time": (i + 1) * 2,
                "gt": "fake" if y_true[i] == 1 else "real"
            }
            for i in range(len(y_pred))
        ]

        return {
            'window_results': window_results,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    def calculate_detection_window_counts(self,
                                      pred_segments: List[Tuple[float, float]],
                                      gt_segments: List[Tuple[float, float]],
                                      video_duration: float,
                                      window_size: float = 0.01) -> Dict[str, int]:
        """Compute F1Det counts on the paper's 0.01-second temporal grid."""
        if video_duration <= 0:
            return {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

        num_windows = int(math.ceil(video_duration / window_size))
        pred_mask = np.zeros(num_windows, dtype=bool)
        gt_mask = np.zeros(num_windows, dtype=bool)

        def mark_segments(mask, segments):
            for segment in segments:
                if not segment or len(segment) < 2:
                    continue
                start, end = float(segment[0]), float(segment[1])
                start = max(0.0, min(start, video_duration))
                end = max(0.0, min(end, video_duration))
                if end <= start:
                    continue
                start_idx = max(0, int(math.floor(start / window_size)))
                end_idx = min(num_windows, int(math.ceil(end / window_size)))
                if end_idx > start_idx:
                    mask[start_idx:end_idx] = True

        mark_segments(pred_mask, pred_segments)
        mark_segments(gt_mask, gt_segments)

        tp = int(np.sum(pred_mask & gt_mask))
        fp = int(np.sum(pred_mask & ~gt_mask))
        fn = int(np.sum(~pred_mask & gt_mask))
        tn = int(np.sum(~pred_mask & ~gt_mask))
        return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

    def calculate_loc_counts_for_iou_threshold(self,
                                           pred_segments: List[Tuple[float, float]],
                                           gt_segments: List[Tuple[float, float]],
                                           iou_threshold: float) -> Dict[str, int]:
        """Paper-style segment retrieval counts for one IoU threshold."""
        if not pred_segments and not gt_segments:
            return {'tp': 0, 'fp': 0, 'fn': 0}
        if not pred_segments:
            return {'tp': 0, 'fp': 0, 'fn': len(gt_segments)}
        if not gt_segments:
            return {'tp': 0, 'fp': len(pred_segments), 'fn': 0}

        pairs = []
        for pred_idx, pred_seg in enumerate(pred_segments):
            for gt_idx, gt_seg in enumerate(gt_segments):
                iou = self.calculate_segment_iou(pred_seg, gt_seg)
                if iou >= iou_threshold:
                    pairs.append((iou, pred_idx, gt_idx))
        pairs.sort(key=lambda item: item[0], reverse=True)

        matched_pred = set()
        matched_gt = set()
        for _, pred_idx, gt_idx in pairs:
            if pred_idx in matched_pred or gt_idx in matched_gt:
                continue
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)

        tp = len(matched_pred)
        fp = len(pred_segments) - tp
        fn = len(gt_segments) - len(matched_gt)
        return {'tp': tp, 'fp': fp, 'fn': fn}

    @staticmethod
    def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {'precision': precision, 'recall': recall, 'f1': f1}

    def convert_predictions_to_segments(self, pred_scores: List,
                                    video_duration: float,
                                    threshold: float = 0.5) -> List[Tuple[float, float]]:
        if not pred_scores:
            return []

        fake_scores = []
        for score in pred_scores:
            if isinstance(score, np.ndarray) and score.ndim == 0:
                fake_scores.append(score.item())
            elif isinstance(score, np.ndarray) and score.size == 1:
                fake_scores.append(score.item())
            else:
                fake_scores.append(float(score))

        binary_preds = [1 if score > threshold else 0 for score in fake_scores]

        segments = []
        current_start = None

        for i, pred in enumerate(binary_preds):
            window_start = i * self.window_duration
            window_end = min((i + 1) * self.window_duration, video_duration)

            if pred == 1 and current_start is None:
                current_start = window_start
            elif pred == 0 and current_start is not None:
                segments.append((current_start, window_start))
                current_start = None

        if current_start is not None:
            segments.append((current_start, video_duration))

        return segments

    def postprocess_video_segments(self, pred_segments: List[Tuple[float, float]],
                                 video_name: str,
                                 video_type: str) -> List[Tuple[float, float]]:
        processed_segments = pred_segments.copy()


        return processed_segments

    def evaluate_video_segmentation(self,
                                pred_scores: List,
                                gt_labels: List,
                                video_duration: float,
                                gt_fake_segments: List[Tuple[float, float]],
                                video_type: str,
                                video_name: str = "") -> Dict[str, Any]:
        results = {}

        if video_type == 'fake':
            frame_metrics = self.calculate_frame_level_f1(pred_scores, gt_labels, threshold=0.5)
        else:
            frame_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 0}
        results.update(frame_metrics)

        pred_segments = self.convert_predictions_to_segments(pred_scores, video_duration, threshold=0.5)

        if video_type == 'fake':
            pred_segments = self.postprocess_video_segments(pred_segments, video_name, video_type)

        results['pred_segments'] = pred_segments

        det_counts = self.calculate_detection_window_counts(
            pred_segments=pred_segments,
            gt_segments=gt_fake_segments,
            video_duration=video_duration,
            window_size=0.01
        )
        det_metrics = self.precision_recall_f1(det_counts['tp'], det_counts['fp'], det_counts['fn'])
        results.update({
            'F1Det': det_metrics['f1'],
            'F1Det_precision': det_metrics['precision'],
            'F1Det_recall': det_metrics['recall'],
            'F1Det_tp': det_counts['tp'],
            'F1Det_fp': det_counts['fp'],
            'F1Det_fn': det_counts['fn'],
            'F1Det_tn': det_counts['tn']
        })

        iou_thresholds = [0.1, 0.3, 0.5, 0.7]
        if video_type == 'fake' and gt_fake_segments:
            loc_f1_scores = {}

            for threshold in iou_thresholds:
                f1_score = self.calculate_loc_f1_for_iou_threshold(pred_segments, gt_fake_segments, threshold)
                loc_f1_scores[f'loc_f1_{threshold}'] = f1_score

            avg_loc_f1 = np.mean(list(loc_f1_scores.values()))
            loc_f1_scores['avg_loc_f1'] = avg_loc_f1
            results.update(loc_f1_scores)
        else:
            results.update({
                'loc_f1_0.1': 0.0, 'loc_f1_0.3': 0.0, 'loc_f1_0.5': 0.0, 'loc_f1_0.7': 0.0, 'avg_loc_f1': 0.0
            })

        for threshold in iou_thresholds:
            loc_counts = self.calculate_loc_counts_for_iou_threshold(pred_segments, gt_fake_segments, threshold)
            key = str(threshold)
            loc_metrics = self.precision_recall_f1(loc_counts['tp'], loc_counts['fp'], loc_counts['fn'])
            results[f'F1Loc@{key}'] = loc_metrics['f1']
            results[f'F1Loc@{key}_precision'] = loc_metrics['precision']
            results[f'F1Loc@{key}_recall'] = loc_metrics['recall']
            results[f'F1Loc@{key}_tp'] = loc_counts['tp']
            results[f'F1Loc@{key}_fp'] = loc_counts['fp']
            results[f'F1Loc@{key}_fn'] = loc_counts['fn']

        results['F1Loc'] = float(np.mean([results[f'F1Loc@{threshold}'] for threshold in ['0.1', '0.3', '0.5', '0.7']]))

        fake_scores = []
        for score in pred_scores:
            if isinstance(score, np.ndarray) and score.ndim == 0:
                fake_scores.append(score.item())
            elif isinstance(score, np.ndarray) and score.size == 1:
                fake_scores.append(score.item())
            else:
                fake_scores.append(float(score))

        has_fake_prediction = any(score > 0.5 for score in fake_scores)
        video_pred_type = 'fake' if has_fake_prediction else 'real'
        video_true_type = video_type

        results['video_pred_type'] = video_pred_type
        results['video_true_type'] = video_true_type
        results['video_correct'] = True if video_pred_type == video_true_type else False

        results['fake_segments'] = [[i[0], i[1]] for i in results['pred_segments']]
        results['has_fake'] = results['fake_segments'] != []
        results['fake_window_count'] = len(results['fake_segments'])

        return results

    def evaluate_all_videos(self, video_data_list: List[Dict]) -> Dict[str, Any]:
        all_results = {}
        video_level_results = []

        for i, video_data in enumerate(video_data_list):
            video_name = video_data.get('video_name', f'video_{i}')

            results = self.evaluate_video_segmentation(
                pred_scores=video_data['pred_scores'],
                gt_labels=video_data['gt_labels'],
                video_duration=video_data['video_duration'],
                gt_fake_segments=video_data['gt_fake_segments'],
                video_type=video_data['video_type'],
                video_name=video_name
            )

            all_results[video_name] = results
            video_level_results.append(results)

        return self._calculate_overall_metrics(all_results, video_level_results)

    def _calculate_overall_metrics(self, all_results: Dict[str, Any], video_level_results: List[Dict]) -> Dict[str, Any]:
        video_accuracy = np.mean([r['video_correct'] for r in video_level_results])

        fake_video_results = [r for r in video_level_results if r['video_true_type'] == 'fake']
        real_video_results = [r for r in video_level_results if r['video_true_type'] == 'real']

        if fake_video_results:
            frame_f1_scores = [r['f1'] for r in fake_video_results]
            frame_precision_scores = [r['precision'] for r in fake_video_results]
            frame_recall_scores = [r['recall'] for r in fake_video_results]

            avg_loc_f1_scores = [r['avg_loc_f1'] for r in fake_video_results if 'avg_loc_f1' in r]
            loc_f1_01_scores = [r['loc_f1_0.1'] for r in fake_video_results if 'loc_f1_0.1' in r]
            loc_f1_03_scores = [r['loc_f1_0.3'] for r in fake_video_results if 'loc_f1_0.3' in r]
            loc_f1_05_scores = [r['loc_f1_0.5'] for r in fake_video_results if 'loc_f1_0.5' in r]
            loc_f1_07_scores = [r['loc_f1_0.7'] for r in fake_video_results if 'loc_f1_0.7' in r]
        else:
            frame_f1_scores = frame_precision_scores = frame_recall_scores = [0.0]
            avg_loc_f1_scores = loc_f1_01_scores = loc_f1_03_scores = loc_f1_05_scores = loc_f1_07_scores = [0.0]

        det_tp = sum(r.get('F1Det_tp', 0) for r in video_level_results)
        det_fp = sum(r.get('F1Det_fp', 0) for r in video_level_results)
        det_fn = sum(r.get('F1Det_fn', 0) for r in video_level_results)
        det_tn = sum(r.get('F1Det_tn', 0) for r in video_level_results)
        det_metrics = self.precision_recall_f1(det_tp, det_fp, det_fn)

        loc_thresholds = ['0.1', '0.3', '0.5', '0.7']
        paper_loc_metrics = {}
        loc_f1_values = []
        for threshold in loc_thresholds:
            tp = sum(r.get(f'F1Loc@{threshold}_tp', 0) for r in video_level_results)
            fp = sum(r.get(f'F1Loc@{threshold}_fp', 0) for r in video_level_results)
            fn = sum(r.get(f'F1Loc@{threshold}_fn', 0) for r in video_level_results)
            metrics = self.precision_recall_f1(tp, fp, fn)
            paper_loc_metrics[f'F1Loc@{threshold}'] = metrics['f1']
            paper_loc_metrics[f'F1Loc@{threshold}_precision'] = metrics['precision']
            paper_loc_metrics[f'F1Loc@{threshold}_recall'] = metrics['recall']
            paper_loc_metrics[f'F1Loc@{threshold}_tp'] = tp
            paper_loc_metrics[f'F1Loc@{threshold}_fp'] = fp
            paper_loc_metrics[f'F1Loc@{threshold}_fn'] = fn
            loc_f1_values.append(metrics['f1'])

        paper_f1det = det_metrics['f1']
        paper_f1loc = float(np.mean(loc_f1_values)) if loc_f1_values else 0.0

        overall_metrics = {
            'F1Det': paper_f1det,
            'F1Loc': paper_f1loc,
            'paper': {
                'F1Det': paper_f1det,
                'F1Det_precision': det_metrics['precision'],
                'F1Det_recall': det_metrics['recall'],
                'F1Det_tp': det_tp,
                'F1Det_fp': det_fp,
                'F1Det_fn': det_fn,
                'F1Det_tn': det_tn,
                'F1Loc': paper_f1loc,
                **paper_loc_metrics
            },
            'frame_level': {
                'f1': paper_f1det,
                'precision': det_metrics['precision'],
                'recall': det_metrics['recall'],
                'legacy_fake_video_mean_f1': np.mean(frame_f1_scores).item(),
                'legacy_fake_video_mean_precision': np.mean(frame_precision_scores).item(),
                'legacy_fake_video_mean_recall': np.mean(frame_recall_scores).item()
            },
            'video_level': {
                'accuracy': video_accuracy.item()
            },
            'localization_fake_videos': {
                'avg_loc_f1': paper_f1loc,
                'loc_f1_0.1': paper_loc_metrics['F1Loc@0.1'],
                'loc_f1_0.3': paper_loc_metrics['F1Loc@0.3'],
                'loc_f1_0.5': paper_loc_metrics['F1Loc@0.5'],
                'loc_f1_0.7': paper_loc_metrics['F1Loc@0.7'],
                'legacy_fake_video_mean_avg_loc_f1': np.mean(avg_loc_f1_scores).item(),
                'legacy_fake_video_mean_loc_f1_0.1': np.mean(loc_f1_01_scores).item(),
                'legacy_fake_video_mean_loc_f1_0.3': np.mean(loc_f1_03_scores).item(),
                'legacy_fake_video_mean_loc_f1_0.5': np.mean(loc_f1_05_scores).item(),
                'legacy_fake_video_mean_loc_f1_0.7': np.mean(loc_f1_07_scores).item()
            },
            'counts': {
                'total_videos': len(video_level_results),
                'fake_videos': len(fake_video_results),
                'real_videos': len(real_video_results)
            }
        }

        return {
            'per_video_results': all_results,
            'overall_metrics': overall_metrics
        }

def print_evaluation_metrics(metrics_dict):
    print("Temporal segmentation evaluation")
    print("=" * 50)

    paper_metrics = metrics_dict.get('paper', {})
    print("Paper metrics:")
    print(f"  F1Det: {paper_metrics.get('F1Det', metrics_dict.get('F1Det', 0)):.4f}  "
          f"Precision: {paper_metrics.get('F1Det_precision', 0):.4f}  "
          f"Recall: {paper_metrics.get('F1Det_recall', 0):.4f}")
    print(f"  F1Loc: {paper_metrics.get('F1Loc', metrics_dict.get('F1Loc', 0)):.4f}  "
          f"F1Loc@0.1: {paper_metrics.get('F1Loc@0.1', 0):.4f}  "
          f"F1Loc@0.3: {paper_metrics.get('F1Loc@0.3', 0):.4f}  "
          f"F1Loc@0.5: {paper_metrics.get('F1Loc@0.5', 0):.4f}  "
          f"F1Loc@0.7: {paper_metrics.get('F1Loc@0.7', 0):.4f}")

    frame_metrics = metrics_dict.get('frame_level', {})
    print("\nFrame-level metrics (legacy aliases; now paper-style global F1Det):")
    print(f"  F1: {frame_metrics.get('f1', 0):.4f}  Precision: {frame_metrics.get('precision', 0):.4f}  Recall: {frame_metrics.get('recall', 0):.4f}")

    video_metrics = metrics_dict.get('video_level', {})
    print("\nVideo-level metrics:")
    print(f"  Classification accuracy: {video_metrics.get('accuracy', 0):.4f}")

    loc_metrics = metrics_dict.get('localization_fake_videos', {})
    print("\nLocalization metrics (legacy aliases; now paper-style global F1Loc):")
    print(f"  Avg Loc F1: {loc_metrics.get('avg_loc_f1', 0):.4f}  F1@0.1: {loc_metrics.get('loc_f1_0.1', 0):.4f}  F1@0.3: {loc_metrics.get('loc_f1_0.3', 0):.4f}  F1@0.5: {loc_metrics.get('loc_f1_0.5', 0):.4f}  F1@0.7: {loc_metrics.get('loc_f1_0.7', 0):.4f}")

    counts = metrics_dict.get('counts', {})
    print("\nDataset stats:")
    print(f"  Total: {counts.get('total_videos', 0)}  Fake: {counts.get('fake_videos', 0)}  Real: {counts.get('real_videos', 0)}")
    print("=" * 50)

def build_model(model_name):
    if model_name == 'XCLIP_DeMamba_4':
        return models.XCLIP_DeMamba(class_num=4)
    raise ValueError(f"Unsupported model_name: {model_name}")

def eval_model(model, inference_loader, test_fake_segments):
    model.eval()
    outpred_list = []
    gt_label_list = []
    video_list = []
    print("******** Start Testing. ********")

    video_seg_pred, video_seg_gt = {}, {}
    video_loc_gt = {}

    with torch.no_grad():
        for window_idx, input, target, _class_label, video_id in tqdm(
            inference_loader, desc="Inference", total=len(inference_loader)
        ):
            input = input[:,0]
            varInput = torch.autograd.Variable(input.float().cuda())
            varTarget = torch.autograd.Variable(target.contiguous().cuda())

            logit = model(varInput)

            pred_probs = torch.softmax(logit, dim=1)
            fake_scores = pred_probs[:, 1:].sum(dim=1)
            outpred_list.append(fake_scores.cpu().detach().numpy())

            gt_label_list.append(varTarget.cpu().detach().numpy())
            video_list.append(video_id)

            ''' Per-video predictions. '''
            for j in range(len(video_id)):
                video_id_j = video_id[j]
                if video_id_j not in video_seg_pred:
                    video_seg_pred[video_id_j] = []
                    video_seg_gt[video_id_j] = []
                    video_loc_gt[video_id_j] = test_fake_segments[video_id_j]

                pred_probs = torch.softmax(logit[j], dim=0)
                pred_score = (pred_probs[1:].max() > pred_probs[0]).float().cpu().detach().numpy()


                video_seg_pred[video_id_j].append((window_idx[j], pred_score))
                video_seg_gt[video_id_j].append((window_idx[j], target[j].cpu().detach().numpy()))

    loc_data = []
    for video_id in video_seg_pred:
        video_seg_pred[video_id] = sorted(video_seg_pred[video_id], key=lambda x: x[0])
        video_seg_gt[video_id] = sorted(video_seg_gt[video_id], key=lambda x: x[0])
        assert len(video_seg_pred[video_id]) == len(video_seg_gt[video_id])

        pred_scores = [item[1] for item in video_seg_pred[video_id]]
        gt_labels = [item[1] for item in video_seg_gt[video_id]]


        loc_data.append({
            'video_id': video_id,
            'pred_scores': pred_scores,
            'gt_labels': gt_labels,
            'video_duration': video_loc_gt[video_id]['duration'],
            'gt_fake_segments': video_loc_gt[video_id]['fake_segments'],
            'video_type': video_loc_gt[video_id]['type'],
            'video_name': video_id
        })

    evaluator = TemporalSegmentationEvaluator(window_duration=2.0, fps=8)

    all_videos_results = evaluator.evaluate_all_videos(loc_data)

    print_evaluation_metrics(all_videos_results['overall_metrics'])

    outpred = np.concatenate(outpred_list, 0)
    gt_label = np.concatenate(gt_label_list, 0)
    video_list = np.concatenate(video_list, 0)

    if gt_label.ndim > 1 and gt_label.shape[1] >= 4:
        true_labels = np.where(gt_label[:, 0] > 0.5, 0, 1)
    else:
        true_labels = (gt_label > 0.5).astype(int)
    pred_labels = [1 if item > 0.5 else 0 for item in outpred]
    pred_accuracy = accuracy_score(true_labels, pred_labels)

    return pred_accuracy, video_list, pred_labels, true_labels, outpred, all_videos_results
