import models
from pathlib import Path
from runtime import atomic_save
import time
import torch
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, average_precision_score, roc_auc_score

import numpy as np
from typing import List, Dict, Tuple, Any
import warnings

class TemporalSegmentationEvaluator:
    """Temporal-segmentation evaluator."""

    def __init__(self, window_duration: float = 2.0, fps: int = 8):
        """
        Args:
            window_duration: Window length in seconds.
            fps: Frame rate.
        """
        self.window_duration = window_duration
        self.fps = fps

    def calculate_segment_iou(self, seg1: Tuple[float, float], seg2: Tuple[float, float]) -> float:
        """Compute the IoU between two time segments."""
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
        """
        Compute the localization F1 at a given IoU threshold.

        Args:
            pred_segments: Predicted fake segments.
            gt_segments: Ground-truth fake segments.
            iou_threshold: IoU threshold for a true positive match.

        Returns:
            F1 score.
        """
        if not gt_segments and not pred_segments:
            return 1.0  # both empty -> perfect match
        elif not gt_segments:
            return 0.0  # predictions but no ground truth -> precision = 0
        elif not pred_segments:
            return 0.0  # ground truth but no predictions -> recall = 0

        # Match predicted segments against ground-truth segments.
        matched_gt = set()
        matched_pred = set()

        # True-positive count.
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

        fp = len(pred_segments) - len(matched_pred)  # unmatched predictions
        fn = len(gt_segments) - len(matched_gt)      # unmatched ground truths

        # Precision / recall / F1.
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1

    def calculate_frame_level_f1(self, pred_scores: List,
                           gt_labels: List,
                           threshold: float = 0.5,
                           is_binary: bool = False) -> Dict[str, float]:
        """
        Frame-level F1 score.

        Args:
            pred_scores: Predicted scores.
            gt_labels: Ground-truth labels.
            threshold: Threshold for classifying a window as fake.
            is_binary: True if running in binary classification mode.
        """
        if len(pred_scores) != len(gt_labels):
            raise ValueError('Prediction and ground-truth window counts differ')

        # Convert to binary predictions and labels.
        y_pred = []
        y_true = []

        for pred, gt in zip(pred_scores, gt_labels):
            if is_binary:
                # Binary mode: use the predicted score directly.
                pred_fake_score = float(pred)
                true_fake = 0 if gt[0] > 0.5 else 1
            else:
                # Four-class mode: index 0 of `target` is the background class.
                pred_fake_score = float(pred)
                # gt is a 4-way one-hot vector: index 0 == real (background), others == fake (foreground).
                if isinstance(gt, (list, np.ndarray)) and len(gt) >= 4:
                    true_fake = 0 if gt[0] > 0.5 else 1  # idx0=1 -> background(0); else -> foreground(1)
                else:
                    # Legacy format.
                    true_fake = 1 if gt > 0.5 else 0

            pred_fake = 1 if pred_fake_score > threshold else 0
            y_pred.append(pred_fake)
            y_true.append(true_fake)

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        # Compute TP, FP, FN.
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
        """
        Paper-style F1Det counts.

        The paper evaluates detection by splitting each video into fine-grained
        0.01s windows and comparing whether each window overlaps a predicted /
        ground-truth fake segment.
        """
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
                                    threshold: float = 0.5,
                                    is_binary: bool = False) -> List[Tuple[float, float]]:
        """
        Convert per-window predictions into temporal segments.
        """
        if not pred_scores:
            return []

        # Extract fake scores.
        fake_scores = []
        for score in pred_scores:
            if isinstance(score, np.ndarray) and score.ndim == 0:  # 0-d array
                fake_scores.append(score.item())
            elif isinstance(score, np.ndarray) and score.size == 1:  # 1-element array
                fake_scores.append(score.item())
            else:
                fake_scores.append(float(score))

        # Threshold to obtain a binary prediction per window.
        binary_preds = [1 if score > threshold else 0 for score in fake_scores]

        segments = []
        current_start = None

        # Find consecutive runs of fake windows.
        for i, pred in enumerate(binary_preds):
            window_start = i * self.window_duration
            window_end = min((i + 1) * self.window_duration, video_duration)

            if pred == 1 and current_start is None:  # segment start
                current_start = window_start
            elif pred == 0 and current_start is not None:  # segment end
                segments.append((current_start, window_start))
                current_start = None

        # Handle a segment that extends to the end.
        if current_start is not None:
            segments.append((current_start, video_duration))

        return segments

    def postprocess_video_segments(self, pred_segments: List[Tuple[float, float]],
                                 video_name: str,
                                 video_type: str) -> List[Tuple[float, float]]:
        """
        Optional post-processing for the predicted segments.

        Useful operations include:
        - filtering out short segments,
        - merging adjacent segments,
        - simple smoothing.
        """
        processed_segments = pred_segments.copy()

        # Plug in custom post-processing here (e.g. drop short segments or merge adjacent ones).

        return processed_segments

    def evaluate_video_segmentation(self,
                                pred_scores: List,
                                gt_labels: List,
                                video_duration: float,
                                gt_fake_segments: List[Tuple[float, float]],
                                video_type: str,
                                video_name: str = "",
                                is_binary: bool = False) -> Dict[str, Any]:
        """
        Evaluate temporal segmentation for a single video.
        """
        results = {}

        # 1. Frame-level F1 (threshold 0.5) - computed only for fake videos.
        if video_type == 'fake':
            frame_metrics = self.calculate_frame_level_f1(pred_scores, gt_labels, threshold=0.5, is_binary=is_binary)
        else:
            # Real videos: skip frame-level metrics.
            frame_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 0}
        results.update(frame_metrics)

        # 2. Convert predictions into segments.
        pred_segments = self.convert_predictions_to_segments(pred_scores, video_duration, threshold=0.5, is_binary=is_binary)

        # 3. Optional post-processing (fake videos only).
        if video_type == 'fake':
            pred_segments = self.postprocess_video_segments(pred_segments, video_name, video_type)

        results['pred_segments'] = pred_segments

        # Paper-style F1Det counts over 0.01s windows. These counts are later
        # summed globally before computing precision / recall / F1.
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

        # 4. Localization F1 at multiple IoU thresholds (fake videos only).
        iou_thresholds = [0.1, 0.3, 0.5, 0.7]
        if video_type == 'fake' and gt_fake_segments:
            loc_f1_scores = {}

            for threshold in iou_thresholds:
                f1_score = self.calculate_loc_f1_for_iou_threshold(pred_segments, gt_fake_segments, threshold)
                loc_f1_scores[f'loc_f1_{threshold}'] = f1_score

            # Average localization F1.
            avg_loc_f1 = np.mean(list(loc_f1_scores.values()))
            loc_f1_scores['avg_loc_f1'] = avg_loc_f1
            results.update(loc_f1_scores)
        else:
            # For real videos the localization metrics are zero.
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

        # 5. Video-level classification: any fake-window prediction marks the video as fake.
        fake_scores = []
        for score in pred_scores:
            if isinstance(score, np.ndarray) and score.ndim == 0:  # 0-d array
                fake_scores.append(score.item())
            elif isinstance(score, np.ndarray) and score.size == 1:  # 1-element array
                fake_scores.append(score.item())
            else:
                fake_scores.append(float(score))

        has_fake_prediction = any(score > 0.5 for score in fake_scores)
        video_pred_type = 'fake' if has_fake_prediction else 'real'
        video_true_type = video_type

        results['video_pred_type'] = video_pred_type
        results['video_true_type'] = video_true_type
        results['video_correct'] = True if video_pred_type == video_true_type else False

        ''' fy '''
        # parts = results['video_name'].split('_')
        # results['video_path'] = f"videos/{parts[0]}/{parts[1]}/videos/stitched/test/{'_'.join(parts[4:])}.mp4"
        # results['video_name'] = f"{'_'.join(parts[4:])}"
        results['fake_segments'] = [[i[0], i[1]] for i in results['pred_segments']]
        results['has_fake'] = results['fake_segments'] != []
        results['fake_window_count'] = len(results['fake_segments'])

        return results

    def evaluate_all_videos(self, video_data_list: List[Dict], is_binary: bool = False) -> Dict[str, Any]:
        """
        Evaluate every video.

        Args:
            video_data_list: One dict per video.
            is_binary: True if running in binary mode.
        """
        all_results = {}
        video_level_results = []

        for i, video_data in enumerate(video_data_list):
            video_name = video_data.get('video_name', f'video_{i}')

            # Evaluate one video.
            results = self.evaluate_video_segmentation(
                pred_scores=video_data['pred_scores'],
                gt_labels=video_data['gt_labels'],
                video_duration=video_data['video_duration'],
                gt_fake_segments=video_data['gt_fake_segments'],
                video_type=video_data['video_type'],
                video_name=video_name,
                is_binary=is_binary
            )

            all_results[video_name] = results
            video_level_results.append(results)

        # Compute the overall metrics.
        return self._calculate_overall_metrics(all_results, video_level_results, is_binary)

    def _calculate_overall_metrics(self, all_results: Dict[str, Any], video_level_results: List[Dict], is_binary: bool = False) -> Dict[str, Any]:
        """Aggregate per-video results into overall metrics."""
        # Video-level classification accuracy.
        video_accuracy = np.mean([r['video_correct'] for r in video_level_results])

        # Frame-level and localization metrics for fake videos.
        fake_video_results = [r for r in video_level_results if r['video_true_type'] == 'fake']
        real_video_results = [r for r in video_level_results if r['video_true_type'] == 'real']

        if fake_video_results:
            # Frame-level metrics (fake videos only).
            frame_f1_scores = [r['f1'] for r in fake_video_results]
            frame_precision_scores = [r['precision'] for r in fake_video_results]
            frame_recall_scores = [r['recall'] for r in fake_video_results]

            # Localization metrics (fake videos only).
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
    """Pretty-print the evaluation metrics dict."""
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

def build_model(model_name, neuron_indices_path=None, xclip_model_path=None, xclip_checkpoint_path=None):
    if model_name == 'XCLIP_DeMamba_4':
        return models.XCLIP_DeMamba(class_num=4, xclip_model_path=xclip_model_path)
    raise ValueError(f'Unsupported MSLoc-PG model: {model_name}')

def accumulate_batch_gradients(model, inputs, labels, loss_ce, micro_batch_size, is_binary=False):
    """Accumulate the full-batch mean loss; the caller performs one optimizer step.

    Weight the final partial chunk by its actual size. Stochastic layers can
    differ from a single forward.
    """
    if micro_batch_size <= 0 or inputs.shape[0] == 0:
        raise ValueError('Accumulation requires positive batch sizes')
    device = next(model.parameters()).device
    batch_size = inputs.shape[0]
    batch_loss = 0.0
    for start in range(0, batch_size, micro_batch_size):
        stop = min(start + micro_batch_size, batch_size)
        logits = model(inputs[start:stop].to(device))
        targets = labels[start:stop].contiguous().to(device)
        if not is_binary:
            targets = targets.long()[:, 0]
        weighted_loss = loss_ce(logits, targets) * ((stop - start) / batch_size)
        weighted_loss.backward()
        batch_loss += weighted_loss.item()
        del logits, targets, weighted_loss
    return batch_loss


def evaluation_forward(model, inputs, micro_batch_size=0):
    """Keep window order and logical batch boundaries while limiting forward size."""
    if micro_batch_size < 0:
        raise ValueError('Evaluation micro-batch size must be non-negative')
    if not micro_batch_size:
        return model(inputs)
    if model.training or torch.is_grad_enabled():
        raise RuntimeError('Chunked evaluation requires eval mode and disabled gradients')
    return torch.cat([model(chunk) for chunk in inputs.split(micro_batch_size)], dim=0)


def eval_model(cfg, model, val_loader, loss_ce, val_batch_size, test_fake_segments):
    model.eval()
    outpred_list = []
    gt_label_list = []
    video_list = []
    valLoss = 0
    lossTrainNorm = 0
    print("******** Start Testing. ********")

    video_seg_pred, video_seg_gt = {}, {}
    # Foreground probabilities are retained only as per-video summary inputs for
    # proposal-quality analysis.  They do not change the existing proposal rule.
    video_foreground_scores = {}
    video_loc_gt = {}

    # Detect binary mode.
    is_binary = cfg.get('mode', '') == 'binary'

    cache = Path(cfg['_evaluation_cache']) if cfg.get('_evaluation_cache') else None
    next_batch = 0
    if cache and cache.is_file():
        saved = torch.load(cache, map_location='cpu', weights_only=False)
        next_batch = saved['next_batch']
        outpred_list, gt_label_list, video_list = saved['outpred_list'], saved['gt_label_list'], saved['video_list']
        video_seg_pred, video_seg_gt = saved['video_seg_pred'], saved['video_seg_gt']
        video_foreground_scores, video_loc_gt = saved['video_foreground_scores'], saved['video_loc_gt']
        valLoss, lossTrainNorm = saved['valLoss'], saved['lossTrainNorm']
    with torch.no_grad():
        for i, (window_idx, input, target, binary_label, video_id) in enumerate(tqdm(val_loader, desc="Validation", total=len(val_loader))):
            if i < next_batch:
                continue
            if i == 0:
                ss_time = time.time()

            input = input[:,0]
            varInput = torch.autograd.Variable(input.float().cuda())
            varTarget = torch.autograd.Variable(target.contiguous().cuda())
            var_Binary_Target = torch.autograd.Variable(binary_label.contiguous().cuda())

            logit = evaluation_forward(model, varInput, cfg.get('eval_micro_batch_size', 0))
            if is_binary:
                lossvalue = loss_ce(logit, var_Binary_Target)
            else:
                lossvalue = loss_ce(logit, var_Binary_Target.long()[:, 0])

            valLoss += lossvalue.item()
            lossTrainNorm += 1

            # Process predictions according to the mode.
            if is_binary:
                # Binary: sigmoid of the first logit.
                outpred_list.append(logit[:,0].sigmoid().cpu().detach().numpy().copy())
            else:
                # Four-class: take the foreground probabilities.
                pred_probs = torch.softmax(logit, dim=1)
                # Sum over foreground classes (everything except class 0) as the fake score.
                fake_scores = pred_probs[:, 1:].sum(dim=1)
                outpred_list.append(fake_scores.cpu().detach().numpy().copy())

            # Retain independent values, never views into pinned DataLoader storage.
            gt_label_list.append(target.numpy().copy())
            video_list.append(list(video_id))

            ''' Per-video predictions. '''
            for j in range(len(video_id)):
                video_id_j = video_id[j]
                if video_id_j not in video_seg_pred:
                    video_seg_pred[video_id_j] = []
                    video_seg_gt[video_id_j] = []
                    video_foreground_scores[video_id_j] = []
                    video_loc_gt[video_id_j] = test_fake_segments[video_id_j]

                # Per-window prediction handled by mode.
                if is_binary:
                    foreground_score = logit[j,0].sigmoid().cpu().detach().numpy()
                    pred_score = foreground_score
                else:
                    pred_probs = torch.softmax(logit[j], dim=0)
                    pred_score = (pred_probs[1:].max() > pred_probs[0]).float().cpu().detach().numpy()
                    # Use total foreground probability as a confidence summary;
                    # the binary proposal decision above remains max(foreground) > real.
                    foreground_score = pred_probs[1:].sum().cpu().detach().numpy()


                index = int(window_idx[j])
                video_seg_pred[video_id_j].append((index, float(pred_score)))
                video_seg_gt[video_id_j].append((index, target[j].numpy().copy()))
                video_foreground_scores[video_id_j].append((index, float(foreground_score)))

            if cache and ((i + 1) % 100 == 0 or i + 1 == len(val_loader)):
                atomic_save(dict(next_batch=i + 1, outpred_list=outpred_list, gt_label_list=gt_label_list,
                                 video_list=video_list, video_seg_pred=video_seg_pred, video_seg_gt=video_seg_gt,
                                 video_foreground_scores=video_foreground_scores, video_loc_gt=video_loc_gt,
                                 valLoss=valLoss, lossTrainNorm=lossTrainNorm), cache)

    if set(video_seg_pred) != set(test_fake_segments):
        raise ValueError('Evaluation predictions do not cover every input video')
    for video_id, windows in video_seg_pred.items():
        expected = math.ceil(test_fake_segments[video_id]['duration'] / cfg['window_length'])
        if sorted(int(index) for index, _ in windows) != list(range(expected)):
            raise ValueError(f'Missing or duplicate windows: {video_id}')
    # Build per-video evaluation data.
    loc_data = []
    for video_id in video_seg_pred:
        video_seg_pred[video_id] = sorted(video_seg_pred[video_id], key=lambda x: x[0])
        video_seg_gt[video_id] = sorted(video_seg_gt[video_id], key=lambda x: x[0])
        assert len(video_seg_pred[video_id]) == len(video_seg_gt[video_id])

        # Extract predicted scores and ground-truth labels.
        if is_binary:
            pred_scores = [item[1].item() if hasattr(item[1], 'item') else float(item[1]) for item in video_seg_pred[video_id]]
            gt_labels = [item[1] for item in video_seg_gt[video_id]]
        else:
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

    # Evaluate over all videos.
    all_videos_results = evaluator.evaluate_all_videos(loc_data, is_binary=is_binary)

    # Summarize confidence for the already-generated binary proposals.  This is
    # deliberately post-hoc: it must not alter the detector's segmentation.
    for video_id, window_scores in video_foreground_scores.items():
        window_scores = sorted(window_scores, key=lambda item: item[0])
        foreground_scores = [score for _, score in window_scores]
        video_result = all_videos_results['per_video_results'][video_id]
        window_predictions = sorted(video_seg_pred[video_id], key=lambda item: item[0])
        proposal_scores = [
            foreground_scores[index]
            for index, (_, prediction) in enumerate(window_predictions)
            if float(prediction) > 0.5 and index < len(foreground_scores)
        ]

        video_result['proposal_confidence_mean'] = (
            float(np.mean(proposal_scores)) if proposal_scores else 0.0
        )
        video_result['proposal_confidence_max'] = (
            float(np.max(proposal_scores)) if proposal_scores else 0.0
        )
        # For a fake decision, its strongest positive window supports the
        # decision.  For a real decision, use the mean real probability.
        if video_result.get('video_pred_type') == 'fake':
            video_result['video_decision_confidence'] = video_result['proposal_confidence_max']
        else:
            video_result['video_decision_confidence'] = (
                float(np.mean([1.0 - score for score in foreground_scores]))
                if foreground_scores else 0.0
            )

    print_evaluation_metrics(all_videos_results['overall_metrics'])

    valLoss = valLoss / lossTrainNorm

    outpred = np.concatenate(outpred_list, 0)
    gt_label = np.concatenate(gt_label_list, 0)
    video_list = np.concatenate(video_list, 0)

    # Compute accuracy according to the mode.
    if is_binary:
        # Binary: 0.5 threshold.
        pred_labels = [1 if item > 0.5 else 0 for item in outpred]
        true_labels = np.argmax(gt_label, axis=1) if gt_label.ndim > 1 else (gt_label > 0.5).astype(int)
    else:
        # Four-class: idx0=1 -> background; otherwise foreground.
        if gt_label.ndim > 1 and gt_label.shape[1] >= 4:
            # 4-way one-hot: idx0=1 -> background(0); else -> foreground(1).
            true_labels = np.where(gt_label[:, 0] > 0.5, 0, 1)
        else:
            # Legacy format.
            true_labels = (gt_label > 0.5).astype(int)

        pred_labels = [1 if item > 0.5 else 0 for item in outpred]
    pred_accuracy = accuracy_score(true_labels, pred_labels)

    return pred_accuracy, video_list, pred_labels, true_labels, outpred, all_videos_results

def train_batches(cfg, model, loss_ce, scheduler, optimizer, epochID, train_loader):
    model.train()
    trainLoss = 0
    lossTrainNorm = 0
    scheduler.step()

    pbar = tqdm(total=min(len(train_loader), cfg['bath_per_epoch'] + 1), desc=f'Train {epochID + 1}')
    for batchID, (index, input, target, binary_label) in enumerate(train_loader):
        if batchID > cfg['bath_per_epoch']:
            break
        if batchID == 0:
            ss_time = time.time()
        input = input[:,0].float()
        optimizer.zero_grad()
        if cfg.get('micro_batch_size', 0):
            batch_loss = accumulate_batch_gradients(
                model, input, binary_label, loss_ce, cfg['micro_batch_size'],
                is_binary=cfg['mode'] == 'binary',
            )
        else:
            varInput = torch.autograd.Variable(input).cuda()
            var_Binary_Target = torch.autograd.Variable(binary_label.contiguous().cuda())
            logit = model(varInput)
            if cfg['mode'] == 'binary':
                lossvalue = loss_ce(logit, var_Binary_Target)
            else:
                lossvalue = loss_ce(logit, var_Binary_Target.long()[:, 0])
            lossvalue.backward()
            batch_loss = lossvalue.item()
            del lossvalue, logit, varInput, var_Binary_Target
        optimizer.step()

        trainLoss += batch_loss
        lossTrainNorm += 1
        pbar.set_postfix(loss=trainLoss / lossTrainNorm)
        pbar.update(1)

    pbar.close()
    if lossTrainNorm == 0:
        raise RuntimeError("Training loader produced no batches")
    trainLoss = trainLoss / lossTrainNorm
    return trainLoss, time.time() - ss_time


def train_one_epoch(cfg, model, loss_ce, scheduler, optimizer, epochID, max_epoch, max_acc,
                    train_loader, val_loader, snapshot_path, test_fake_segments,
                    pending_training=None, on_training_complete=None):
    ss_time = time.time()
    if pending_training is None:
        trainLoss, train_seconds = train_batches(
            cfg, model, loss_ce, scheduler, optimizer, epochID, train_loader,
        )
        # Persist updated weights before potentially long-running validation.
        if on_training_complete is not None:
            on_training_complete(dict(train_loss=trainLoss, train_seconds=train_seconds))
    else:
        trainLoss = pending_training['train_loss']
        print(f'Resuming validation for epoch {epochID + 1}; training updates are already saved.')

    if (epochID+1) % 1 == 0:
        pred_accuracy, video_id, pred_labels, true_labels, outpred, all_videos_results = eval_model(cfg, model, val_loader, loss_ce, cfg['val_batch_size'], test_fake_segments)

        torch.save(
            {"epoch": epochID + 1, "model_state_dict": model.state_dict()},
            snapshot_path + f"/{str(epochID + 1)}"+ ".pth",
            )

        seg_f1 = all_videos_results['overall_metrics']['frame_level']['f1']
        det_acc = all_videos_results['overall_metrics']['video_level']['accuracy']
        loc_f1 = all_videos_results['overall_metrics']['localization_fake_videos']['avg_loc_f1']

        if loc_f1 > max_acc:
            max_epoch, max_acc = epochID, loc_f1
            torch.save(
            {"epoch": epochID + 1, "model_state_dict": model.state_dict()},
            snapshot_path + "/best_acc"+ ".pth",
            )

        df_result = pd.DataFrame({
            'data_path': video_id,
            'predicted_label': pred_labels,
            'actual_label': true_labels,
            'predicted_prob':outpred
        })

        temp_result_txt = snapshot_path+'/Epoch_'+str(epochID)+'_accuracy.txt'
        with open(temp_result_txt, 'w') as file:
            true_labels = df_result['actual_label']
            pred_probs = df_result['predicted_prob']
            # auc = roc_auc_score(true_labels, pred_probs)
            # ap = average_precision_score(true_labels, pred_probs)
            file.write(f"Det: {det_acc:.2%}\n")
            file.write(f"Seg: {seg_f1:.2%}\n")
            file.write(f"Loc: {loc_f1:.2%}\n")
            file.write(f"Temporal-segmentation metrics: {all_videos_results}\n")

        print("*****Average Training loss",str(trainLoss),"*****Epoch", str(epochID), "*****Acc ", str(loc_f1), '*****',
            '\n', "*****Max acc epoch", str(max_epoch), "*****Acc ", str(max_acc), '*****\n')

    end_time = time.time()

    return max_epoch, max_acc, end_time - ss_time, trainLoss, all_videos_results['overall_metrics']
