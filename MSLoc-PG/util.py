import models
import time
from itertools import islice
from pathlib import Path
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
            warnings.warn(f"Predicted count ({len(pred_scores)}) does not match GT count ({len(gt_labels)})")
            min_len = min(len(pred_scores), len(gt_labels))
            pred_scores = pred_scores[:min_len]
            gt_labels = gt_labels[:min_len]
        
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
        
        # 4. Localization F1 at multiple IoU thresholds (fake videos only).
        if video_type == 'fake' and gt_fake_segments:
            iou_thresholds = [0.1, 0.3, 0.5, 0.7]
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
        
        overall_metrics = {
            'frame_level': {
                'f1': np.mean(frame_f1_scores).item(),
                'precision': np.mean(frame_precision_scores).item(),
                'recall': np.mean(frame_recall_scores).item()
            },
            'video_level': {
                'accuracy': video_accuracy.item()
            },
            'localization_fake_videos': {
                'avg_loc_f1': np.mean(avg_loc_f1_scores).item(),
                'loc_f1_0.1': np.mean(loc_f1_01_scores).item(),
                'loc_f1_0.3': np.mean(loc_f1_03_scores).item(),
                'loc_f1_0.5': np.mean(loc_f1_05_scores).item(),
                'loc_f1_0.7': np.mean(loc_f1_07_scores).item()
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
    
    frame_metrics = metrics_dict.get('frame_level', {})
    print("Frame-level metrics (fake videos only):")
    print(f"  F1: {frame_metrics.get('f1', 0):.4f}  Precision: {frame_metrics.get('precision', 0):.4f}  Recall: {frame_metrics.get('recall', 0):.4f}")
    
    video_metrics = metrics_dict.get('video_level', {})
    print("\nVideo-level metrics:")
    print(f"  Classification accuracy: {video_metrics.get('accuracy', 0):.4f}")
    
    loc_metrics = metrics_dict.get('localization_fake_videos', {})
    print("\nLocalization metrics (fake videos only):")
    print(f"  Avg Loc F1: {loc_metrics.get('avg_loc_f1', 0):.4f}  F1@0.1: {loc_metrics.get('loc_f1_0.1', 0):.4f}  F1@0.3: {loc_metrics.get('loc_f1_0.3', 0):.4f}  F1@0.5: {loc_metrics.get('loc_f1_0.5', 0):.4f}  F1@0.7: {loc_metrics.get('loc_f1_0.7', 0):.4f}")

    counts = metrics_dict.get('counts', {})
    print("\nDataset stats:")
    print(f"  Total: {counts.get('total_videos', 0)}  Fake: {counts.get('fake_videos', 0)}  Real: {counts.get('real_videos', 0)}")
    print("=" * 50)

def build_model(model_name):
    if model_name == 'XCLIP_DeMamba':
        return models.XCLIP_DeMamba()
    if model_name == 'XCLIP_DeMamba_4':
        return models.XCLIP_DeMamba(class_num=4)
    if model_name == 'CLIP_DeMamba':
        return models.CLIP_DeMamba()
    raise ValueError(f"Unsupported model_name: {model_name}")

def eval_model(cfg, model, val_loader, loss_ce, val_batch_size, test_fake_segments):
    model.eval()
    outpred_list = []
    gt_label_list = []
    video_list = []
    valLoss = 0
    lossTrainNorm = 0
    print("******** Start Testing. ********")

    video_seg_pred, video_seg_gt = {}, {}
    video_loc_gt = {}
    
    # Detect binary mode.
    is_binary = cfg.get('mode', '') == 'binary'

    memory_status = Path('/proc/self/status')
    with torch.no_grad(), tqdm(val_loader, desc="Validation", total=len(val_loader)) as progress:
        for i, (window_idx, input, target, binary_label, video_id) in enumerate(progress):
            if i == 0:
                ss_time = time.time()

            input = input[:,0]
            varInput = torch.autograd.Variable(input.float().cuda())
            var_Binary_Target = torch.autograd.Variable(binary_label.contiguous().cuda())

            logit = model(varInput)
            if is_binary:
                lossvalue = loss_ce(logit, var_Binary_Target)
            else:
                lossvalue = loss_ce(logit, var_Binary_Target.long()[:, 0])

            valLoss += lossvalue.item()
            lossTrainNorm += 1
            
            # Process predictions according to the mode.
            if is_binary:
                # Binary: sigmoid of the first logit.
                fake_scores = logit[:, 0].sigmoid()
                segment_scores = fake_scores
            else:
                # Four-class: take the foreground probabilities.
                pred_probs = torch.softmax(logit, dim=1)
                # Sum over foreground classes (everything except class 0) as the fake score.
                fake_scores = pred_probs[:, 1:].sum(dim=1)
                segment_scores = (pred_probs[:, 1:].max(dim=1).values > pred_probs[:, 0]).float()
            
            # Keep independent CPU values, not views into DataLoader or tensor storage.
            outpred_list.append(fake_scores.cpu().numpy().copy())
            gt_label_list.append(target.numpy().copy())
            video_list.append(video_id)
            window_indices = window_idx.tolist()
            segment_scores = segment_scores.cpu().tolist()
            target_values = target.tolist()

            ''' Per-video predictions. '''
            for j in range(len(video_id)):
                video_id_j = video_id[j]
                if video_id_j not in video_seg_pred:
                    video_seg_pred[video_id_j] = []
                    video_seg_gt[video_id_j] = []
                    video_loc_gt[video_id_j] = test_fake_segments[video_id_j]
                
                video_seg_pred[video_id_j].append((window_indices[j], segment_scores[j]))
                video_seg_gt[video_id_j].append((window_indices[j], target_values[j]))

            del input, target, binary_label, window_idx, varInput, var_Binary_Target, logit, lossvalue, fake_scores
            if not is_binary:
                del pred_probs
            if (i == 0 or (i + 1) % 50 == 0 or i + 1 == len(val_loader)) and memory_status.is_file():
                for line in memory_status.read_text().splitlines():
                    if line.startswith('VmRSS:'):
                        progress.set_postfix(ram_gib=f'{int(line.split()[1]) / 1024 ** 2:.2f}')
                        break

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

def train_one_epoch(cfg, model, loss_ce, scheduler, optimizer, epochID, max_epoch, max_acc, train_loader, val_loader, snapshot_path, test_fake_segments, micro_batch_size=None):
    model.train()
    trainLoss = 0
    lossTrainNorm = 0

    ss_time = time.time()
    num_batches = min(len(train_loader), cfg['bath_per_epoch'])
    if num_batches <= 0 or (micro_batch_size is not None and micro_batch_size <= 0):
        raise ValueError('Training requires positive batch counts and micro-batch size')
    with tqdm(islice(train_loader, num_batches), total=num_batches,
              desc=f'Train {epochID + 1}/{cfg["max_epoch"]}') as pbar:
        for index, input, target, binary_label in pbar:
            input = input[:, 0].float()
            batch_size = input.shape[0]
            chunk_size = micro_batch_size or batch_size
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0

            # Weight each micro-batch by its size to preserve the full-batch mean loss.
            for start in range(0, batch_size, chunk_size):
                stop = min(start + chunk_size, batch_size)
                varInput = input[start:stop].cuda()
                var_Binary_Target = binary_label[start:stop].contiguous().cuda()
                logit = model(varInput)
                if cfg['mode'] == 'binary':
                    lossvalue = loss_ce(logit, var_Binary_Target)
                else:
                    lossvalue = loss_ce(logit, var_Binary_Target.long()[:, 0])
                lossvalue = lossvalue * ((stop - start) / batch_size)
                lossvalue.backward()
                batch_loss += lossvalue.item()
                del lossvalue, logit, varInput, var_Binary_Target
            optimizer.step()

            trainLoss += batch_loss
            lossTrainNorm += 1
            pbar.set_postfix(loss=trainLoss / lossTrainNorm, lr=optimizer.param_groups[0]['lr'])
    scheduler.step()

    trainLoss = trainLoss / lossTrainNorm
    
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

    return max_epoch, max_acc, end_time - ss_time

