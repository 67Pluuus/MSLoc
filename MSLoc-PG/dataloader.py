import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import albumentations
import random
import os
import numpy as np
import cv2
import math
import warnings
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm

def get_video_fake_segments(json_data: List[Dict]) -> Dict[str, Dict]:
    """
    Build a per-video summary containing the fake segments and the video type.

    Args:
        json_data: Raw JSON data with video info and annotations.

    Returns:
        A dict mapping video name to {fake_segments, type, ...}.
    """
    video_info_dict = {}
    
    for video_item in json_data:
        video_path = video_item["video_path"]
        video_type = video_item["type"]
        duration = video_item["duration"]
        annotations = video_item.get("annotations", [])
        
        # Strip the .mp4 extension and replace path separators with underscores.
        video_name = Path(video_path).stem
        formatted_video_name = video_path[:-4].replace('/', '_')
        
        # Collect fake-segment intervals.
        fake_segments = []
        tool = 'None'
        for ann in annotations:
            if "segment" in ann and len(ann["segment"]) == 2 and video_type == "fake":
                start_time, end_time = ann["segment"]
                fake_segments.append((start_time, end_time))
                tool = ann["model"] if (tool in ["None", "Wan2.1"]) else tool
        
        # If no explicit fake segments are listed but the type is fake, use the whole duration.
        if not fake_segments and video_type == "fake":
            fake_segments = [(-1, duration)]

        video_info_dict[formatted_video_name] = {
            "type": video_type,
            "duration": duration,
            "fake_segments": fake_segments,
            "tool": tool,
            "real_fake_segments": [x['segment'] for x in annotations]
        }
    
    return video_info_dict

def parse_json_to_windows(json_data: List[Dict], dataset_base_path: str, 
                          window_length: float = 1.0, frames_per_window: int = 4,
                          mode: str = "binary", show_progress: bool = False) -> List[Dict]:
    """
    Parse JSON data and produce sliding-window samples.

    Args:
        json_data: Raw JSON data.
        dataset_base_path: Dataset root path.
        window_length: Window length in seconds.
        frames_per_window: Number of frames sampled per window.
        mode: Classification mode, "binary" or "four_class".
        show_progress: Display video preparation progress.

    Returns:
        A list of window dicts containing frame paths and labels.
    """
    fps = 8  # fixed frame rate
    windows = []
    
    for video_item in tqdm(json_data, desc='Preparing video windows', disable=not show_progress):
        video_path = video_item["video_path"]
        video_type = video_item["type"]
        duration = video_item["duration"]
        annotations = video_item.get("annotations", [])
        
        # Build the frame directory path.
        video_name = Path(video_path).stem  # strip .mp4 suffix
        frame_dir = Path(dataset_base_path) / video_path[:-4]
        new_video_name = video_path[:-4].replace('/', '_')
        
        if not frame_dir.exists():
            warnings.warn(f"Frame directory not found: {frame_dir}")
            continue
        
        # Collect every frame file.
        frame_files = sorted(frame_dir.glob("frame_*.jpg"), key=lambda x: int(x.stem.split('_')[1]))
        if not frame_files:
            warnings.warn(f"No frame files found in: {frame_dir}")
            continue
        
        # Total frames and per-frame duration.
        total_frames = len(frame_files)
        frame_duration = 1.0 / fps  # seconds per frame
        
        # Number of sliding windows.
        num_windows = int(duration // window_length)
        if duration % window_length > 0:
            num_windows += 1
        
        # Time intervals of fake segments.
        fake_segments = []
        for ann in annotations:
            if "segment" in ann and len(ann["segment"]) == 2 and video_type == "fake":
                start_time, end_time = ann["segment"]
                fake_segments.append((start_time, end_time))
        if fake_segments == []:
            fake_segments = [(-1, duration)]
        
        # Generate windows.
        for window_idx in range(num_windows):
            window_start_time = window_idx * window_length
            window_end_time = min((window_idx + 1) * window_length, duration)
            
            # Frame range of the current window.
            start_frame_idx = int(window_start_time * fps)
            end_frame_idx = min(int(window_end_time * fps), total_frames - 1)
            
            if start_frame_idx >= total_frames:
                continue
                
            # Sample frames uniformly within the window.
            frame_indices = []
            if end_frame_idx - start_frame_idx + 1 >= frames_per_window:
                # Enough frames -> uniform sampling.
                step = max(1, (end_frame_idx - start_frame_idx) // frames_per_window)
                frame_indices = list(range(start_frame_idx, end_frame_idx + 1, step))[:frames_per_window]
            else:
                # Not enough frames -> repeat the last one.
                frame_indices = list(range(start_frame_idx, end_frame_idx + 1))
                while len(frame_indices) < frames_per_window:
                    frame_indices.append(frame_indices[-1])
                frame_indices = frame_indices[:frames_per_window]
            
            # Resolve frame paths.
            frame_paths = []
            for frame_idx in frame_indices:
                if frame_idx < len(frame_files):
                    frame_paths.append(str(frame_files[frame_idx]))
                else:
                    # Out of range -> use the last frame.
                    frame_paths.append(str(frame_files[-1]))
            
            # Window label.
            if mode == "binary":
                label = determine_binary_label(window_start_time, window_end_time, 
                                             fake_segments, video_type)
            else:  # four_class
                label = determine_four_class_label(window_start_time, window_end_time, 
                                                  fake_segments, video_type)
            
            window_info = {
                "window_id": f"{new_video_name}_window_{window_idx}",
                "video_name": new_video_name,
                "window_idx": window_idx,
                "start_time": window_start_time,
                "end_time": window_end_time,
                "frame_paths": frame_paths,
                "label": label,
                "original_type": video_type
            }
            windows.append(window_info)
    return windows


def determine_binary_label(window_start: float, window_end: float, 
                          fake_segments: List[Tuple[float, float]], 
                          video_type: str) -> int:
    """
    Compute the binary label for a window.
    """
    if video_type == "real":
        return 0  # real
    
    # Mark as fake if the window overlaps with any fake segment.
    for seg_start, seg_end in fake_segments:
        if not (window_end <= seg_start or window_start >= seg_end):
            return 1  # fake
    
    return 0  # real


def determine_four_class_label(window_start: float, window_end: float, 
                              fake_segments: List[Tuple[float, float]], 
                              video_type: str) -> int:
    """
    Compute the four-class label for a window.
    0: real, 1: fake, 2: real-to-fake, 3: fake-to-real
    """
    if video_type == "real":
        return 0  # real
    
    # Compute how much of the window overlaps with fake segments.
    window_fake_ratio = 0.0
    fake_duration_in_window = 0.0
    window_duration = window_end - window_start
    
    for seg_start, seg_end in fake_segments:
        overlap_start = max(window_start, seg_start)
        overlap_end = min(window_end, seg_end)
        if overlap_start < overlap_end:
            fake_duration_in_window += (overlap_end - overlap_start)
    
    window_fake_ratio = fake_duration_in_window / window_duration
    
    # Decide which label to use.
    if window_fake_ratio == 0.0:
        return 0  # real
    elif window_fake_ratio == 1.0:
        return 1  # fake
    elif window_fake_ratio > 0.5:
        # Mostly fake -> figure out the transition type.
        if window_start < fake_segments[0][0]:  # assume a single transition point
            return 3  # fake-to-real
        else:
            return 1  # mostly fake -> treat as plain fake
    else:
        # Mostly real -> figure out the transition type.
        if window_end > fake_segments[0][1]:  # assume a single transition point
            return 2  # real-to-fake
        else:
            return 0  # mostly real -> treat as plain real


def crop_center_by_percentage(image, percentage):
    height, width = image.shape[:2]

    if width > height:
        left_pixels = int(width * percentage)
        right_pixels = int(width * percentage)
        start_x = left_pixels
        end_x = width - right_pixels
        cropped_image = image[:, start_x:end_x]
    else:
        up_pixels = int(height * percentage)
        down_pixels = int(height * percentage)
        start_y = up_pixels
        end_y = height - down_pixels
        cropped_image = image[start_y:end_y, :]

    return cropped_image


class VideoWindowDatasetTrain(Dataset):
    def __init__(self, windows: List[Dict], mode: str = "binary", transform_config: Dict = None):
        """
        Train-time video-window dataset (with balanced sampling).

        Args:
            windows: List of window dicts.
            mode: Classification mode, "binary" or "four_class".
            transform_config: Augmentation configuration.
        """
        self.windows = windows
        self.mode = mode
        self.transform_config = transform_config or {}
        
        # Number of classes is determined by the mode.
        self.num_classes = 2 if mode == "binary" else 4
        
        # Balanced sampling.
        self._balance_samples()
    
    def _balance_samples(self):
        """Balance positive and negative samples."""
        if self.mode == "binary":
            # Binary balancing.
            positive_indices = [i for i, w in enumerate(self.windows) if w['label'] == 1]
            negative_indices = [i for i, w in enumerate(self.windows) if w['label'] == 0]
            
            min_samples = min(len(positive_indices), len(negative_indices))
            self.balanced_indices = []
            self.balanced_indices.extend(random.sample(positive_indices, min_samples))
            self.balanced_indices.extend(random.sample(negative_indices, min_samples))
        else:
            # Four-class balancing.
            class_indices = {0: [], 1: [], 2: [], 3: []}
            for i, w in enumerate(self.windows):
                class_indices[w['label']].append(i)
            
            min_samples = min(len(indices) for indices in class_indices.values())
            self.balanced_indices = []
            self.balanced_indices.extend(random.sample(class_indices[0], min(len(class_indices[0]), len(class_indices[1])+len(class_indices[2])+len(class_indices[3]))))
            self.balanced_indices.extend(random.sample(class_indices[1], len(class_indices[1])))
            self.balanced_indices.extend(random.sample(class_indices[2], len(class_indices[2])))
            self.balanced_indices.extend(random.sample(class_indices[3], len(class_indices[3])))
        
        random.shuffle(self.balanced_indices)
    
    def __len__(self):
        return len(self.balanced_indices)
    
    def __getitem__(self, idx):
        real_idx = self.balanced_indices[idx]
        window_info = self.windows[real_idx]
        
        frame_paths = window_info['frame_paths']
        label = window_info['label']
        select_frame_nums = len(frame_paths)
        
        # Build augmentation pipeline.
        aug_list = [albumentations.Resize(224, 224)]
        
        # Train-time augmentations.
        if random.random() < 0.5:
            aug_list.append(albumentations.HorizontalFlip(p=1.0))
        if random.random() < 0.5:
            quality_score = random.randint(50, 100)
            aug_list.append(albumentations.ImageCompression(quality_lower=quality_score, quality_upper=quality_score))
        if random.random() < 0.3:
            aug_list.append(albumentations.GaussNoise(p=1.0))
        if random.random() < 0.3:
            aug_list.append(albumentations.GaussianBlur(blur_limit=(3, 5), p=1.0))
        if random.random() < 0.001:
            aug_list.append(albumentations.ToGray(p=1.0))
        
        aug_list.append(albumentations.Normalize(mean=(0.485, 0.456, 0.406), 
                                               std=(0.229, 0.224, 0.225), 
                                               max_pixel_value=255.0, p=1.0))
        trans = albumentations.Compose(aug_list)
        
        # Read and preprocess frames.
        frames = []
        valid_frames = 0
        
        for frame_path in frame_paths:
            try:
                if os.path.exists(frame_path):
                    image = cv2.imread(frame_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Apply optional center-crop.
                    if self.transform_config.get('crop_youku', False) and 'youku' in frame_path:
                        image = crop_center_by_percentage(image, 0.15)
                    
                    augmented = trans(image=image)
                    image = augmented["image"]
                    frames.append(image.transpose(2, 0, 1)[np.newaxis, :])
                    valid_frames += 1
                else:
                    # Missing frame -> zero padding.
                    frames.append(np.zeros((3, 224, 224))[np.newaxis, :])
            except Exception as e:
                warnings.warn(f"Error loading frame {frame_path}: {e}")
                frames.append(np.zeros((3, 224, 224))[np.newaxis, :])
        
        # Pad if fewer than the requested number of frames are available.
        if valid_frames < select_frame_nums:
            pad_num = select_frame_nums - valid_frames
            for i in range(pad_num):
                frames.append(np.zeros((3, 224, 224))[np.newaxis, :])
        
        # Build the label.
        label_onehot = [0] * self.num_classes
        label_onehot[label] = 1
        
        # Convert to tensors.
        frames = np.concatenate(frames, 0)
        frames = torch.tensor(frames[np.newaxis, :])
        label_onehot = torch.FloatTensor(label_onehot)
        binary_label = torch.FloatTensor([label])
        
        return real_idx, frames, label_onehot, binary_label


class VideoWindowDatasetTest(Dataset):
    def __init__(self, windows: List[Dict], mode: str = "binary", transform_config: Dict = None, task: str = "normal"):
        """
        Test-time video-window dataset (no balanced sampling; evaluate all samples).

        Args:
            windows: List of window dicts.
            mode: Classification mode, "binary" or "four_class".
            transform_config: Augmentation configuration.
            task: Task type, used for task-specific preprocessing.
        """
        self.windows = windows
        self.mode = mode
        self.transform_config = transform_config or {}
        self.task = task
        
        # Number of classes is determined by the mode.
        self.num_classes = 2 if mode == "binary" else 4
        
        # Use every sample at test time (no balancing).
        self.indices = list(range(len(windows)))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        window_info = self.windows[real_idx]
        
        frame_paths = window_info['frame_paths']
        label = window_info['label']
        video_name = window_info['video_name']
        window_idx = window_info['window_idx']
        select_frame_nums = len(frame_paths)
        
        # Build the augmentation pipeline (no random augmentations at test time).
        aug_list = [albumentations.Resize(224, 224)]
        
        aug_list.append(albumentations.Normalize(mean=(0.485, 0.456, 0.406), 
                                               std=(0.229, 0.224, 0.225), 
                                               max_pixel_value=255.0, p=1.0))
        trans = albumentations.Compose(aug_list)
        
        # Read and preprocess frames.
        frames = []
        valid_frames = 0
        
        for frame_path in frame_paths:
            try:
                if os.path.exists(frame_path):
                    image = cv2.imread(frame_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Apply optional center-crop.
                    if self.transform_config.get('crop_youku', False) and 'youku' in frame_path:
                        image = crop_center_by_percentage(image, 0.15)
                    
                    augmented = trans(image=image)
                    image = augmented["image"]
                    frames.append(image.transpose(2, 0, 1)[np.newaxis, :])
                    valid_frames += 1
                else:
                    # Missing frame -> zero padding.
                    frames.append(np.zeros((3, 224, 224))[np.newaxis, :])
            except Exception as e:
                warnings.warn(f"Error loading frame {frame_path}: {e}")
                frames.append(np.zeros((3, 224, 224))[np.newaxis, :])
        
        # Pad if fewer than the requested number of frames are available.
        if valid_frames < select_frame_nums:
            pad_num = select_frame_nums - valid_frames
            for i in range(pad_num):
                frames.append(np.zeros((3, 224, 224))[np.newaxis, :])
        
        # Build the label.
        label_onehot = [0] * self.num_classes
        label_onehot[label] = 1
        
        # Convert to tensors.
        frames = np.concatenate(frames, 0)
        frames = torch.tensor(frames[np.newaxis, :])
        label_onehot = torch.FloatTensor(label_onehot)
        binary_label = torch.FloatTensor([label])
        return window_idx, frames, label_onehot, binary_label, video_name


def generate_dataset_loader_from_json(cfg):
    """
    Build train / test data loaders from JSON annotation files.

    Args:
        cfg: Config dict with keys:
            - train_json_path:    Train annotation JSON.
            - test_json_path:     Test annotation JSON.
            - dataset_base_path:  Dataset root path.
            - window_length:      Window length in seconds.
            - frames_per_window:  Frames sampled per window.
            - mode:               Classification mode.
            - task:               Task type.
            - other training-related parameters.
    """
    # Load JSON files.
    with open(cfg['train_json_path'], 'r') as f:
        train_json = json.load(f)
    with open(cfg['test_json_path'], 'r') as f:
        test_json = json.load(f)

    # Keep complete videos so localization metrics never use truncated windows.
    val_videos = cfg.get('val_videos', 0)
    if val_videos < 0:
        raise ValueError('val_videos must be nonnegative')
    if val_videos > 0:
        video_paths = list(dict.fromkeys(item['video_path'] for item in test_json))
        selected_paths = set(video_paths[:val_videos])
        test_json = [item for item in test_json if item['video_path'] in selected_paths]
        print(f'Validation videos: {len(selected_paths)}/{len(video_paths)} (limit={val_videos})')
    
    # Convert to per-window samples.
    train_windows = parse_json_to_windows(
        train_json, 
        cfg['dataset_base_path'],
        cfg.get('window_length', 1.0),
        cfg.get('frames_per_window', 4),
        cfg.get('mode', 'binary'),
        show_progress=cfg.get('show_progress', False)
    )
    
    test_windows = parse_json_to_windows(
        test_json,
        cfg['dataset_base_path'],
        cfg.get('window_length', 1.0),
        cfg.get('frames_per_window', 4),
        cfg.get('mode', 'binary'),
        show_progress=cfg.get('show_progress', False)
    )
    test_fake_segments = get_video_fake_segments(test_json)
    
    # Build datasets.
    train_dataset = VideoWindowDatasetTrain(
        train_windows,
        mode=cfg.get('mode', 'binary'),
        transform_config=cfg.get('transform_config', {})
    )
    
    test_dataset = VideoWindowDatasetTest(
        test_windows,
        mode=cfg.get('mode', 'binary'),
        transform_config=cfg.get('transform_config', {}),
        task=cfg.get('task', 'normal')
    )
    
    # Build dataloaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['train_batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['val_batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        drop_last=False
    )


    
    print(f"******* Training Windows: {len(train_windows)} -> Balanced: {len(train_dataset)}")
    
    return train_loader, test_loader, test_fake_segments

