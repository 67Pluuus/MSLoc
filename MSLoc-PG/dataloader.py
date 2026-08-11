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


def is_fake_annotation(video_type: str, annotation: Dict[str, Any]) -> bool:
    """Accept TASLE's implicit labels without treating explicit real segments as fake."""
    segment = annotation.get("segment")
    if video_type != "fake" or not isinstance(segment, (list, tuple)) or len(segment) != 2:
        return False
    segment_label = annotation.get("segment_label")
    return segment_label is None or segment_label == "fake"


def collect_fake_segments(video_item: Dict[str, Any]) -> List[Tuple[float, float]]:
    video_type = video_item.get("type")
    annotations = video_item.get("annotations", [])
    fake_segments = [tuple(ann["segment"]) for ann in annotations if is_fake_annotation(video_type, ann)]

    if video_type == "fake" and not annotations:
        fake_segments = [(0, video_item["duration"])]
    return fake_segments

def get_video_fake_segments(json_data: List[Dict]) -> Dict[str, Dict]:
    video_info_dict = {}

    for video_item in json_data:
        video_path = video_item["video_path"]
        video_type = video_item["type"]
        duration = video_item["duration"]
        annotations = video_item.get("annotations", [])

        video_name = Path(video_path).stem
        formatted_video_name = video_path[:-4].replace('/', '_')

        fake_segments = collect_fake_segments(video_item)
        tool = "unknown"
        for ann in annotations:
            if not is_fake_annotation(video_type, ann):
                continue
            candidate = (
                ann.get("model")
                or video_item.get("tool_domain")
                or ann.get("tool_domain")
                or "unknown"
            )
            if tool in {"unknown", "Wan2.1"}:
                tool = candidate

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
                          mode: str = "binary") -> List[Dict]:
    fps = 8
    windows = []

    for video_item in json_data:
        video_path = video_item["video_path"]
        video_type = video_item["type"]
        duration = video_item["duration"]
        annotations = video_item.get("annotations", [])

        video_name = Path(video_path).stem
        frame_dir = Path(dataset_base_path) / video_path[:-4]
        new_video_name = video_path[:-4].replace('/', '_')

        if not frame_dir.exists():
            warnings.warn(f"Frame directory not found: {frame_dir}")
            continue

        frame_files = sorted(frame_dir.glob("frame_*.jpg"), key=lambda x: int(x.stem.split('_')[1]))
        if not frame_files:
            warnings.warn(f"No frame files found in: {frame_dir}")
            continue

        total_frames = len(frame_files)
        frame_duration = 1.0 / fps

        num_windows = int(duration // window_length)
        if duration % window_length > 0:
            num_windows += 1

        fake_segments = collect_fake_segments(video_item)

        for window_idx in range(num_windows):
            window_start_time = window_idx * window_length
            window_end_time = min((window_idx + 1) * window_length, duration)

            start_frame_idx = int(window_start_time * fps)
            end_frame_idx = min(int(window_end_time * fps), total_frames - 1)

            if start_frame_idx >= total_frames:
                continue

            frame_indices = []
            if end_frame_idx - start_frame_idx + 1 >= frames_per_window:
                step = max(1, (end_frame_idx - start_frame_idx) // frames_per_window)
                frame_indices = list(range(start_frame_idx, end_frame_idx + 1, step))[:frames_per_window]
            else:
                frame_indices = list(range(start_frame_idx, end_frame_idx + 1))
                while len(frame_indices) < frames_per_window:
                    frame_indices.append(frame_indices[-1])
                frame_indices = frame_indices[:frames_per_window]

            frame_paths = []
            for frame_idx in frame_indices:
                if frame_idx < len(frame_files):
                    frame_paths.append(str(frame_files[frame_idx]))
                else:
                    frame_paths.append(str(frame_files[-1]))

            if mode == "binary":
                label = determine_binary_label(window_start_time, window_end_time,
                                             fake_segments, video_type)
            else:
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
    if video_type == "real":
        return 0

    for seg_start, seg_end in fake_segments:
        if not (window_end <= seg_start or window_start >= seg_end):
            return 1

    return 0


def determine_four_class_label(window_start: float, window_end: float,
                              fake_segments: List[Tuple[float, float]],
                              video_type: str) -> int:
    """Return background, fake, real-to-fake, or fake-to-real as labels 0-3."""
    if video_type == "real":
        return 0

    window_duration = window_end - window_start
    if window_duration <= 0:
        raise ValueError(f"Invalid window [{window_start}, {window_end}]")

    overlaps = []
    for seg_start, seg_end in sorted(fake_segments):
        overlap_start = max(window_start, seg_start)
        overlap_end = min(window_end, seg_end)
        if overlap_start < overlap_end:
            overlaps.append((overlap_start, overlap_end))

    if not overlaps:
        return 0

    merged = []
    for overlap_start, overlap_end in overlaps:
        if merged and overlap_start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], overlap_end))
        else:
            merged.append((overlap_start, overlap_end))
    fake_duration = sum(end - start for start, end in merged)
    epsilon = min(1e-6, window_duration / 1000.0)
    if fake_duration >= window_duration - epsilon:
        return 1

    def is_fake_at(timestamp: float) -> bool:
        return any(seg_start <= timestamp < seg_end for seg_start, seg_end in fake_segments)

    starts_fake = is_fake_at(window_start + epsilon)
    ends_fake = is_fake_at(window_end - epsilon)
    if not starts_fake and ends_fake:
        return 2
    if starts_fake and not ends_fake:
        return 3

    return 3 if starts_fake else 2


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
        self.windows = windows
        self.mode = mode
        self.transform_config = transform_config or {}

        self.num_classes = 2 if mode == "binary" else 4

        self._balance_samples()

    def _balance_samples(self):
        if self.mode == "binary":
            positive_indices = [i for i, w in enumerate(self.windows) if w['label'] == 1]
            negative_indices = [i for i, w in enumerate(self.windows) if w['label'] == 0]

            min_samples = min(len(positive_indices), len(negative_indices))
            self.balanced_indices = []
            self.balanced_indices.extend(random.sample(positive_indices, min_samples))
            self.balanced_indices.extend(random.sample(negative_indices, min_samples))
        else:
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
        if select_frame_nums == 0:
            raise RuntimeError(f"Window has no frame paths: {window_info['window_id']}")

        aug_list = [albumentations.Resize(224, 224)]

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

        frames = []

        for frame_path in frame_paths:
            try:
                image = cv2.imread(frame_path)
                if image is None:
                    raise ValueError("cv2.imread returned None")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if self.transform_config.get('crop_youku', False) and 'youku' in frame_path:
                    image = crop_center_by_percentage(image, 0.15)

                augmented = trans(image=image)
                image = augmented["image"]
                frame = image.transpose(2, 0, 1)[np.newaxis, :]
            except Exception as exc:
                warnings.warn(f"Error loading frame {frame_path}: {exc}")
                frame = np.zeros((1, 3, 224, 224), dtype=np.float32)
            frames.append(frame)

        if len(frames) != select_frame_nums:
            raise RuntimeError(
                f"Unexpected frame count: expected={select_frame_nums}, actual={len(frames)}"
            )

        label_onehot = [0] * self.num_classes
        label_onehot[label] = 1

        frames = np.concatenate(frames, 0)
        frames = torch.tensor(frames[np.newaxis, :])
        label_onehot = torch.FloatTensor(label_onehot)
        binary_label = torch.FloatTensor([label])

        return real_idx, frames, label_onehot, binary_label


class VideoWindowDatasetTest(Dataset):
    def __init__(self, windows: List[Dict], mode: str = "binary", transform_config: Dict = None, task: str = "normal"):
        self.windows = windows
        self.mode = mode
        self.transform_config = transform_config or {}
        self.task = task

        self.num_classes = 2 if mode == "binary" else 4

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
        if select_frame_nums == 0:
            raise RuntimeError(f"Window has no frame paths: {window_info['window_id']}")

        aug_list = [albumentations.Resize(224, 224)]

        aug_list.append(albumentations.Normalize(mean=(0.485, 0.456, 0.406),
                                               std=(0.229, 0.224, 0.225),
                                               max_pixel_value=255.0, p=1.0))
        trans = albumentations.Compose(aug_list)

        frames = []

        for frame_path in frame_paths:
            try:
                image = cv2.imread(frame_path)
                if image is None:
                    raise ValueError("cv2.imread returned None")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if self.transform_config.get('crop_youku', False) and 'youku' in frame_path:
                    image = crop_center_by_percentage(image, 0.15)

                augmented = trans(image=image)
                image = augmented["image"]
                frame = image.transpose(2, 0, 1)[np.newaxis, :]
            except Exception as exc:
                warnings.warn(f"Error loading frame {frame_path}: {exc}")
                frame = np.zeros((1, 3, 224, 224), dtype=np.float32)
            frames.append(frame)

        if len(frames) != select_frame_nums:
            raise RuntimeError(
                f"Unexpected frame count: expected={select_frame_nums}, actual={len(frames)}"
            )

        label_onehot = [0] * self.num_classes
        label_onehot[label] = 1

        frames = np.concatenate(frames, 0)
        frames = torch.tensor(frames[np.newaxis, :])
        label_onehot = torch.FloatTensor(label_onehot)
        binary_label = torch.FloatTensor([label])
        return window_idx, frames, label_onehot, binary_label, video_name


def generate_dataset_loader_from_json(cfg):
    with open(cfg['train_json_path'], 'r') as f:
        train_json = json.load(f)
    with open(cfg['test_json_path'], 'r') as f:
        test_json = json.load(f)

    train_windows = parse_json_to_windows(
        train_json,
        cfg['dataset_base_path'],
        cfg.get('window_length', 1.0),
        cfg.get('frames_per_window', 4),
        cfg.get('mode', 'binary')
    )

    test_windows = parse_json_to_windows(
        test_json,
        cfg['dataset_base_path'],
        cfg.get('window_length', 1.0),
        cfg.get('frames_per_window', 4),
        cfg.get('mode', 'binary')
    )
    test_fake_segments = get_video_fake_segments(test_json)

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

def build_inference_loader(cfg):
    """Build the test loader without invoking the legacy training-loader path."""
    with open(cfg['test_json_path'], 'r', encoding='utf-8') as f:
        test_json = json.load(f)

    test_windows = parse_json_to_windows(
        test_json,
        cfg['dataset_base_path'],
        cfg.get('window_length', 1.0),
        cfg.get('frames_per_window', 4),
        cfg.get('mode', 'four_class')
    )
    test_dataset = VideoWindowDatasetTest(
        test_windows,
        mode=cfg.get('mode', 'four_class'),
        transform_config=cfg.get('transform_config', {}),
        task=cfg.get('task', 'normal')
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.get('batch_size', cfg.get('val_batch_size', 16)),
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        drop_last=False
    )

    print(f"******* Inference Windows: {len(test_windows)}")
    return test_loader, get_video_fake_segments(test_json)
