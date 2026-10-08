# Original preprocessing: LLaVA / FastChat, Apache-2.0.

import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from trace.constants import IGNORE_INDEX, MMODAL_TOKEN_INDEX
from trace.conversation import conv_templates
from trace.mm_utils import process_video_ref_split, tokenizer_MMODAL_token_all
from trace.model.multimodal_encoder.time_encoder import TimeTokenizer, TimeTower
from trace.model.multimodal_encoder.score_encoder import ScoreTokenizer, ScoreTower


QUESTION = ('<video>\nPlease examine and localize any inconsistencies or obvious signs of '
            'forgery in the video, stating their commencement and completion timestamps '
            'and provide a succinct explanation.')


def load_records(path):
    with open(path, encoding='utf-8') as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f'Expected a nonempty annotation list: {path}')
    ids = [item['video_path'] for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate video_path entries: {path}')
    return records


def cot_entry(annotation, key):
    """Read a rationale dict or the first entry in a rationale list."""
    value = annotation.get(key, [])
    if isinstance(value, list):
        value = value[0] if value else {}
    if not isinstance(value, dict):
        raise TypeError(f'{key} must be a dict or a list of dicts')
    return value


def annotation_target(annotation):
    start = cot_entry(annotation, 'bnd_cot_st')
    event = cot_entry(annotation, 'obj_cot')
    end = cot_entry(annotation, 'bnd_cot_ed')
    if 'Round4' in annotation['combine_dir']:
        caption = 'Spatio-temporal forgery.\n' + event['obj_caption']
        classes = ['', event.get('bnd_sub_class', ''), '']
    else:
        caption = 'Temporal forgery.\n' + '\n'.join((
            start.get('bnd_caption', ''), event['obj_caption'], end.get('bnd_caption', ''),
        ))
        classes = [start.get('bnd_class', ''), event.get('bnd_sub_class', ''), end.get('bnd_class', '')]
    if not event['obj_caption'].strip() or not any(classes):
        raise ValueError(f'Missing forensic rationale or anomaly classes: {annotation}')
    return caption, classes


def validate_interval(interval, duration, context):
    if len(interval) != 2:
        raise ValueError(f'Expected [start, end] for {context}: {interval}')
    start, end = map(float, interval)
    if not all(map(math.isfinite, (start, end, duration))) or not 0 <= start < end <= duration + 1e-6:
        raise ValueError(f'Invalid interval for {context}: {interval}, duration={duration}')
    return start, end


class ProposalDataset(Dataset):
    def __init__(self, data_path, proposal_path, data_folder, tokenizer, processor,
                 class_to_idx, bnd_ratio=0.2, video_dtype=torch.float32, show_progress=True,
                 max_videos=0):
        if max_videos < 0:
            raise ValueError('max_videos must be nonnegative')
        self.tokenizer = tokenizer
        self.processor = processor
        self.data_folder = Path(data_folder)
        self.class_to_idx = class_to_idx
        self.bnd_ratio = bnd_ratio
        self.video_dtype = video_dtype
        self.records = load_records(data_path)
        proposals = {item['video_path']: item for item in load_records(proposal_path)}
        self.all_video_ids = {item['video_path'] for item in self.records}
        if max_videos > 0:
            # Explicit debug mode selects from the PG output within this split.
            available = [item for item in self.records if item['video_path'] in proposals]
            self.records = available[:max_videos]
            if not self.records:
                raise ValueError(f'No PG predictions overlap {data_path}; generate proposals for this split')
            if show_progress:
                print(f'Debug videos: {len(self.records)}/{len(self.all_video_ids)} from {data_path}; '
                      f'PG coverage={len(available)}, limit={max_videos}')
        self.video_ids = {item['video_path'] for item in self.records}
        missing = self.video_ids - proposals.keys()
        if missing:
            raise ValueError(f'Missing PG predictions for {len(missing)} videos, e.g. {sorted(missing)[:3]}')
        # Count time and score token placeholders.
        self.time_encoder = TimeTower(TimeTokenizer(), hidden_dim=1)
        self.score_encoder = ScoreTower(ScoreTokenizer(), hidden_dim=1)
        self.samples = []
        no_proposal = 0
        for item in tqdm(self.records, desc='Preparing proposals', disable=not show_progress):
            video = item['video_path']
            duration = float(item['duration'])
            if not (self.data_folder / video).is_file():
                raise FileNotFoundError(self.data_folder / video)
            if item['type'] not in {'real', 'fake'}:
                raise ValueError(f'Unknown video type for {video}: {item["type"]}')
            annotations = item['annotations'] if item['type'] == 'fake' else []
            if item['type'] == 'fake' and not annotations:
                raise ValueError(f'Fake video has no segment annotations: {video}')
            for ann in annotations:
                validate_interval(ann['segment'], duration, video)
                _, classes = annotation_target(ann)
                unknown = {name for name in classes if name and name not in class_to_idx}
                if unknown:
                    raise ValueError(f'Unknown anomaly classes for {video}: {unknown}')
            intervals = proposals[video]['model_inference']['segment']
            if not intervals:
                no_proposal += 1
            for interval in intervals:
                window = validate_interval(interval, duration, video)
                self.samples.append((video, window, annotations))
        if not self.samples:
            raise ValueError(f'No proposals available for {data_path}')
        if show_progress:
            print(f'{data_path}: {len(self.samples)} proposals; {no_proposal} videos with no PG proposals')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_path, (win_start, win_end), annotations = self.samples[index]
        video, timestamps = process_video_ref_split(
            str(self.data_folder / video_path), self.processor, aspect_ratio='pad',
            bnd_frames=16, seg_frames=8, bnd_ratio=self.bnd_ratio,
            start_time=win_start, end_time=win_end,
        )
        if video.shape[0] != 40 or len(timestamps) != 40 or not torch.isfinite(video).all():
            raise ValueError(f'Expected 40 finite video frames: {video_path}')
        times, captions, matched_classes = [], [], []
        for ann in annotations:
            start, end = ann['segment']
            overlap_start, overlap_end = max(start, win_start), min(end, win_end)
            if overlap_end > overlap_start:
                # Align clipped segment boundaries to sampled timestamps.
                relative = [overlap_start - win_start, overlap_end - win_start]
                times.append([min(timestamps, key=lambda t: abs(t[0] - value))[0] for value in relative])
                caption, classes = annotation_target(ann)
                captions.append(caption)
                matched_classes.append(classes)
        if times:
            # Use the first matched segment for the three anomaly labels.
            classes = matched_classes[0]
        else:
            times, captions = [[]], ['No forgery.']
            classes = ['Normal', 'Normal', 'Normal']
        scores = [[] for _ in times]
        answer = ''.join(
            '<sync>' + '<time>' * len(self.time_encoder.encode(interval))
            + '<score>' * len(self.score_encoder.encode(score)) + caption
            for interval, score, caption in zip(times, scores, captions)
        )
        conv = conv_templates['mistral_instruct'].copy()
        conv.append_message(conv.roles[0], QUESTION)
        conv.append_message(conv.roles[1], answer)
        input_ids = tokenizer_MMODAL_token_all(conv.get_prompt(), self.tokenizer, return_tensors='pt')
        # Mask the prompt before the first answer token.
        sync_positions = torch.where(input_ids == MMODAL_TOKEN_INDEX['SYNC'])[0]
        if len(sync_positions) != len(times):
            raise ValueError(f'Unexpected sync markers in the rationale for {video_path}')
        labels = input_ids.clone()
        labels[:sync_positions[0]] = IGNORE_INDEX
        # RefProjector: 32 boundary frames * (8 slots + 2 difference tokens + 6 time tokens)
        # plus 32 event slots + 8*6 event time tokens and 3 anomaly tokens, replacing one video marker.
        expanded_length = len(input_ids) - 1 + 32 * 16 + 32 + 8 * 6 + 3
        if expanded_length > self.tokenizer.model_max_length:
            raise ValueError(f'{video_path}: {expanded_length} multimodal tokens exceed model_max_length; '
                             'increase the limit instead of truncating supervision')
        class_indices = [self.class_to_idx[name] if name else IGNORE_INDEX for name in classes]
        if all(value == IGNORE_INDEX for value in class_indices):
            raise ValueError(f'No anomaly supervision for {video_path}')
        return dict(input_ids=input_ids, labels=labels, video=video.to(self.video_dtype),
                    time=times, score=scores, video_timestamps=timestamps,
                    class_indices=torch.tensor(class_indices, dtype=torch.long))


class DataCollatorForSupervisedDataset:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item['input_ids'] for item in instances], batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [item['labels'] for item in instances], batch_first=True, padding_value=IGNORE_INDEX,
        )
        # Use lengths because pad_token and unk_token share an ID.
        lengths = torch.tensor([len(item['input_ids']) for item in instances])
        mask = torch.arange(input_ids.shape[1])[None, :] < lengths[:, None]
        return dict(
            input_ids=input_ids, labels=labels, attention_mask=mask,
            images=[[item['video'] for item in instances], ['video'] * len(instances)],
            times=[item['time'] for item in instances], scores=[item['score'] for item in instances],
            video_timestamps=[item['video_timestamps'] for item in instances],
            closs_labels=torch.stack([item['class_indices'] for item in instances]),
        )
