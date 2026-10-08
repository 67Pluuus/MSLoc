"""Create the shared video-level train/validation split for MSLoc."""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_file', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.val_ratio < 1:
        raise ValueError('val_ratio must be between 0 and 1')
    with open(args.anno_file, encoding='utf-8') as handle:
        records = json.load(handle)
    video_ids = [item['video_path'] for item in records]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError('Duplicate video_path entries in training annotations')

    # Split by video to avoid overlap between training and validation.
    count = round(len(records) * args.val_ratio)
    if not 0 < count < len(records):
        raise ValueError('Both training and validation splits must be nonempty')
    val_ids = set(random.Random(args.seed).sample(video_ids, count))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, selected in (
        ('msloc_train.json', [r for r in records if r['video_path'] not in val_ids]),
        ('msloc_val.json', [r for r in records if r['video_path'] in val_ids]),
    ):
        output_path = output_dir / name
        # Reuse an existing split only if its contents match.
        if output_path.exists():
            with output_path.open(encoding='utf-8') as handle:
                if json.load(handle) != selected:
                    raise FileExistsError(f'Different split already exists: {output_path}')
        else:
            with output_path.open('w', encoding='utf-8') as handle:
                json.dump(selected, handle, ensure_ascii=False, indent=2)
        print(f'{name}: {len(selected)} videos -> {output_path}')


if __name__ == '__main__':
    main()
