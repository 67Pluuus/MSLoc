"""Extract BGE features for anomaly classes."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def extract_classes(data_path):
    with open(data_path, encoding='utf-8') as handle:
        records = json.load(handle)
    classes = {'Normal'}
    for item in records:
        for annotation in item['annotations']:
            for key, field in (('bnd_cot_st', 'bnd_class'), ('obj_cot', 'bnd_sub_class'),
                               ('bnd_cot_ed', 'bnd_class')):
                value = annotation.get(key, [])
                if isinstance(value, list):
                    value = value[0] if value else {}
                if not isinstance(value, dict):
                    raise TypeError(f'Invalid {key} for {item["video_path"]}')
                name = value.get(field, '')
                if name:
                    if not isinstance(name, str):
                        raise TypeError(f'Expected a class name for {item["video_path"]}: {name}')
                    classes.add(name)
    if classes == {'Normal'}:
        raise ValueError('No anomaly classes found in the supplied training annotations')
    return sorted(classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--bge_model_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError('batch_size must be positive')
    model_path = Path(args.bge_model_path)
    if not model_path.is_dir() or 'bge-large-en-v1.5' not in str(model_path).lower():
        raise ValueError('bge_model_path must point to the local bge-large-en-v1.5 model')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    torch.manual_seed(42)

    # Load the class names and text encoder.
    classes = extract_classes(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(args.device).eval()

    # Extract normalized CLS features.
    features = []
    with torch.no_grad():
        for start in tqdm(range(0, len(classes), args.batch_size), desc='Encoding anomaly classes'):
            batch = tokenizer(classes[start:start + args.batch_size], padding=True,
                              truncation=True, max_length=512, return_tensors='pt').to(args.device)
            embeddings = model(**batch).last_hidden_state[:, 0]
            features.append(F.normalize(embeddings, p=2, dim=1).float().cpu())
    features = torch.cat(features)
    if features.shape != (len(classes), 1024) or not torch.isfinite(features).all():
        raise ValueError(f'Invalid BGE class features: {features.shape}')
    # Save the class feature bank.
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(class_features=features, class_names=classes, feat_dim=1024,
                    num_classes=len(classes), model_name='bge-large-en-v1.5',
                    pooling_method='CLS', normalized=True), output_path)
    print(f'Saved {len(classes)} anomaly classes to {output_path}')


if __name__ == '__main__':
    main()
