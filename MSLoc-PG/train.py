import argparse
import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import MultiStepLR

from dataloader import generate_dataset_loader_from_json
from util import build_model, train_one_epoch


def train_model(cfg, args):
    """Train the four-class DeMamba model."""
    if cfg['model'] != 'XCLIP_DeMamba_4' or cfg['mode'] != 'four_class':
        raise ValueError('MSLoc-PG training requires the boundary-aware four-class model')
    if cfg['tuning_mode'] != 'sft':
        raise ValueError('This entry point trains the standard DeMamba model with SFT')
    if cfg['window_length'] != 2.0 or cfg['frames_per_window'] != 8:
        raise ValueError('MSLoc-PG uses 2-second windows and 8 frames per window')
    with open(cfg['train_json_path'], encoding='utf-8') as handle:
        train_ids = {item['video_path'] for item in json.load(handle)}
    with open(cfg['test_json_path'], encoding='utf-8') as handle:
        val_ids = {item['video_path'] for item in json.load(handle)}
    if not train_ids or not val_ids or train_ids & val_ids:
        raise ValueError('Training and validation must contain nonempty, disjoint video sets')
    # Stop on missing or unreadable data.
    warnings.filterwarnings('error', category=UserWarning, module='dataloader')

    # Set the random seed and devices.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device_ids = [int(value) for value in args.device_ids.split(',')]
    if (not torch.cuda.is_available() or not device_ids or min(device_ids) < 0
            or max(device_ids) >= torch.cuda.device_count()
            or len(set(device_ids)) != len(device_ids)):
        raise ValueError(f'Invalid CUDA device IDs: {args.device_ids}')
    torch.cuda.set_device(device_ids[0])

    # Prepare the output directory.
    output_dir = Path(cfg['save_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / 'training_state.pth'
    config_path = output_dir / 'training_config.yaml'
    run_config = dict(cfg, seed=args.seed)
    if args.clean:
        if not config_path.is_file():
            raise ValueError('--clean requires an output directory created by this training entry point')
        # Remove training outputs only.
        for path in output_dir.iterdir():
            if path.is_file() and (
                path.name in {'training_state.pth', 'training_state.tmp', 'training_config.yaml', 'best_acc.pth'}
                or (path.suffix == '.pth' and path.stem.isdecimal())
                or (path.name.startswith('Epoch_') and path.name.endswith('_accuracy.txt'))
            ):
                path.unlink()
    if args.resume:
        if not state_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(f'No resumable run in {output_dir}')
        with config_path.open(encoding='utf-8') as handle:
            if yaml.safe_load(handle) != run_config:
                raise ValueError('Resume requires the original training configuration and seed')
    elif config_path.exists() or any(output_dir.glob('*.pth')):
        raise FileExistsError(f'Training outputs already exist in {output_dir}; use --resume or --clean')
    with config_path.open('w', encoding='utf-8') as handle:
        yaml.safe_dump(run_config, handle, sort_keys=False)

    # Build the model and optimizer.
    model = build_model(cfg['model']).cuda()
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-8)
    scheduler = MultiStepLR(optimizer, milestones=[7], gamma=0.3)
    loss = torch.nn.CrossEntropyLoss()
    start_epoch, best_epoch, best_score = 0, 0, float('-inf')

    # Resume from the last completed epoch.
    if args.resume:
        state = torch.load(state_path, map_location='cpu')
        unwrapped = model.module if isinstance(model, torch.nn.DataParallel) else model
        unwrapped.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch, best_epoch, best_score = state['epoch'], state['best_epoch'], state['best_score']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])

    # Run training and validation.
    print(f'MSLoc-PG: four_class, SFT, seed={args.seed}, devices={device_ids}')
    for epoch in range(start_epoch, cfg['max_epoch']):
        print(f'******* Training epoch {epoch + 1}/{cfg["max_epoch"]} *******')
        train_loader, val_loader, segments = generate_dataset_loader_from_json(cfg)
        if len(train_loader) == 0 or len(val_loader) == 0:
            raise ValueError('Empty training or validation loader; check data and batch size')
        best_epoch, best_score, elapsed = train_one_epoch(
            cfg, model, loss, scheduler, optimizer, epoch, best_epoch, best_score,
            train_loader, val_loader, str(output_dir), segments,
        )
        unwrapped = model.module if isinstance(model, torch.nn.DataParallel) else model
        state = dict(
            epoch=epoch + 1, best_epoch=best_epoch, best_score=best_score,
            model=unwrapped.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), python_rng=random.getstate(),
            numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),
        )
        temporary = output_dir / 'training_state.tmp'
        torch.save(state, temporary)
        os.replace(temporary, state_path)
        print(f'Epoch {epoch + 1} completed in {elapsed:.1f}s; best validation Loc_F1={best_score:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device-ids', default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true', help='Resume the last completed epoch')
    parser.add_argument('--clean', action='store_true', help='Remove this run\'s training outputs')
    args = parser.parse_args()
    if args.resume and args.clean:
        raise ValueError('--resume and --clean are mutually exclusive')

    # Load the config.
    with open(args.config, encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)

    train_model(cfg, args)


if __name__ == '__main__':
    main()
