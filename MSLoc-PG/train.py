"""Train MSLoc-PG with boundary-aware four-class supervision."""
import random
import json
from pathlib import Path
from runtime import prepare_run, restore_rng, save_training_state, save_json, plot_history
import argparse
import yaml
import torch
import numpy as np
import os
import util
from util import build_model, train_one_epoch
from dataloader import generate_dataset_loader_from_json
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, LambdaLR, MultiStepLR


def parse_device_ids(value):
    """Parse explicit CUDA device IDs, e.g. ``0`` or ``0,1,...,7``."""
    try:
        device_ids = [int(item.strip()) for item in value.split(',') if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError('--device-ids must be comma-separated integers') from error
    if not device_ids or len(set(device_ids)) != len(device_ids) or min(device_ids) < 0:
        raise argparse.ArgumentTypeError('--device-ids must contain unique non-negative integers')
    return device_ids


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings of detector in yaml format')
    parser.add_argument('--device-ids', default='0,1,2,3,4,5,6,7',
                        help='CUDA device IDs: 0 for one GPU, or 0,1,2,3,4,5,6,7 for eight GPUs')
    parser.add_argument('--train-batch-size', type=int, default=None,
                        help='Optional override for cfg.train_batch_size')
    parser.add_argument('--val-batch-size', type=int, default=None,
                        help='Optional override for cfg.val_batch_size')
    parser.add_argument('--micro-batch-size', type=int, default=0,
                        help='Windows per training forward across selected GPUs; 0 disables accumulation.')
    parser.add_argument('--eval-micro-batch-size', type=int, default=0,
                        help='Windows per evaluation forward; 0 uses the complete logical batch.')
    parser.add_argument('--max-epoch', type=int, default=None,
                        help='Optional override for cfg.max_epoch')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible training')
    parser.add_argument('--resume', action='store_true', help='Resume saved training or pending validation.')
    parser.add_argument('--clean', action='store_true', help='Remove this run\'s generated outputs.')
    parser.add_argument('--train-videos', type=int, default=0, help='0 uses all training videos.')
    parser.add_argument('--val-videos', type=int, default=0, help='0 uses all evaluation videos.')
    args = parser.parse_args()
    args.device_ids = parse_device_ids(args.device_ids)
    return args


if __name__ == '__main__':
    args = get_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    for key, value in (('train_batch_size', args.train_batch_size),
                       ('val_batch_size', args.val_batch_size),
                       ('max_epoch', args.max_epoch)):
        if value is not None:
            if value <= 0:
                raise ValueError(f'--{key.replace("_", "-")} must be positive')
            cfg[key] = value
    if args.train_videos < 0 or args.val_videos < 0:
        raise ValueError('Video limits must be non-negative')
    cfg.update(train_videos=args.train_videos, val_videos=args.val_videos, device_ids=args.device_ids)
    if args.micro_batch_size < 0 or args.eval_micro_batch_size < 0:
        raise ValueError('Micro-batch sizes must be non-negative')
    cfg.update(micro_batch_size=args.micro_batch_size,
               eval_micro_batch_size=args.eval_micro_batch_size)
    print(f"Training batch={cfg['train_batch_size']}, forward batch="
          f"{args.micro_batch_size or cfg['train_batch_size']}; "
          f"evaluation forward batch={args.eval_micro_batch_size or cfg['val_batch_size']}")
    train_records = json.loads(Path(cfg['train_json_path']).read_text())
    test_records = json.loads(Path(cfg['test_json_path']).read_text())
    if {r['video_path'] for r in train_records} & {r['video_path'] for r in test_records}:
        raise ValueError('Training and evaluation videos overlap')
    print(f"******* Random seed: {args.seed} *******")
    print("******* Building models. *******")
    print(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by the current DeMamba training loop')
    available = torch.cuda.device_count()
    if max(args.device_ids) >= available:
        raise ValueError(f'Requested --device-ids {args.device_ids}, but only {available} CUDA device(s) are visible')
    primary_device = args.device_ids[0]
    torch.cuda.set_device(primary_device)
    prepare_run(cfg['save_dir'], cfg, args.seed, args.resume, args.clean)
    print(f"******* Using CUDA device IDs: {args.device_ids} (primary cuda:{primary_device}) *******")
    model = util.build_model(
        cfg['model'],
        neuron_indices_path=cfg.get('neuron_indices_path'),
        xclip_model_path=cfg.get('xclip_model_path'),
        xclip_checkpoint_path=cfg.get('xclip_checkpoint_path'),
    )
    model = model.to(torch.device(f'cuda:{primary_device}'))

    if cfg['tuning_mode'] == 'lp':
        for param in model.encoder.parameters():
            param.requires_grad = False

    if len(args.device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids, output_device=primary_device)

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters remain after applying tuning_mode")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=cfg['lr'], weight_decay=1e-8)
    scheduler = MultiStepLR(optimizer, milestones=[7], gamma=0.3)
    if cfg['mode'] == 'binary':
        loss = nn.BCEWithLogitsLoss()
    else:
        loss = nn.CrossEntropyLoss()

    trMaxEpoch = cfg['max_epoch']
    snapshot_path = cfg['save_dir']
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    max_epoch, max_acc = 0, 0
    start_epoch, history = 0, []
    pending_training, validation_rng = None, None
    state_path = Path(snapshot_path) / 'training_state.pth'
    if args.resume:
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch, max_epoch, max_acc = state['next_epoch'], state['best_epoch'], state['best_score']
        history = state['history']
        pending_training = state.get('pending_training')
        validation_rng = state['rng'] if pending_training is not None else None
        restore_rng(state['rng'])
        del state
    else:
        save_training_state(snapshot_path, model, optimizer, scheduler, 0, max_epoch, max_acc, history)

    for epochID in range(start_epoch, trMaxEpoch):
        print("******* Training epoch", str(epochID)," *******")
        print("******* Building datasets. *******")
        train_loader, val_loader, test_fake_segments = generate_dataset_loader_from_json(cfg)
        cache_path = Path(snapshot_path) / 'validation_cache.pth'
        cfg['_evaluation_cache'] = str(cache_path)
        if pending_training is None:
            cache_path.unlink(missing_ok=True)
        elif validation_rng is not None:
            # Rebuilding the loaders must not consume the saved validation RNG.
            restore_rng(validation_rng)

        def checkpoint_before_validation(pending):
            save_training_state(snapshot_path, model, optimizer, scheduler, epochID,
                                max_epoch, max_acc, history, pending_training=pending)

        max_epoch, max_acc, epoch_time, train_loss, metrics = train_one_epoch(
            cfg, model, loss, scheduler, optimizer, epochID, max_epoch, max_acc,
            train_loader, val_loader, snapshot_path, test_fake_segments,
            pending_training=pending_training, on_training_complete=checkpoint_before_validation,
        )
        history.append(dict(epoch=epochID + 1, train_loss=train_loss, F1Det=metrics['F1Det'],
                            F1Loc=metrics['F1Loc'], lr=optimizer.param_groups[0]['lr']))
        save_training_state(snapshot_path, model, optimizer, scheduler, epochID + 1, max_epoch, max_acc, history)
        cache_path.unlink(missing_ok=True)
        pending_training, validation_rng = None, None
        save_json(history, Path(snapshot_path) / 'history.json')
        plot_history(history, snapshot_path)
        print("******* Ending epoch", str(epochID)," Time ", str(epoch_time), "*******")
