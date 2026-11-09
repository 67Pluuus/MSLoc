"""Checkpointing, reproducibility checks, and training curves for MSLoc-PG."""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_save(value, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, destination)


def save_json(value, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, destination)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_run(directory, cfg, seed, resume=False, clean=False, checkpoint=None):
    """Bind outputs to exact inputs and reject accidental overwrites."""
    if resume and clean:
        raise ValueError('--resume and --clean cannot be combined')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'run_config.json'
    source = Path(__file__).parent
    identity = dict(config=cfg, seed=seed,
                    annotations={key: sha256(cfg[key]) for key in ('train_json_path', 'test_json_path')},
                    implementation={name: sha256(source / name) for name in
                                    ('train.py', 'eval.py', 'util.py', 'dataloader.py', 'runtime.py',
                                     'models/DeMamba.py', 'models/clip/pscan.py')})
    if checkpoint:
        identity['checkpoint_sha256'] = sha256(checkpoint)
    if clean:
        if not manifest.is_file() and any(directory.iterdir()):
            raise ValueError('Refusing to clean an output directory without run_config.json')
        owned = {'run_config.json', 'training_state.pth', 'history.json', 'training_curves.png',
                 'best_acc.pth', 'inference_cache.pth', 'validation_cache.pth', 'predictions.json', 'evaluation_results.json',
                 'summary_results.json', 'detailed_results.csv', 'fake_proposal_quality.csv'}
        for path in directory.iterdir():
            name = path.name.removesuffix('.tmp')
            if path.is_file() and (name in owned or
                                  (name.endswith('.pth') and name[:-4].isdigit()) or
                                  (name.startswith('Epoch_') and name.endswith('_accuracy.txt'))):
                path.unlink()
    if manifest.exists():
        if not resume:
            raise FileExistsError(f'Existing run: {directory}; use --resume or --clean')
        if json.loads(manifest.read_text()) != identity:
            raise ValueError('Resume requires identical source, config, seed, annotations and checkpoint')
    elif resume:
        raise FileNotFoundError(f'No run to resume: {manifest}')
    else:
        # Shell log files may already be open; reject previous model/eval products.
        if any(path.suffix in {'.pth', '.json'} for path in directory.iterdir()):
            raise FileExistsError(f'Unmanaged results already exist: {directory}')
        save_json(identity, manifest)


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def save_training_state(directory, model, optimizer, scheduler, next_epoch, best_epoch, best_score,
                        history, pending_training=None):
    atomic_save(dict(model_state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), next_epoch=next_epoch, best_epoch=best_epoch,
                     best_score=best_score, history=history, rng=capture_rng(),
                     pending_training=pending_training),
                Path(directory) / 'training_state.pth')


def plot_history(history, directory):
    rng = capture_rng()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    epochs = [row['epoch'] for row in history]
    axes[0].plot(epochs, [row['train_loss'] for row in history], marker='o')
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy')
    for key in ('F1Det', 'F1Loc'):
        axes[1].plot(epochs, [row[key] for row in history], marker='o', label=key)
    axes[1].set(xlabel='Epoch', ylabel='Test score', ylim=(0, 1))
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(Path(directory) / 'training_curves.png', dpi=160)
    plt.close(figure)
    restore_rng(rng)
