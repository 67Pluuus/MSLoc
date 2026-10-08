# Original training pipeline: LLaVA / FastChat, Apache-2.0.

import os
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Load the local trace package before the standard library module.
project_root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, project_root)
if 'trace' in sys.modules and not hasattr(sys.modules['trace'], '__path__'):
    del sys.modules['trace']
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, Trainer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from trace.model import TraceMistralForCausalLM
from trace.model.multimodal_projector.builder import RefProjector
from trace.train_data import ProposalDataset, DataCollatorForSupervisedDataset


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default='../../MSLoc_assets/Trace/ckpts/trace-uni')
    vision_tower: str = field(default='../../MSLoc_assets/Trace/ckpts/clip-vit-large-patch14-336')
    class_feature_path: str = field(default='../../MSLoc_assets/Trace/class_features_bge.pt')
    mm_projector_type: str = field(default='ref_projector')
    mm_vision_select_layer: int = field(default=-2)
    mm_vision_select_feature: str = field(default='patch')
    pretrain_mm_mlp_adapter: str = field(default=None)
    downsample_num: int = field(default=1)
    num_frames: int = field(default=40)
    closs: bool = field(default=True)
    freeze_backbone: bool = field(default=True)
    tune_mm_embed_head: bool = field(default=True)
    tune_lm_embed_head: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(default='../../MSLoc_assets/data/Tasle-CoT-10K/annos/msloc_train.json')
    eval_data_path: str = field(default='../../MSLoc_assets/data/Tasle-CoT-10K/annos/msloc_val.json')
    proposal_path: str = field(default='../../MSLoc_assets/DeMamba/results/all_Class_4/eval_train/predictions.json')
    data_folder: str = field(default='../../MSLoc_assets/data/Tasle-CoT-10K/videos')
    bnd_ratio: float = field(default=0.2)
    bnd_frames: int = field(default=16)
    seg_frames: int = field(default=8)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default='../../MSLoc_assets/Trace/output/trace_vllava/ref2')
    num_train_epochs: float = field(default=2.0)
    learning_rate: float = field(default=3e-5)
    global_batch_size: int = field(default=16)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=1)
    train_videos: int = field(default=0, metadata={
        'help': 'Debug: use up to N training videos with PG predictions; 0 requires all videos'})
    val_videos: int = field(default=0, metadata={
        'help': 'Debug: use up to N validation videos with PG predictions; 0 requires all videos'})
    gradient_accumulation_steps: Optional[int] = field(default=None)
    weight_decay: float = field(default=0.0)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default='cosine')
    optim: str = field(default='adamw_torch')
    evaluation_strategy: str = field(default='epoch')
    save_strategy: str = field(default='epoch')
    logging_steps: float = field(default=1)
    dataloader_num_workers: int = field(default=4)
    remove_unused_columns: bool = field(default=False)
    prediction_loss_only: bool = field(default=True)
    label_names: List[str] = field(default_factory=lambda: ['labels'])
    report_to: List[str] = field(default_factory=list)
    seed: int = field(default=42)
    data_seed: int = field(default=42)
    bf16: bool = field(default=True)
    tf32: bool = field(default=False)
    gradient_checkpointing: bool = field(default=False)
    ddp_find_unused_parameters: bool = field(default=False)
    model_max_length: int = field(default=4096)
    resume_from_checkpoint: str = field(default=None)
    clean: bool = field(default=False)
    disable_tqdm: bool = field(default=False)

    def __post_init__(self):
        if self.train_videos < 0 or self.val_videos < 0:
            raise ValueError('train_videos and val_videos must be nonnegative')
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        forward_batch = self.per_device_train_batch_size * world_size
        if self.global_batch_size <= 0 or forward_batch <= 0 or self.global_batch_size % forward_batch:
            raise ValueError('global_batch_size must be a positive multiple of per-device batch * world size')
        accumulation = self.global_batch_size // forward_batch
        if self.gradient_accumulation_steps is None:
            self.gradient_accumulation_steps = accumulation
        elif self.gradient_accumulation_steps != accumulation:
            raise ValueError('Batch size, world size and accumulation must match global_batch_size')
        super().__post_init__()


class TraceTrainer(Trainer):
    """Trainer for joint localization, explanation and classification loss."""

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        if loss is None or not torch.isfinite(loss).all():
            raise FloatingPointError('MSLoc-PR returned a missing or nonfinite training loss')
        return (loss, outputs) if return_outputs else loss


def initialize_ref_projector(model, model_args):
    """Initialize boundary and event slots from the spatial projector."""
    original = model.get_model().mm_projector
    if isinstance(original, RefProjector):
        model.get_model().initialize_vision_modules(model_args)
        return
    if model.config.mm_projector_type != 'spatial_slot':
        raise ValueError('Expected a TRACE spatial_slot base model or an existing ref_projector')
    original_state = {name: value.detach().clone() for name, value in original.state_dict().items()}
    model.get_model().initialize_vision_modules(model_args)
    projector = model.get_model().mm_projector
    with torch.no_grad():
        for destination in (projector.boundary_proj, projector.event_proj):
            slots = original_state['slots']
            repeats = (destination.slots.shape[1] + slots.shape[1] - 1) // slots.shape[1]
            destination.slots.copy_(slots.repeat(1, repeats)[:, :destination.slots.shape[1]])
            destination.ln_vision.weight.copy_(original_state['ln_vision.weight'])
            destination.ln_vision.bias.copy_(original_state['ln_vision.bias'])
            destination.readout.weight.copy_(original_state['readout.weight'])


def configure_trainable_parameters(model, model_args):
    """Set trainable adapters, embeddings and prediction heads."""
    model.requires_grad_(not model_args.freeze_backbone)
    model.get_vision_tower().requires_grad_(False)
    model.get_model().mm_projector.requires_grad_(True)
    for module in (model.get_time_tower(), model.get_score_tower(), model.get_sync_tower(),
                   model.time_head, model.score_head, model.sync_head):
        module.requires_grad_(model_args.tune_mm_embed_head)
    model.get_input_embeddings().requires_grad_(model_args.tune_lm_embed_head)
    model.lm_head.requires_grad_(model_args.tune_lm_embed_head)
    model.get_model().closs_tokens.requires_grad_(True)
    model.get_model().closs_head.requires_grad_(True)
    # Freeze unused contrastive modules.
    model.get_model().closs_projector.requires_grad_(False)
    model.get_model().class_feature_bank.requires_grad_(False)
    model.get_model().closs_logit_scale.requires_grad_(False)


def check_closs_logits(module, inputs, output):
    # Check logits before the classification loss replaces nonfinite values.
    if not torch.isfinite(output).all():
        raise FloatingPointError('Nonfinite anomaly classification logits')


def prepare_training_output(model_args, data_args, training_args):
    """Prepare one training directory on rank zero, then share the result."""
    result = [None, None]
    if training_args.process_index == 0:
        try:
            output_dir = Path(training_args.output_dir).resolve()
            resume = training_args.resume_from_checkpoint
            if training_args.clean and resume:
                raise ValueError('--clean and --resume_from_checkpoint are mutually exclusive')
            if training_args.overwrite_output_dir:
                raise ValueError('Use --clean to restart; overwrite_output_dir can leave stale checkpoints')
            if output_dir.exists() and not output_dir.is_dir():
                raise NotADirectoryError(f'Training output path is not a directory: {output_dir}')
            marker = output_dir / 'msloc_training.json'
            if training_args.clean:
                protected = [model_args.model_name_or_path, model_args.vision_tower,
                             model_args.class_feature_path, data_args.data_path, data_args.eval_data_path,
                             data_args.proposal_path, data_args.data_folder, project_root]
                if any(Path(path).resolve().is_relative_to(output_dir) for path in protected):
                    raise ValueError('The output directory contains model, data or code inputs')
            if training_args.clean and output_dir.is_dir() and any(output_dir.iterdir()):
                owned = marker.is_file() and json.loads(marker.read_text()) == {'stage': 'MSLoc-PR'}
                # Recognize runs saved before the training marker was introduced.
                if not owned and (output_dir / 'config.json').is_file():
                    config = json.loads((output_dir / 'config.json').read_text())
                    owned = (config.get('model_type') == 'trace_mistral'
                             and config.get('mm_projector_type') == 'ref_projector'
                             and (output_dir / 'training_args.bin').is_file())
                if not owned:
                    raise ValueError(f'Cannot clean nonempty, unrecognized training directory: {output_dir}; '
                                     'use a different --output_dir or move existing files before restarting')
                marker.write_text(json.dumps({'stage': 'MSLoc-PR'}), encoding='utf-8')
                for path in output_dir.iterdir():
                    if path == marker:
                        continue
                    if path.is_symlink():
                        path.unlink()
                    elif path.is_dir():
                        if not path.resolve().is_relative_to(output_dir):
                            raise ValueError(f'Output path escapes the training directory: {path}')
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                print(f'Cleaned MSLoc-PR training outputs: {output_dir}')
            if resume == 'latest':
                resume = get_last_checkpoint(str(output_dir)) if output_dir.is_dir() else None
                if resume is None:
                    raise FileNotFoundError(f'No saved training state in {output_dir}')
            if resume:
                checkpoint = Path(resume)
                if checkpoint.resolve().parent != output_dir:
                    raise ValueError('Resume from a checkpoint inside output_dir')
                required = ['trainer_state.json', 'optimizer.pt', 'scheduler.pt']
                required += ([f'rng_state_{rank}.pth' for rank in range(training_args.world_size)]
                             if training_args.world_size > 1 else ['rng_state.pth'])
                missing = [name for name in required if not (checkpoint / name).is_file()]
                if missing:
                    raise FileNotFoundError(f'Incomplete resume state in {checkpoint}: {missing}')
                rng_files = {path.name for path in checkpoint.glob('rng_state*.pth')}
                if rng_files != {name for name in required if name.startswith('rng_state')}:
                    raise ValueError('Resume with the same GPU process count as the saved run')
                weights = ['model.safetensors', 'pytorch_model.bin',
                           'model.safetensors.index.json', 'pytorch_model.bin.index.json']
                if not any((checkpoint / name).is_file() for name in weights):
                    raise FileNotFoundError(f'No model weights in {checkpoint}')
                for name in weights[2:]:
                    index = checkpoint / name
                    if index.is_file():
                        shards = json.loads(index.read_text())['weight_map'].values()
                        if any(not (checkpoint / shard).is_file() for shard in shards):
                            raise FileNotFoundError(f'Missing model shards in {checkpoint}')
                print(f'Resuming MSLoc-PR from {checkpoint}')
            elif output_dir.exists() and any(output_dir.iterdir()) and not training_args.clean:
                raise FileExistsError(f'{output_dir} is nonempty; use --resume_from_checkpoint or --clean')
            output_dir.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({'stage': 'MSLoc-PR'}), encoding='utf-8')
            result[0] = resume
        except Exception as error:
            result[1] = f'{type(error).__name__}: {error}'
    if torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(result, src=0)
    if result[1] is not None:
        raise RuntimeError(result[1])
    return result[0]


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if model_args.mm_projector_type != 'ref_projector' or not model_args.closs:
        raise ValueError('Full MSLoc-PR training requires ref_projector and closs=True')
    if (data_args.bnd_frames, data_args.seg_frames, model_args.num_frames) != (16, 8, 40):
        raise ValueError('The existing RefProjector requires 16 + 8 + 16 = 40 frames')
    if not 0 < data_args.bnd_ratio < 0.5:
        raise ValueError('bnd_ratio must be between 0 and 0.5')
    if model_args.pretrain_mm_mlp_adapter is not None:
        raise ValueError('Initialize adapters from model_name_or_path for this training entry point')
    if training_args.report_to:
        raise ValueError('Use --report_to none; external experiment logging is disabled')
    if training_args.remove_unused_columns or not training_args.prediction_loss_only:
        raise ValueError('Keep remove_unused_columns=False and prediction_loss_only=True')
    if training_args.label_smoothing_factor != 0:
        raise ValueError('Joint text/time/score/AA supervision requires label_smoothing_factor=0')
    for path in (model_args.model_name_or_path, model_args.vision_tower):
        if not Path(path).is_dir():
            raise FileNotFoundError(path)
    if not Path(model_args.class_feature_path).is_file():
        raise FileNotFoundError(model_args.class_feature_path)
    if not torch.cuda.is_available():
        raise RuntimeError('MSLoc-PR training requires CUDA')
    if training_args.n_gpu > 1:
        raise ValueError('Use torchrun for multiple GPUs, or CUDA_VISIBLE_DEVICES=0 for one GPU')
    if training_args.deepspeed or training_args.fsdp:
        raise ValueError('This entry point supports single-GPU training and torchrun DDP')
    if training_args.save_only_model:
        raise ValueError('Keep save_only_model=False to preserve resumable training state')

    set_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Prepare the output directory and resume state.
    resume = prepare_training_output(model_args, data_args, training_args)

    # Load the model and initialize the training modules.
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, local_files_only=True)
    if config.model_type != 'trace_mistral' or config.hidden_size != 4096:
        raise ValueError('Use the Mistral-based TRACE model supported by this repository')
    config.mm_vision_tower = model_args.vision_tower
    config.mm_vision_select_layer = model_args.mm_vision_select_layer
    config.mm_vision_select_feature = model_args.mm_vision_select_feature
    config.closs = True
    config.downsample_num = model_args.downsample_num
    config._attn_implementation = 'eager'
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32
    model = TraceMistralForCausalLM.from_pretrained(
        model_args.model_name_or_path, config=config, local_files_only=True, torch_dtype=dtype,
    )
    initialize_ref_projector(model, model_args)
    model.get_model().initialize_time_modules(model_args)
    model.get_model().initialize_score_modules(model_args)
    model.get_model().initialize_closs_modules(model_args.class_feature_path, config.hidden_size)
    model.get_model().closs_head.register_forward_hook(check_closs_logits)
    class_to_idx = model.get_model().class_to_idx
    if 'Normal' not in class_to_idx:
        raise ValueError('The anomaly class bank must include Normal')
    model.config.closs_class_names = model.get_model().class_names
    model.config.closs_class_to_idx = class_to_idx
    model.config.use_cache = False
    model.config.image_aspect_ratio = 'pad'
    model.config.mm_use_im_start_end = False
    model.config.mm_use_im_patch_token = False
    model.config.bnd_ratio = data_args.bnd_ratio
    model.config.bnd_frames = 16
    model.config.seg_frames = 8
    # Save the full model, including the anomaly classification head.
    model.config.tune_mm_mlp_adapter = False

    # Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, local_files_only=True, use_fast=True,
        model_max_length=training_args.model_max_length, padding_side='right',
    )
    if tokenizer.unk_token is None:
        raise ValueError('Expected the TRACE/Mistral tokenizer with an unknown token')
    tokenizer.pad_token = tokenizer.unk_token
    model.config.tokenizer_padding_side = 'right'
    model.config.tokenizer_model_max_length = training_args.model_max_length
    configure_trainable_parameters(model, model_args)
    model.to(dtype=dtype)
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # Prepare training and validation data.
    processor = model.get_vision_tower().image_processor
    dataset_kwargs = dict(
        proposal_path=data_args.proposal_path, data_folder=data_args.data_folder,
        tokenizer=tokenizer, processor=processor, class_to_idx=class_to_idx,
        bnd_ratio=data_args.bnd_ratio, video_dtype=dtype,
        show_progress=training_args.process_index == 0,
    )
    train_dataset = ProposalDataset(data_args.data_path, max_videos=training_args.train_videos, **dataset_kwargs)
    eval_dataset = ProposalDataset(data_args.eval_data_path, max_videos=training_args.val_videos, **dataset_kwargs)
    if train_dataset.all_video_ids & eval_dataset.all_video_ids:
        raise ValueError('Training and validation videos overlap')
    if training_args.process_index == 0:
        effective_batch = (training_args.per_device_train_batch_size * training_args.world_size
                           * training_args.gradient_accumulation_steps)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'MSLoc-PR: DAM + EAM + L_AA, 40 frames, boundary ratio={data_args.bnd_ratio}')
        print(f'Epochs={training_args.num_train_epochs}, effective batch={effective_batch}, '
              f'lr={training_args.learning_rate}, trainable parameters={trainable:,}')
    # Run training.
    trainer = TraceTrainer(
        model=model, tokenizer=tokenizer, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=DataCollatorForSupervisedDataset(tokenizer),
    )
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_state()
    model.config.use_cache = True
    trainer.save_model(training_args.output_dir)


if __name__ == '__main__':
    train()
