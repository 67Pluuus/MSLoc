#    Copyright 2024 Alibaba DAMO Academy
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import re

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.regnet import RegStage
from timm.models.layers import LayerNorm, LayerNorm2d
from transformers import TRANSFORMERS_CACHE


def parse_snapshot_folder(repo_id, cache_dir=None, repo_type="model"):
    revision = "main"
    # 1. parse the downloaded cache folder
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    else:
        cache_dir = cache_dir
    object_id = repo_id.replace("/", "--")
    repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
    # 2. resolve refs (for instance to convert main to the associated commit sha)
    refs_dir = os.path.join(repo_cache, "refs")
    if os.path.isdir(refs_dir):
        revision_file = os.path.join(refs_dir, revision)
        if os.path.isfile(revision_file):
            with open(revision_file) as f:
                revision = f.read()
    # 3. acquire the snapshot folder
    folder = os.path.join(repo_cache, "snapshots", revision)

    return folder


def load_mm_projector(model_path, cache_dir=None, token=None):
    """Load mm_projector weights.

    - If `model_path` is a local directory, we will NOT call `snapshot_download`.
    - If `model_path` looks like a HF repo id, we resolve snapshot folder and download if needed.

    Expected file: `mm_projector.bin`.
    """
    folder = None

    # Local path case
    if os.path.isdir(model_path):
        folder = model_path

        # 1) Preferred: legacy dump
        projector_file = os.path.join(folder, "mm_projector.bin")
        if os.path.exists(projector_file):
            mm_projector_weights = torch.load(projector_file, map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            return mm_projector_weights

        # 2) Fallback: extract from sharded safetensors checkpoint
        #    (e.g. model-00003-of-00004.safetensors contains model.mm_projector.*)
        safetensors_files = sorted(
            [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.endswith(".safetensors") and "model-" in f
            ]
        )
        if len(safetensors_files) == 0:
            raise FileNotFoundError(
                f"Local model_path '{model_path}' does not contain mm_projector.bin or any model-*.safetensors shards."
            )

        try:
            from safetensors.torch import safe_open
        except Exception as e:
            raise RuntimeError(
                "Need safetensors to extract mm_projector weights from sharded checkpoint. "
                "Please install safetensors."
            ) from e

        extracted = {}
        prefix = "model.mm_projector."
        for shard in safetensors_files:
            with safe_open(shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k.startswith(prefix):
                        extracted[k] = f.get_tensor(k).to(torch.float16)
            if len(extracted) > 0:
                # usually mm_projector lives in a single shard; stop early
                break

        if len(extracted) == 0:
            raise FileNotFoundError(
                f"Could not find any '{prefix}*' keys in shards under '{model_path}'."
            )

        return extracted

    # HF repo id case
    else:
        folder = parse_snapshot_folder(model_path, cache_dir=cache_dir, repo_type="model")
        projector_file = os.path.join(folder, "mm_projector.bin")
        if not os.path.exists(projector_file):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_path, cache_dir=cache_dir, token=token)

        mm_projector_weights = torch.load(projector_file, map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        return mm_projector_weights


class IdentityMap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "linear":
        # NOTE: for both linear and mlp2x_gelu projector type, mean pooling is adopted to aggreate video features
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type == "stc_connector":
        return STCConnector(config)
    elif projector_type == "stp_connector":
        return STPConnector(config)
    elif projector_type == "stc_connector_v35":
        return STCConnectorV35(config)
    elif projector_type == "spatial_conv":
        return SpatialConv(config)
    elif projector_type == "spatial_pool":
        return SpatialPool(config)
    elif projector_type == "slot":
        return SlotPool(config)
    elif projector_type == "spatial_slot":
        return SpatialSlotPool(config)
    elif projector_type == 'spatial_time_slot':
        return SpatialTimeSlotPool(config)
    elif projector_type == 'ref_projector':
        return RefProjector(config)
    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class STCConnector(nn.Module):

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        """Temporal Convolutional Vision-Language Connector.
        
        Args:
            config: config object.
            downsample: (temporal, height, width) downsample rate.
            depth: depth of the spatial interaction blocks.
            mlp_depth: depth of the vision-language projector layers.
        """
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size = config.mm_hidden_size
        self.hidden_size = hidden_size = config.hidden_size
        self.output_hidden_size = output_hidden_size = config.hidden_size
        # TODO: make these as config arguments
        self.depth = depth
        self.mlp_depth = mlp_depth
        self.downsample = downsample
        self.downsample_num = config.downsample_num
        print(f'downsample num {self.downsample_num}')
        if depth != 0:
            self.s1 = RegStage(
                depth=depth,
                in_chs=encoder_hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s1 = nn.Identity()
        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=downsample,
                stride=downsample,
                padding=1,
                bias=True
            ),
            nn.SiLU()
        )
        if depth != 0:
            self.s2 = RegStage(
                depth=depth,
                in_chs=hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s2 = nn.Identity()
        self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)

    def forward(self, x, h=24, w=24):
        """Aggregate tokens on the temporal and spatial dimensions.
        Args:
            x: input tokens [b, t, h, w, d] / [b, t, l, d]
        Returns:
            aggregated tokens [b, l, d]
        """
        t = x.size(1)
        if x.ndim == 4:
            x = einops.rearrange(x, "b t (h w) d -> b d t h w", h=h, w=w)
        elif x.ndim == 5:
            x = einops.rearrange(x, "b t h w d -> b d t h w")

        x = einops.rearrange(x, "b d t h w -> (b t) d h w")

        # 1. the first stage of the adapter
        x = self.s1(x)
        x = einops.rearrange(x, "(b t) d h w -> b d t h w", t=t)
        # 2. downsampler
        x = self.sampler(x)
        new_t = x.size(2)
        # 3. the second stage of the adapter
        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        x = self.s2(x)
        x = einops.rearrange(x, "(b t) d h w -> b (t h w) d", t=new_t)
        x = self.readout(x)
        return x


class STPConnector(STCConnector):

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)
        self.sampler = nn.Sequential(nn.AvgPool3d(downsample), nn.SiLU())


class STCConnectorV35(STCConnector):

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)
        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=downsample,
                stride=downsample,
                padding=0,
                bias=True
            ),
            nn.SiLU())


class SpatialConv(STCConnector):

    def __init__(self, config, downsample=(1, 2, 2), depth=0, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)


class SpatialPool(STPConnector):

    def __init__(self, config, downsample=(1, 2, 2), depth=0, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)


# copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Mistral
# TODO @Arthur no longer copied from LLama after static cache
class SlotRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # IMPORTANT: in this codebase, x is often `[B, S, D]` or `[(B*T), N, D]`.
        # We guarantee returned cos/sin are ALWAYS on the same device as `x`.
        if seq_len > self.max_seq_len_cached:
            # build cache on inv_freq device to keep buffers consistent
            self._set_cos_sin_cache(seq_len=seq_len, device=self.inv_freq.device, dtype=x.dtype)

        cos = self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device)
        return cos, sin


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    half_dim = x.shape[-1] // 2
    if half_dim == 0:
        return torch.zeros_like(x)
    x1 = x[..., : half_dim]
    x2 = x[..., half_dim : 2*half_dim]  # Ensure we only take exactly half_dim elements
    result = torch.cat((-x2, x1), dim=-1)
    return result


# copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
# TODO @Arthur no longer copied from LLama after static cache
def apply_rotary_pos_emb(x, cos, sin, position_ids):
    """Applies Rotary Position Embedding to the last dimension.

    Notes:
      - This projector uses token-wise RoPE (like Llama/Mistral attention), but here `x` is typically
        shaped as `[(b*t), n, d]` or `[b, (t*n), d]`.
      - We only apply RoPE on the first `rope_dim` (an even number) of the last dimension.
    """
    # position_ids: [seq]
    position_ids = position_ids.to(device=x.device)

    # IMPORTANT: make sure cached cos/sin are on the same device BEFORE indexing
    cos = cos.to(device=x.device)
    sin = sin.to(device=x.device)

    # cos/sin: [seq_len, rope_dim]
    cos = cos.index_select(0, position_ids).to(dtype=x.dtype, device=x.device)
    sin = sin.index_select(0, position_ids).to(dtype=x.dtype, device=x.device)

    # Broadcast to x: [1, seq, rope_dim] when x is [B, seq, D]
    if x.dim() == 3 and cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    rope_dim = cos.shape[-1]
    if rope_dim == 0:
        return x

    x_rope = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]
    rotated = rotate_half(x_rope)
    x_rope = (x_rope * cos) + (rotated * sin)
    return torch.cat([x_rope, x_pass], dim=-1)


class SlotPool(nn.Module):

    def __init__(self, config, num_slots=1024):
        super().__init__()

        self.slots = nn.Parameter(torch.randn(config.mm_hidden_size, num_slots))
        self.ln_vision = LayerNorm(config.mm_hidden_size)
        self.readout = nn.Linear(config.mm_hidden_size, config.hidden_size, bias=False)
        # IMPORTANT: RoPE is defined on head_dim, not hidden_dim. Here we treat projector hidden as a single-head.
        # Use a smaller rope_dim to avoid extremely large inv_freq tables and potential numerical issues.
        self.rope_dim = getattr(config, "mm_rope_dim", 128)
        if self.rope_dim > config.mm_hidden_size:
            self.rope_dim = config.mm_hidden_size
        if self.rope_dim % 2 == 1:
            self.rope_dim -= 1
        self.rotary_emb = SlotRotaryEmbedding(self.rope_dim)
        self.num_slots = num_slots


    def forward(self, x):
        """Aggregate tokens on the temporal and spatial dimensions.
        Args:
            x: input tokens [b, t, h, w, d] / [b, t, l, d]
        Returns:
            aggregated tokens [b, l, d]
        """
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("ERROR: SlotPool input x contains NaN or Inf!")

        t = x.size(1)

        if x.ndim == 4:
            n = x.size(2)
            x = einops.rearrange(x, "b t n d -> b (t n) d")
        elif x.ndim == 5:
            n = x.size(2) * x.size(3)
            x = einops.rearrange(x, "b t h w d -> b (t h w) d") # b n d

        # Force LayerNorm to run in float32
        weight = self.ln_vision.weight.float()
        bias = self.ln_vision.bias.float() if self.ln_vision.bias is not None else None
        x = F.layer_norm(x.float(), self.ln_vision.normalized_shape, weight, bias).to(x.dtype)

        position_ids = torch.repeat_interleave(torch.tensor(range(t)), repeats=n).to(x.device)
        cos, sin = self.rotary_emb(x, seq_len=t)
        x = apply_rotary_pos_emb(x, cos, sin, position_ids)
        
        logits = torch.matmul(x, self.slots) # b n s
        attn = torch.softmax(logits.float().clamp(-50, 50), dim=1).to(logits.dtype)

        res = torch.matmul(x.permute(0,2,1), attn).permute(0, 2, 1) # b s d

        return self.readout(res)



class SpatialSlotPool(nn.Module):

    def __init__(self, config, num_slots=8):
        super().__init__()

        print(num_slots)

        self.slots = nn.Parameter(torch.randn(config.mm_hidden_size, num_slots))
        self.ln_vision = LayerNorm(config.mm_hidden_size)
        self.readout = nn.Linear(config.mm_hidden_size, config.hidden_size, bias=False)
        self.rope_dim = getattr(config, "mm_rope_dim", 128)
        if self.rope_dim > config.mm_hidden_size:
            self.rope_dim = config.mm_hidden_size
        if self.rope_dim % 2 == 1:
            self.rope_dim -= 1
        self.rotary_emb = SlotRotaryEmbedding(self.rope_dim)
        self.num_slots = num_slots


    def forward(self, x):
        """Aggregate tokens on the temporal and spatial dimensions.
        Args:
            x: input tokens [b, t, h, w, d] / [b, t, l, d]
        Returns:
            aggregated tokens [b, l, d]
        """
        
        t = x.size(1)

        if x.ndim == 4:
            n = x.size(2)
            x = einops.rearrange(x, "b t n d -> (b t) n d")
        elif x.ndim == 5:
            n = x.size(2) * x.size(3)
            x = einops.rearrange(x, "b t h w d -> (b t) (h w) d") # (b t) n d

        x = self.ln_vision(x)

        position_ids = torch.arange(
                n, dtype=torch.long, device=x.device
            )
        cos, sin = self.rotary_emb(x, seq_len=n)
        x = apply_rotary_pos_emb(x, cos, sin, position_ids)
        
        logits = torch.matmul(x, self.slots) # (b t) n s
        attn = torch.softmax(logits, dim=1)

        res = torch.matmul(x.permute(0,2,1), attn).permute(0, 2, 1) # (b t) s d
        # res = einops.rearrange(res, "(b t) s d -> b (t s) d", t=t) # (b t) n d
        # v5
        res = einops.rearrange(res, "(b t) s d -> b t s d", t=t) # (b t) n d

        return self.readout(res)

class SpatialTimeSlotPool(nn.Module):

    def __init__(self, config, num_spatial_slots=8, num_time_slots=1):
        super().__init__()


        self.spatial_slots = nn.Parameter(torch.randn(config.mm_hidden_size, num_spatial_slots))
        self.time_slots = nn.Parameter(torch.randn(config.mm_hidden_size, num_time_slots))
        self.ln_vision = LayerNorm(config.mm_hidden_size)
        self.readout = nn.Linear(config.mm_hidden_size, config.hidden_size, bias=False)
        self.rope_dim = getattr(config, "mm_rope_dim", 128)
        if self.rope_dim > config.mm_hidden_size:
            self.rope_dim = config.mm_hidden_size
        if self.rope_dim % 2 == 1:
            self.rope_dim -= 1
        self.rotary_emb = SlotRotaryEmbedding(self.rope_dim)
        self.num_spatial_slots = num_spatial_slots
        self.num_time_slots = num_time_slots

    
    def forward(self, x, image_dim=576):
        """Aggregate tokens on the temporal and spatial dimensions.
        Args:
            x: input tokens [b, t, h, w, d] / [b, t, l, d]
        Returns:
            aggregated tokens [b, l, d]
        """
        
        
        t = x.size(1)

        if x.ndim == 4:
            n = x.size(2)
            x = einops.rearrange(x, "b t n d -> (b t) n d")
        elif x.ndim == 5:
            n = x.size(2) * x.size(3)
            x = einops.rearrange(x, "b t h w d -> (b t) (h w) d") # (b t) n d

        # for image part
        image_x, time_x = torch.split(x, image_dim, dim=1)

        image_x = self.ln_vision(image_x)

        image_position_ids = torch.arange(
                image_dim, dtype=torch.long, device=image_x.device
            )
        image_cos, image_sin = self.rotary_emb(image_x, seq_len=image_dim)
        image_x = apply_rotary_pos_emb(image_x, image_cos, image_sin, image_position_ids)
        
        image_logits = torch.matmul(image_x, self.spatial_slots) # (b t) n s
        image_attn = torch.softmax(image_logits, dim=1)

        image_outputs = torch.matmul(image_x.permute(0,2,1), image_attn).permute(0, 2, 1) # (b t) s d
        image_outputs = einops.rearrange(image_outputs, "(b t) s d -> b t s d", t=t) # (b t) n d
        image_outputs = self.readout(image_outputs)

        # for time part
        if time_x.shape[1] == 0:
            time_outputs = torch.zeros((x.shape[0], 0, self.hidden_size), device=x.device, dtype=x.dtype)
            # Reshape to match expected output format b t s d
            time_outputs = einops.rearrange(time_outputs, "(b t) s d -> b t s d", t=t)
        else:
            time_position_ids = torch.arange(
                    n - image_dim, dtype=torch.long, device=time_x.device
                )
            time_cos, time_sin = self.rotary_emb(time_x, seq_len=n - image_dim)
            time_x = apply_rotary_pos_emb(time_x, time_cos, time_sin, time_position_ids)

            time_logits = torch.matmul(time_x, self.time_slots) # (b t) n s
            time_attn = torch.softmax(time_logits, dim=1)

            time_outputs = torch.matmul(time_x.permute(0,2,1), time_attn).permute(0, 2, 1) # (b t) s d
            time_outputs = einops.rearrange(time_outputs, "(b t) s d -> b t s d", t=t) # (b t) n d

        # for final outputs
        outputs = torch.cat([image_outputs, time_outputs], dim=2)


        return outputs


class DiffGuidedSpatialSlotPool(nn.Module):

    def __init__(self, config, num_slots=8):
        super().__init__()
        
        if config.mm_hidden_size <= 0:
            raise ValueError(f"Invalid mm_hidden_size: {config.mm_hidden_size}")

        self.slots = nn.Parameter(torch.randn(config.mm_hidden_size, num_slots))
        self.ln_vision = LayerNorm(config.mm_hidden_size)
        self.readout = nn.Linear(config.mm_hidden_size, config.hidden_size, bias=False)
        self.rope_dim = getattr(config, "mm_rope_dim", 128)
        if self.rope_dim > config.mm_hidden_size:
            self.rope_dim = config.mm_hidden_size
        if self.rope_dim % 2 == 1:
            self.rope_dim -= 1
        self.rotary_emb = SlotRotaryEmbedding(self.rope_dim)
        self.num_slots = num_slots
        
        self.scale = config.mm_hidden_size ** -0.5


    def forward(self, x):
        """
        x: [b, t, n, d] or [b, t, h, w, d]
        """
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("ERROR: DiffGuidedSpatialSlotPool input x contains NaN or Inf!")

        t = x.size(1)

        if x.ndim == 4:
            n = x.size(2)
            # x: [b, t, n, d]
        elif x.ndim == 5:
            n = x.size(2) * x.size(3)
            x = einops.rearrange(x, "b t h w d -> b t (h w) d") # b t n d

        # Flatten for processing
        x_flat = einops.rearrange(x, "b t n d -> (b t) n d")

        # Force LayerNorm to run in float32
        # This is critical for stability
        weight = self.ln_vision.weight.float()
        bias = self.ln_vision.bias.float() if self.ln_vision.bias is not None else None
        
        x_norm = F.layer_norm(x_flat.float(), self.ln_vision.normalized_shape, weight, bias).to(x.dtype)

        # Calculate Diff Score for extra tokens
        # Reshape back to [b, t, n, d]
        x_norm_reshaped = einops.rearrange(x_norm, "(b t) n d -> b t n d", t=t)
        
        # Calculate similarity/difference with previous frame (cyclic)
        # Using roll to get x[t-1] (with x[0] -> x[T-1])
        x_prev = torch.roll(x_norm_reshaped, shifts=1, dims=1)
        
        # Calculate difference score (Dot Product)
        # L2^2 = |x|^2 + |y|^2 - 2(x.y)
        # Since LayerNorm is applied, |x| and |y| are approximately constant.
        # Thus, larger L2 distance corresponds to smaller dot product.
        # We want diff_score to represent "magnitude of change", so we use negative dot product.
        # Shape: [b, t, n]
        dot_prod = (x_norm_reshaped * x_prev).sum(dim=-1)
        diff_score = -dot_prod
        
        # Token 1: Motion (High difference)
        # Softmax over spatial dimension N
        attn_motion = torch.softmax(diff_score, dim=-1) # [b, t, n]
        
        # Token 2: Static (Low difference -> High negative difference)
        # Softmax over spatial dimension N
        attn_static = torch.softmax(-diff_score, dim=-1) # [b, t, n]

        # Stack extra attentions and flatten: [b, t, n, 2] -> [(b t), n, 2]
        attn_extra = torch.stack([attn_motion, attn_static], dim=-1)
        attn_extra = einops.rearrange(attn_extra, "b t n k -> (b t) n k")

        position_ids = torch.arange(
                n, dtype=torch.long, device=x.device
            )
        cos, sin = self.rotary_emb(x_norm, seq_len=n)
        x_norm_rope = apply_rotary_pos_emb(x_norm, cos, sin, position_ids)
        
        # Attention logits
        # Base logits: x @ slots -> [(b t), n, s]
        logits = torch.matmul(x_norm_rope, self.slots) * self.scale
        
        attn_slots = torch.softmax(logits.float().clamp(-50, 50), dim=1).to(logits.dtype)

        # Concatenate all attentions: [(b t), n, s+2]
        attn_all = torch.cat([attn_slots, attn_extra], dim=-1)

        # Aggregate all tokens at once
        # [(b t), d, n] @ [(b t), n, s+2] -> [(b t), d, s+2]
        res_all = torch.matmul(x_norm_rope.transpose(1, 2), attn_all)
        
        # Transpose and reshape: [(b t), s+2, d] -> [b, t, s+2, d]
        res_all = res_all.transpose(1, 2)
        final_res = einops.rearrange(res_all, "(b t) s d -> b t s d", t=t)

        return self.readout(final_res)


class RefProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Boundary parts: 16 frames per side. Time dimension is preserved so
        # we get one compressed token per frame plus its time token.
        self.boundary_proj = DiffGuidedSpatialSlotPool(config, num_slots=8)

        # Event part: 8 frames -> a fixed pool of 32 tokens.
        self.event_proj = SlotPool(config, num_slots=32)

        self.config = config

    def forward(self, x, time_features=None):
        # x: [b, 40, h, w, d] (assuming 16+8+16=40 frames)
        t = x.size(1)
        
        # Split
        # Left: 16 frames
        x_left = x[:, :16]
        # Event: 8 frames
        x_event = x[:, 16:24]
        # Right: 16 frames
        x_right = x[:, 24:]
        
        # Process Boundary
        # boundary_proj expects [b, t, ...] returns [b, t, s, d]
        out_left = self.boundary_proj(x_left) # [b, 16, 8, d]
        out_right = self.boundary_proj(x_right) # [b, 16, 8, d]
        
        # Process Event
        # event_proj expects [b, t, ...] returns [b, s, d] (pooled over t)
        out_event = self.event_proj(x_event) # [b, 32, d]
        
        if time_features is not None:
            t_left = time_features[:, :16] # [b, 16, n_t, d]
            t_event = time_features[:, 16:24] # [b, 8, n_t, d]
            t_right = time_features[:, 24:] # [b, 16, n_t, d]
            
            # Merge Left
            # out_left: [b, 16, 8, d], t_left: [b, 16, n_t, d]
            out_left = torch.cat([out_left, t_left], dim=2) # [b, 16, 8+n_t, d]
            
            # Merge Right
            out_right = torch.cat([out_right, t_right], dim=2) # [b, 16, 8+n_t, d]
            
            # Merge Event
            # out_event: [b, 32, d]
            # Concatenate time tokens for event part.
            t_event_flat = einops.rearrange(t_event, 'b t n d -> b (t n) d')
            out_event = torch.cat([out_event, t_event_flat], dim=1) # [b, 32 + 8*n_t, d]

        out_left_flat = einops.rearrange(out_left, "b t s d -> b (t s) d")
        out_right_flat = einops.rearrange(out_right, "b t s d -> b (t s) d")
        
        # Order: Left, Right, Event
        res = torch.cat([out_left_flat, out_right_flat, out_event], dim=1)
        
        return res

