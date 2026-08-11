from transformers import XCLIPVisionModel
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from dataclasses import dataclass
from typing import Union

from timm.models.layers import DropPath
from .clip.pscan import pscan

@dataclass
class MambaConfig:
    d_model: 768
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16
    expand_factor: int = 2
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    drop_prob: float = 0.1

    bias: bool = False
    conv_bias: bool = True
    bimamba: bool = True

    pscan: bool = True

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model)
        self.drop_path = DropPath(drop_prob=config.drop_prob)

    def forward(self, x):


        output = self.drop_path(self.mixer(self.norm(x))) + x
        return output

    def step(self, x, cache):


        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache

class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                              kernel_size=config.d_conv, bias=config.conv_bias,
                              groups=config.d_inner,
                              padding=config.d_conv - 1)

        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(config.d_inner))

        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        self.bimamba = config.bimamba

        if self.bimamba:
            A_b = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
            self.A_b_log = nn.Parameter(torch.log(A_b))

            self.conv1d_b = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                              kernel_size=config.d_conv, bias=config.conv_bias,
                              groups=config.d_inner,
                              padding=config.d_conv - 1)

            self.x_proj_b = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
            self.dt_proj_b = nn.Linear(config.dt_rank, config.d_inner, bias=True)
            self.D_b = nn.Parameter(torch.ones(config.d_inner))

    def forward(self, x):


        _, L, _ = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)

        x = F.silu(x)
        y = self.ssm(x)

        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)

        return output

    def ssm(self, x):


        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.x_proj(x)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        if self.bimamba:
            x_b = x.flip([-1])
            A_b = -torch.exp(self.A_b_log.float())
            D_b = self.D_b.float()
            deltaBC_b = self.x_proj_b(x_b)
            delta_b, B_b, C_b = torch.split(deltaBC_b, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
            delta_b = F.softplus(self.dt_proj_b(delta_b))
            if self.config.pscan:
                y_b = self.selective_scan(x_b, delta_b, A_b, B_b, C_b, D_b)
            else:
                y_b = self.selective_scan_seq(x_b, delta_b, A_b, B_b, C_b, D_b)
            y_b = y_b.flip([-1])
            y = y + y_b
        return y

    def selective_scan(self, x, delta, A, B, C, D):


        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

        BX = deltaB * (x.unsqueeze(-1))

        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):


        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

        BX = deltaB * (x.unsqueeze(-1))

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y + D * x

        return y

    def step(self, x, cache):
        """Advance one token with the state and convolution-history cache."""
        h, inputs = cache

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=1)

        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1]

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)

        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)
        cache = (h, inputs)

        return output, cache

    def ssm_step(self, x, h):


        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.x_proj(x)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)

        BX = deltaB * (x.unsqueeze(-1))

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)

        h = deltaA * h + BX

        y = (h @ C.unsqueeze(-1)).squeeze(2)

        y = y + D * x

        return y, h.squeeze(1)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


def create_reorder_index(N, device):
    new_order = []
    for col in range(N):
        if col % 2 == 0:
            new_order.extend(range(col, N*N, N))
        else:
            new_order.extend(range(col + N*(N-1), col-1, -N))
    return torch.tensor(new_order, device=device)

def reorder_data(data, N):
    assert isinstance(data, torch.Tensor), "data should be a torch.Tensor"
    device = data.device
    new_order = create_reorder_index(N, device)
    B, t, _, _ = data.shape
    index = new_order.repeat(B, t, 1).unsqueeze(-1)
    reordered_data = torch.gather(data, 2, index.expand_as(data))
    return reordered_data

class XCLIP_DeMamba(nn.Module):
    def __init__(
        self, channel_size=768, class_num=1
    ):
        super(XCLIP_DeMamba, self).__init__()
        _xclip_path = os.environ.get(
            "XCLIP_WEIGHTS_PATH",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                         "MSLoc_assets", "DeMamba", "pretrained_weights", "xclip-base-patch16"),
        )
        self.encoder = XCLIPVisionModel.from_pretrained(_xclip_path)
        blocks = []
        channel = 768
        self.fusing_ratios = 1
        self.patch_nums = (14//self.fusing_ratios)**2
        self.mamba_configs = MambaConfig(d_model=channel)
        self.mamba = ResidualBlock(config = self.mamba_configs)
        self.fc1 = nn.Linear((self.patch_nums+1)*channel, class_num)
        self.fc_norm = nn.LayerNorm(self.patch_nums*channel)
        self.fc_norm2 = nn.LayerNorm(768)
        self.initialize_weights(self.fc1)
        self.dropout = nn.Dropout(p=0.0)

    def initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        b, t, _, h, w = x.shape
        images = x.view(b * t, 3, h, w)
        outputs = self.encoder(images, output_hidden_states=True)
        sequence_output = outputs['last_hidden_state'][:,1:,:]
        _, _, c = sequence_output.shape

        global_feat = outputs['pooler_output'].reshape(b, t, -1)
        global_feat = global_feat.mean(1)
        global_feat = self.fc_norm2(global_feat)

        sequence_output = sequence_output.view(b, t, -1, c)
        _, _, f_w, _ = sequence_output.shape
        f_h, f_w = int(math.sqrt(f_w)), int(math.sqrt(f_w))

        s = f_h//self.fusing_ratios
        sequence_output = sequence_output.view(b, t, self.fusing_ratios, s, self.fusing_ratios, s, c)
        x = sequence_output.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(b*s*s, t, -1, c)
        b_l = b*s*s

        x = reorder_data(x, self.fusing_ratios)
        x = x.permute(0, 2, 1, 3).contiguous().view(b_l, -1, c)
        res = self.mamba(x)

        video_level_features = res.mean(1)
        video_level_features = video_level_features.view(b, -1)
        video_level_features = self.fc_norm(video_level_features)
        video_level_features = torch.cat((global_feat, video_level_features), dim=1)

        pred = self.fc1(video_level_features)
        pred = self.dropout(pred)

        return pred
