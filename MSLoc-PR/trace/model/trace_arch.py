# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
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
from abc import ABC, abstractmethod

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_encoder.builder import build_vision_tower, build_time_tower, build_score_tower, build_sync_tower


# ==================== CLoss Projector ====================
class ClossProjector(nn.Module):
    """
    将 LLM 的 Hidden State 投影到 Sentence-BERT 的特征空间
    用于对比学习或分类损失计算
    """
    def __init__(self, llm_hidden_size=4096, text_feat_size=384, hidden_size=1024):
        super().__init__()
        # 添加 LayerNorm 以稳定输入分布
        self.layer_norm = nn.LayerNorm(llm_hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(llm_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, text_feat_size),
        )
        
        # 使用保守的初始化策略
        self.apply(self._init_weights)
        
        # 特别处理最后一层，使其输出接近 0
        nn.init.normal_(self.mlp[-1].weight, std=1e-4)
        if self.mlp[-1].bias is not None:
            nn.init.zeros_(self.mlp[-1].bias)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, hidden_states):
        """hidden_states: (B, 3, H) -> (B, 3, text_feat_size)"""
        hidden_states = self.layer_norm(hidden_states)
        return self.mlp(hidden_states)


class ClassFeatureBank(nn.Module):
    """
    预计算并存储所有类别的 Sentence-BERT 特征（冻结）
    """
    def __init__(self, class_features: torch.Tensor, class_names: list = None):
        super().__init__()
        # class_features: (num_classes, feat_dim)
        self.register_buffer('class_features', class_features)
        self.class_names = class_names or []
        self.num_classes = class_features.shape[0]
        self.feat_dim = class_features.shape[1]
    
    def get_normalized_features(self):
        return self.class_features


# 默认的预提取类别特征文件路径
import os as _os
DEFAULT_CLASS_FEATURE_PATH = _os.environ.get(
    'CLASS_FEATURE_PATH',
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
                  '..', '..', 'MSLoc_assets', 'Trace', 'class_features_bge.pt')
)


def load_class_feature_bank_from_file(feature_file_path: str):
    """
    从预先保存的 .pt 文件加载类别特征
    
    【重要】这是推荐的方式！
    特征文件由 extract_class_features.py 脚本使用 bge-large-en-v1.5 模型预先生成。
    训练时直接加载特征，避免与 DeepSpeed/FSDP 分布式训练框架冲突。
    
    Args:
        feature_file_path: 预计算特征文件路径 (.pt)
    
    Returns:
        ClassFeatureBank: 包含预计算特征的模块
        feat_dim: 特征维度
    """
    import os
    
    # ========== 严格校验文件路径 ==========
    if not os.path.isfile(feature_file_path):
        raise RuntimeError(
            f"CRITICAL ERROR: 类别特征文件不存在！\n"
            f"要求的路径: {feature_file_path}\n"
            f"请先运行 extract_class_features.py 脚本生成特征文件！\n"
            f"\n"
            f"使用方法:\n"
            f"  python trace/scripts/extract_class_features.py \\\n"
            f"      --bge_model_path /path/to/bge-large-en-v1.5 \\\n"
            f"      --output_path {feature_file_path}"
        )
    
    print(f"========== Loading Pre-computed Class Features ==========")
    print(f"  Feature file: {feature_file_path}")
    
    # ========== 加载特征文件 ==========
    try:
        loaded_data = torch.load(feature_file_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(
            f"CRITICAL ERROR: 无法加载类别特征文件！\n"
            f"路径: {feature_file_path}\n"
            f"原始错误: {e}\n"
            f"请检查文件是否损坏，或重新生成特征文件！"
        )
    
    # ========== 校验文件内容 ==========
    required_keys = ['class_features', 'class_names', 'feat_dim', 'model_name']
    for key in required_keys:
        if key not in loaded_data:
            raise RuntimeError(
                f"CRITICAL ERROR: 特征文件格式不正确！\n"
                f"缺少必要的键: {key}\n"
                f"请使用 extract_class_features.py 重新生成特征文件！"
            )
    
    # 验证是 bge-large-en-v1.5 模型生成的特征
    model_name = loaded_data.get('model_name', '')
    if 'bge-large-en-v1.5' not in model_name.lower().replace('_', '-').replace(' ', '-'):
        raise RuntimeError(
            f"CRITICAL ERROR: 特征文件不是由 bge-large-en-v1.5 模型生成！\n"
            f"当前模型: {model_name}\n"
            f"请使用 bge-large-en-v1.5 模型重新生成特征文件！"
        )
    
    class_features = loaded_data['class_features']  # (num_classes, feat_dim)
    class_names = loaded_data['class_names']
    feat_dim = loaded_data['feat_dim']
    num_classes = loaded_data.get('num_classes', len(class_names))
    pooling_method = loaded_data.get('pooling_method', 'CLS')
    normalized = loaded_data.get('normalized', True)
    
    # 验证特征维度
    if class_features.shape[-1] != feat_dim:
        raise RuntimeError(
            f"CRITICAL ERROR: 特征维度不匹配！\n"
            f"文件记录: {feat_dim}, 实际: {class_features.shape[-1]}"
        )
    
    # 验证是 bge-large-en-v1.5 的 1024 维特征
    if feat_dim != 1024:
        print(f"  WARNING: Expected feature dim 1024 for bge-large-en-v1.5, got {feat_dim}")
    
    print(f"========== Class Features Loaded Successfully ==========")
    print(f"  Model: {model_name}")
    print(f"  Pooling: {pooling_method}")
    print(f"  Normalized: {normalized}")
    print(f"  Num classes: {num_classes}")
    print(f"  Feature dim: {feat_dim}")
    print(f"  Feature shape: {class_features.shape}")
    print(f"  Class names: {class_names[:3]}... (showing first 3)")
    
    return ClassFeatureBank(class_features.float(), class_names), feat_dim

from .multimodal_projector.builder import build_vision_projector
from ..mm_utils import get_anyres_image_grid_shape
from ..constants import NUM_FRAMES, IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,DEFAULT_MMODAL_PATCH_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN, MMODAL_TOKEN_INDEX, MMODAL_INDEX_TOKEN


class TraceMetaModel:

    def __init__(self, config):
        super(TraceMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=False)
            self.mm_projector = build_vision_projector(config)
        
        self.time_tokenizer, self.time_tower = build_time_tower(None, None, 4096)
        self.score_tokenizer, self.score_tower = build_score_tower(None, None, 4096)
        self.sync_tower = build_sync_tower(4096)

        if getattr(config, 'closs', False):
            self.closs_tokens = nn.Parameter(torch.randn(3, config.hidden_size) * 1e-4)
            # ClossProjector 和 ClassFeatureBank 将在 initialize_closs_modules 中初始化
            self.closs_projector = None
            self.class_feature_bank = None
            # Learnable logit scale for CLoss (init to log(10) ≈ 2.3)
            self.closs_logit_scale = nn.Parameter(torch.ones([]) * 2.3026)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_time_tower(self):
        time_tower = getattr(self, 'time_tower', None)
        return time_tower

    def get_score_tower(self):
        score_tower = getattr(self, 'score_tower', None)
        return score_tower

    def get_sync_tower(self):
        sync_tower = getattr(self, 'sync_tower', None)
        return sync_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        downsample_num = model_args.downsample_num

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.downsample_num = downsample_num
        
        # Explicitly set num_frames in config to match data_args
        if hasattr(model_args, 'num_frames') and model_args.num_frames is not None:
            self.config.num_frames = model_args.num_frames
        elif hasattr(model_args, 'bnd_frames') and getattr(model_args, 'train_mode', 'loc') == 'ref2':
            # Auto-calculate num_frames for ref2 mode if not set
            bnd_frames = getattr(model_args, 'bnd_frames', 16)
            seg_frames = getattr(model_args, 'seg_frames', 8)
            self.config.num_frames = bnd_frames * 2 + seg_frames
            # print(f"Auto-set num_frames to {self.config.num_frames} for ref2 mode")

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # Check if we need to rebuild mm_projector
            # Specifically for ref_projector transition
            if self.config.mm_projector_type == 'ref_projector' and self.mm_projector.__class__.__name__ != 'RefProjector':
                # print(f"Rebuilding mm_projector: Current type {self.mm_projector.__class__.__name__} does not match requested {self.config.mm_projector_type}. Rebuilding...")
                self.mm_projector = build_vision_projector(self.config)
            else:
                # In case it is frozen by LoRA
                for p in self.mm_projector.parameters():
                    p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            if os.path.exists(pretrain_mm_mlp_adapter):
                is_local = True
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            else:
                # Support loading projector weights from remote HuggingFace model hub
                is_local = False
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.replace('mm_projector.bin', '')
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.strip('/').strip('\\').strip()
                mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

            #self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            # set strict=False to avoid missing key error regarding bert.embeddings.position_ids
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)


    ##################################################################################

    def initialize_time_modules(self, model_args, pretrained_tokenizer=None, pretrained_embedding_weights=None, dim=4096):

        # pretrained_embedding = self.embed_tokens.weight.data
        # print(pretrained_embedding)

        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        if self.get_time_tower() is None:
            # print('initialize time tower from scarch')
            self.time_tokenizer, self.time_tower = build_time_tower(pretrained_tokenizer, pretrained_embedding_weights=pretrained_embedding_weights, dim=dim)
        else:
            # print('use existing time tower')
            return

        if pretrain_mm_mlp_adapter is not None:
            # print('load time tower weights from checkpoint')
            if os.path.exists(pretrain_mm_mlp_adapter):
                is_local = True
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            else:
                # Support loading projector weights from remote HuggingFace model hub
                is_local = False
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.replace('mm_projector.bin', '')
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.strip('/').strip('\\').strip()
                mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

            #self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            # set strict=False to avoid missing key error regarding bert.embeddings.position_ids
            self.time_tower.load_state_dict(get_w(mm_projector_weights, 'time_tower'), strict=True)
            
            # Check for NaN in loaded weights
            for name, param in self.time_tower.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"CRITICAL ERROR: TimeTower parameter {name} contains NaN or Inf after loading!")


    def initialize_score_modules(self, model_args, tokenizer=None, pretrained_embedding_weights=None, dim=4096):

        # pretrained_embedding = self.embed_tokens.weight.data
        # print(pretrained_embedding_weights)

        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        if self.get_score_tower() is None:
            # print('initialize score tower from scarch')
            self.score_tokenizer, self.score_tower = build_score_tower(tokenizer, pretrained_embedding_weights=pretrained_embedding_weights, dim=dim)
        else:
            # print('use existing score tower')
            return

        if pretrain_mm_mlp_adapter is not None:
            # print('load score tower weights from checkpoint')
            if os.path.exists(pretrain_mm_mlp_adapter):
                is_local = True
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            else:
                # Support loading projector weights from remote HuggingFace model hub
                is_local = False
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.replace('mm_projector.bin', '')
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.strip('/').strip('\\').strip()
                mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

            #self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            # set strict=False to avoid missing key error regarding bert.embeddings.position_ids
            self.score_tower.load_state_dict(get_w(mm_projector_weights, 'score_tower'), strict=True)
            
            # Check for NaN in loaded weights
            for name, param in self.score_tower.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"CRITICAL ERROR: ScoreTower parameter {name} contains NaN or Inf after loading!")

    def initialize_closs_modules(self, class_feature_path: str = None, hidden_size: int = 4096):
        """
        初始化 CLoss 模块：ClossProjector 和 ClassFeatureBank
        
        【重要】从预先计算的特征文件加载类别特征，避免与分布式训练框架冲突！
        特征文件由 extract_class_features.py 脚本使用 bge-large-en-v1.5 模型预先生成。
        
        Args:
            class_feature_path: 预计算特征文件路径 (.pt)，默认使用 DEFAULT_CLASS_FEATURE_PATH
            hidden_size: LLM 隐藏层维度
        """
        if not getattr(self.config, 'closs', False):
            print("CLoss is disabled in config, skipping initialize_closs_modules")
            return
        
        # 确定特征文件路径
        import os
        if class_feature_path is not None and os.path.isfile(class_feature_path):
            feature_path = class_feature_path
        elif os.path.isfile(DEFAULT_CLASS_FEATURE_PATH):
            feature_path = DEFAULT_CLASS_FEATURE_PATH
        else:
            raise RuntimeError(
                f"CRITICAL ERROR: 类别特征文件不存在！\n"
                f"尝试的路径:\n"
                f"  - 参数传入: {class_feature_path}\n"
                f"  - 默认路径: {DEFAULT_CLASS_FEATURE_PATH}\n"
                f"\n"
                f"请先运行特征提取脚本生成特征文件:\n"
                f"  python trace/scripts/extract_class_features.py \\\n"
                f"      --bge_model_path /path/to/bge-large-en-v1.5 \\\n"
                f"      --output_path /path/to/class_features_bge.pt"
            )
        
        # 从预计算特征文件加载（避免与 DeepSpeed/FSDP 冲突）
        print(f"\n{'='*60}")
        print(f"Initializing CLoss modules from pre-computed features")
        print(f"{'='*60}")
        
        self.class_feature_bank, text_feat_dim = load_class_feature_bank_from_file(feature_path)
        
        # 获取类别信息
        class_names = self.class_feature_bank.class_names
        
        # 构建 ClossProjector (保留以防万一，但主要使用 closs_head)
        self.closs_projector = ClossProjector(
            llm_hidden_size=hidden_size,
            text_feat_size=text_feat_dim,
            hidden_size=1024
        )
        
        # 新增：初始化分类头 (Linear Classifier)
        # 直接将 hidden_states 映射到类别 logits
        self.closs_head = nn.Linear(hidden_size, len(class_names), bias=False)
        nn.init.normal_(self.closs_head.weight, std=0.02)
        
        # 保存类别信息
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        print(f"\n{'='*60}")
        print(f"CLoss modules initialized successfully!")
        print(f"  - ClossProjector: {hidden_size} -> 1024 -> {text_feat_dim}")
        print(f"  - ClossHead (Classifier): {hidden_size} -> {len(class_names)} classes")
        print(f"  - ClassFeatureBank: {len(class_names)} classes x {text_feat_dim} dim")
        print(f"  - Feature source: pre-computed file (bge-large-en-v1.5)")
        print(f"{'='*60}\n")

    def get_closs_projector(self):
        return getattr(self, 'closs_projector', None)
    
    def get_closs_head(self):
        return getattr(self, 'closs_head', None)
    
    def get_class_feature_bank(self):
        return getattr(self, 'class_feature_bank', None)

    ##################################################################################





class TraceMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def num_frames(self):
        if hasattr(self.config, 'num_frames'):
            return self.config.num_frames
        else:
            return NUM_FRAMES

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_time_tower(self):
        return self.get_model().get_time_tower()

    def get_score_tower(self):
        return self.get_model().get_score_tower()

    def get_sync_tower(self):
        return self.get_model().get_sync_tower()

    def encode_images_or_videos(self, images_or_videos, modalities, video_timestamps, seperate_time_feature=True):
        num_frames = self.config.num_frames if hasattr(self.config, 'num_frames') else NUM_FRAMES

        videos = [x.unsqueeze(0).expand(num_frames, -1, -1, -1) if modal == 'image' else x for x, modal in zip(images_or_videos, modalities)]
        videos = torch.stack(videos, dim=0)


        assert len(videos.size()) == 5
        batch_size = videos.size(0)

        frames = einops.rearrange(videos, 'b t c h w -> (b t) c h w')
        frames_features = self.get_model().get_vision_tower()(frames)
        
        if torch.isnan(frames_features).any() or torch.isinf(frames_features).any():
            print("ERROR: frames_features contains NaN or Inf!")
            print(f"Min: {frames_features.min()}, Max: {frames_features.max()}, Mean: {frames_features.mean()}")
        
        frames_features = einops.rearrange(frames_features, '(b t) n h -> b t n h', b = batch_size)
        hw = int(frames_features.shape[2] ** 0.5)
        
        if hw == 0:
            print("ERROR: hw is 0, avoiding division by zero")
            hw = 1 # Prevent crash to see what happens

        # 提前计算 time_features
        time_features = None
        if video_timestamps is not None: # b t
            video_time_tokens = self.encode_time(video_timestamps) # b t n_t
            
            # Optimize: Batch processing instead of double loop
            all_tokens_list = []
            for batch_idx in range(batch_size):
                for f_idx in range(len(video_time_tokens[batch_idx])):
                    # Remove <sync> token (last one) as per original code: [:-1]
                    all_tokens_list.append(video_time_tokens[batch_idx][f_idx][:-1])
            
            if len(all_tokens_list) > 0:
                # Stack all tokens: [b*t, n_t]
                # Assuming all time tokens have same length (which they should for single timestamp per frame)
                all_tokens_tensor = torch.stack(all_tokens_list).to(frames_features.device)
                
                # Batch embedding lookup
                all_features = self.get_time_tower()(all_tokens_tensor) # [b*t, n_t, h]
                
                # Reshape back to [b, t, n_t, h]
                # Note: video_time_tokens[batch_idx] length might vary if not padded, but here it comes from video_timestamps [b, t]
                # so we assume t is constant = num_frames
                t = len(video_time_tokens[0])
                time_features = einops.rearrange(all_features, '(b t) n h -> b t n h', b=batch_size, t=t)
            
            if torch.isnan(time_features).any() or torch.isinf(time_features).any():
                print("ERROR: time_features contains NaN or Inf!")

        # v5
        if seperate_time_feature:
            if self.config.mm_projector_type == 'ref_projector':
                # print("DEBUG: Calling temporal_aggregator for ref_projector...")
                frames_features = self.temporal_aggregator(frames_features, hw, int(frames_features.shape[2] / hw), time_features=time_features)
                # print("DEBUG: temporal_aggregator returned.")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                # print("DEBUG: CUDA synchronized after temporal_aggregator.")
                
                if torch.isnan(frames_features).any() or torch.isinf(frames_features).any():
                    print("ERROR: frames_features after temporal_aggregator contains NaN or Inf!")
                return frames_features
            else:
                frames_features = self.temporal_aggregator(frames_features, hw, int(frames_features.shape[2] / hw)) # b t s d
                if torch.isnan(frames_features).any() or torch.isinf(frames_features).any():
                    print("ERROR: frames_features after temporal_aggregator contains NaN or Inf!")
        ##################################################################################
        if time_features is not None:
            if not seperate_time_feature:
                time_features = time_features.view(time_features.shape[0], time_features.shape[1], -1, frames_features.shape[-1]) # remove it if v5

                frames_features = torch.cat([frames_features, time_features], dim=2) # b t (s + n_t) h
                frames_features = self.temporal_aggregator(frames_features, hw, int(frames_features.shape[2] / hw))
            else:
                frames_features = torch.cat([frames_features, time_features], dim=2) # b t (s + n_t) h
                # print(frames_features.shape, time_features.shape)
            
            if torch.isnan(frames_features).any() or torch.isinf(frames_features).any():
                print("ERROR: frames_features after merging time features contains NaN or Inf!")

            frames_features =  einops.rearrange(frames_features, 'b t n h -> b (t n) h')

            
            # v5
            # frames_features =  einops.rearrange(frames_features, 'b t n h -> b (t n) h')
            ##################################################################################
            

        return frames_features


    ##################################################################################

    def encode_time(self, times): 

        ## [
        #   [[t_1, t_2], [t_3, t_4], [t_5, t_6], ... ],
        #   [[t_1], [t_2], [t_3]],
        #   [[t_1, t_2]] 
        #  ]

        # if not self.get_model().get_time_tower().initialized:
        #     self.get_model().get_time_tower().update_weights(self.get_model().embed_tokens.weight)

        time_tokens = []
        for batch_times in times: # for each batch
            batch_time_tokens = [self.get_model().get_time_tower().encode(t) for t in batch_times]
            assert all([batch_time_token.shape == batch_time_tokens[0].shape for batch_time_token in batch_time_tokens]), f'{batch_times} {[batch_time_token.shape for batch_time_token in batch_time_tokens]}'
            time_tokens.append(batch_time_tokens)

        return time_tokens # [[event1-tokens, event2-tokens], ..., [event1-tokens, event2-tokens]]

    def encode_score(self, scores): 

        ## [
        #   [[s1], [s2], [s3]],
        #   [[], [], []],
        #   [[s1], [s2]]
        #  ]

        # if not self.get_model().get_score_tower().initialized:
        #     self.get_model().get_score_tower().update_weights(self.get_model().embed_tokens.weight)

        score_tokens = []
        for batch_scores in scores: # for each batch
            batch_score_tokens = [self.get_model().get_score_tower().encode(s) for s in batch_scores]
            score_tokens.append(batch_score_tokens)

        return score_tokens # [[event1-tokens, event2-tokens], ..., [event1-tokens, event2-tokens]]


    ##################################################################################

    def temporal_aggregator(self, frames_features, h, w, time_features=None):
        """Temporal aggregation of frame features.
        Args:
            frames_features (torch.Tensor): Frame features with shape (b, t, n, h).
        Returns:
            torch.Tensor: Video features with shape (b, n, h).
        """
        # TODO: improve the merging method.
        
        # *********** mean pooling *************
        if self.config.mm_projector_type == "mlp2x_gelu" or self.config.mm_projector_type == "linear":
            video_features = self.get_model().mm_projector(frames_features.mean(1))
        # *********** spatial convolution *************
        elif self.config.mm_projector_type == "spatial_conv":
            video_features = self.get_model().mm_projector(frames_features)
        # *********** spatial pooling *************
        elif self.config.mm_projector_type == "spatial_pool":
            video_features = self.get_model().mm_projector(frames_features)
        # *********** time  ************
        elif "tc_connector" in self.config.mm_projector_type or "tp_connector" in self.config.mm_projector_type:
            video_features = self.get_model().mm_projector(frames_features, h, w)
        elif "spatial_time_slot" in self.config.mm_projector_type:
            video_features = self.get_model().mm_projector(frames_features, h * h)
        elif "slot" in self.config.mm_projector_type:
            video_features = self.get_model().mm_projector(frames_features)
        elif "slot" in self.config.mm_projector_type:
            video_features = self.get_model().mm_projector(frames_features)
        elif self.config.mm_projector_type == 'ref_projector':
            video_features = self.get_model().mm_projector(frames_features, time_features=time_features)
        else:
            raise Exception(f"Unsupported projector type {self.config.mm_projector_type}!!!")
        return video_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, X_modalities, times, scores, video_timestamps=None
    ):
        vision_tower = self.get_vision_tower()
        # NOTE: text-only situation
        if vision_tower is None or X_modalities is None or input_ids.shape[1] == 1:
            # if past_key_values is not None and vision_tower is not None and Xs is not None and input_ids.shape[1] == 1:
            #    attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            new_input_embeds = []
            for batch_idx in range(input_ids.shape[0]):
                cur_input_ids = input_ids[batch_idx]

                # embed text input ids
                cur_text_ids = cur_input_ids % self.vocab_size
                text_embeds = self.get_model().embed_tokens(cur_text_ids)
                # embed sync
                sync_positions = (cur_input_ids == self.vocab_size)
                sync_embeds = self.get_sync_tower()(cur_input_ids[sync_positions])
                # embed time input ids
                time_positions = cur_input_ids >= (self.vocab_size + 1) and cur_input_ids < (self.vocab_size + self.time_vocab_size + 1)
                cur_time_ids = cur_input_ids[time_positions] - self.vocab_size - 1
                time_embeds = self.get_time_tower()(cur_time_ids)
                # embed score input ids
                score_positions = cur_input_ids >= (self.vocab_size + self.time_vocab_size + 1)
                cur_score_ids = cur_input_ids[score_positions] - self.vocab_size - self.time_vocab_size - 1
                score_embeds = self.get_score_tower()(cur_score_ids)

                # combine all the things
                text_embeds[time_positions] = time_embeds
                text_embeds[score_positions] = score_embeds
                text_embeds[sync_positions] = sync_embeds

                new_input_embeds.append(text_embeds)
            new_input_embeds = torch.stack(new_input_embeds, dim=0)

            return None, attention_mask, past_key_values, new_input_embeds, labels, None, None, None

        Xs, keys = X_modalities
        # print("DEBUG: Calling encode_images_or_videos...")
        X_features = self.encode_images_or_videos(Xs, keys, video_timestamps)
        # print("DEBUG: encode_images_or_videos returned.")
        
        ##################################################################################
        # print("DEBUG: Calling encode_time...")
        time_tokens = self.encode_time(times)
        # print("DEBUG: encode_time returned.")
        # print("DEBUG: Calling encode_score...")
        score_tokens = self.encode_score(scores)
        # print("DEBUG: encode_score returned.")
        # print(time_tokens, torch.where(input_ids == MMODAL_TOKEN_INDEX['TIME']))

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        new_time_labels = [] if labels is not None else None
        new_score_labels = [] if labels is not None else None
        new_attention_masks = [] if attention_mask is not None else None
        closs_indices = []
        cur_X_idx = 0

        # print(f'{time_tokens}')

        # print(f"DEBUG: Starting loop over batch (size {len(input_ids)})...")
        # replace image/video/audio tokens with pre-computed embeddings
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # print(f"DEBUG: Processing batch item {batch_idx}...")
            # cur_X_features = X_features[batch_idx]

            
            cur_time_tokens = torch.cat(time_tokens[batch_idx], dim=0) if len(time_tokens[batch_idx]) > 0 else torch.tensor([], dtype=torch.int)
            cur_score_tokens =torch.cat(score_tokens[batch_idx], dim=0) if len(score_tokens[batch_idx]) > 0 else torch.tensor([], dtype=torch.int)
            # print(f'{cur_score_tokens} {cur_time_tokens}')
            cur_X_features = X_features[batch_idx]
            
            if torch.isnan(cur_X_features).any() or torch.isinf(cur_X_features).any():
                print(f"CRITICAL ERROR: cur_X_features (batch {batch_idx}) contains NaN or Inf!")

            
            # cur_X_features = self.encode_images_or_videos([Xs[batch_idx]], [keys[batch_idx]], [video_timestamps[batch_idx]]).squeeze(0)
            # print(f'{Xs[batch_idx].shape} {cur_X_features.shape} {cur_X_features.device}')
            
            # DEBUG: Check time tokens
            # print(f"DEBUG: batch {batch_idx}, cur_time_tokens shape: {cur_time_tokens.shape}")
            
            # print(f"DEBUG: batch {batch_idx}, calling get_time_tower...")
            cur_time_features = self.get_time_tower()(cur_time_tokens.to(cur_X_features.device))
            if torch.isnan(cur_time_features).any() or torch.isinf(cur_time_features).any():
                print(f"CRITICAL ERROR: cur_time_features (batch {batch_idx}) contains NaN or Inf!")

            # print(f"DEBUG: batch {batch_idx}, calling get_score_tower...")
            cur_score_features = self.get_score_tower()(cur_score_tokens.to(cur_X_features.device))
            if torch.isnan(cur_score_features).any() or torch.isinf(cur_score_features).any():
                print(f"CRITICAL ERROR: cur_score_features (batch {batch_idx}) contains NaN or Inf!")

            # print(cur_score_features.shape)

            # print('finish score')

            video_position = torch.where((cur_input_ids == MMODAL_TOKEN_INDEX['VIDEO']) + (cur_input_ids == MMODAL_TOKEN_INDEX['IMAGE']))[0]
            assert len(video_position) == 1, "only have one video inputs!"
            video_position = video_position[0]

            cur_new_input_ids = torch.cat([cur_input_ids[:video_position], torch.full((cur_X_features.shape[0],), MMODAL_TOKEN_INDEX['VIDEO'], device=cur_input_ids.device, dtype=cur_input_ids.dtype), cur_input_ids[video_position+1:]], dim=0)
            
            if attention_mask is not None:
                cur_attention_mask = attention_mask[batch_idx]
                cur_new_attention_mask = torch.cat([cur_attention_mask[:video_position], torch.full((cur_X_features.shape[0],), True, device=cur_attention_mask.device, dtype=cur_attention_mask.dtype), cur_attention_mask[video_position+1:]], dim=0)
                new_attention_masks.append(cur_new_attention_mask)

            sync_token_indices = cur_new_input_ids == MMODAL_TOKEN_INDEX['SYNC']
            # print(f"DEBUG: batch {batch_idx}, calling get_sync_tower...")
            cur_sync_features = self.get_sync_tower()(cur_new_input_ids[sync_token_indices].to(cur_X_features.device))
            if torch.isnan(cur_sync_features).any() or torch.isinf(cur_sync_features).any():
                print(f"CRITICAL ERROR: cur_sync_features (batch {batch_idx}) contains NaN or Inf!")

            cur_text_input_ids = torch.clamp(cur_new_input_ids, min=0)
            # print(f"DEBUG: batch {batch_idx}, calling embed_tokens...")
            cur_new_input_embeds = self.get_model().embed_tokens(cur_text_input_ids)
            
            if torch.isnan(cur_new_input_embeds).any() or torch.isinf(cur_new_input_embeds).any():
                print(f"CRITICAL ERROR: cur_new_input_embeds (text embeddings, batch {batch_idx}) contains NaN or Inf!")

            # print(f\"{torch.sum(cur_new_input_ids == MMODAL_TOKEN_INDEX['TIME'])} {cur_time_features.shape} {times[batch_idx]}\")
            # print(cur_sync_features.shape)

            # Verify shapes before assignment to avoid silent failures or mismatches
            num_video_tokens = (cur_new_input_ids == MMODAL_TOKEN_INDEX['VIDEO']).sum()
            if num_video_tokens != cur_X_features.shape[0]:
                print(f"CRITICAL ERROR: Mismatch video tokens ({num_video_tokens}) vs features ({cur_X_features.shape[0]})")
            
            # print(f"DEBUG: batch {batch_idx}, assigning features...")
            cur_new_input_embeds[cur_new_input_ids == MMODAL_TOKEN_INDEX['VIDEO']] = cur_X_features
            cur_new_input_embeds[cur_new_input_ids == MMODAL_TOKEN_INDEX['TIME']] = cur_time_features
            cur_new_input_embeds[cur_new_input_ids == MMODAL_TOKEN_INDEX['SCORE']] = cur_score_features
            cur_new_input_embeds[cur_new_input_ids == MMODAL_TOKEN_INDEX['SYNC']] = cur_sync_features

            if torch.isnan(cur_new_input_embeds).any() or torch.isinf(cur_new_input_embeds).any():
                print(f"CRITICAL ERROR: cur_new_input_embeds (after merging, batch {batch_idx}) contains NaN or Inf!")

            # print(f'{cur_new_input_embeds.shape} {cur_new_input_embeds.device}')

            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = torch.cat([cur_labels[:video_position], torch.full((cur_X_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype), cur_labels[video_position+1:]], dim=0)

                cur_new_labels[cur_new_input_ids < 0] = IGNORE_INDEX
                cur_new_labels[cur_new_input_ids == MMODAL_TOKEN_INDEX['SYNC']] = self.vocab_size

                cur_time_labels = torch.full(cur_new_labels.shape, IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)
                cur_time_labels[cur_new_input_ids == MMODAL_TOKEN_INDEX['TIME']] = cur_time_tokens.to(device=cur_labels.device, dtype=cur_labels.dtype)

                cur_score_labels = torch.full(cur_new_labels.shape, IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)
                cur_score_labels[cur_new_input_ids == MMODAL_TOKEN_INDEX['SCORE']] = cur_score_tokens.to(device=cur_labels.device, dtype=cur_labels.dtype)

            if getattr(self.config, 'closs', False) and hasattr(self.get_model(), 'closs_tokens'):
                closs_tokens_embeds = self.get_model().closs_tokens.to(dtype=cur_new_input_embeds.dtype, device=cur_new_input_embeds.device)
                video_len = cur_X_features.shape[0]
                
                part1 = cur_new_input_embeds[:video_position + video_len]
                part2 = cur_new_input_embeds[video_position + video_len:]
                cur_new_input_embeds = torch.cat([part1, closs_tokens_embeds, part2], dim=0)
                
                if labels is not None:
                    l_part1 = cur_new_labels[:video_position + video_len]
                    l_part2 = cur_new_labels[video_position + video_len:]
                    l_closs = torch.full((3,), IGNORE_INDEX, device=cur_new_labels.device, dtype=cur_new_labels.dtype)
                    cur_new_labels = torch.cat([l_part1, l_closs, l_part2], dim=0)
                    
                    t_part1 = cur_time_labels[:video_position + video_len]
                    t_part2 = cur_time_labels[video_position + video_len:]
                    t_closs = torch.full((3,), IGNORE_INDEX, device=cur_time_labels.device, dtype=cur_time_labels.dtype)
                    cur_time_labels = torch.cat([t_part1, t_closs, t_part2], dim=0)
                    
                    s_part1 = cur_score_labels[:video_position + video_len]
                    s_part2 = cur_score_labels[video_position + video_len:]
                    s_closs = torch.full((3,), IGNORE_INDEX, device=cur_score_labels.device, dtype=cur_score_labels.dtype)
                    cur_score_labels = torch.cat([s_part1, s_closs, s_part2], dim=0)
                
                if attention_mask is not None:
                    cur_new_attention_mask = new_attention_masks.pop()
                    a_part1 = cur_new_attention_mask[:video_position + video_len]
                    a_part2 = cur_new_attention_mask[video_position + video_len:]
                    a_closs = torch.full((3,), True, device=cur_new_attention_mask.device, dtype=cur_new_attention_mask.dtype)
                    cur_new_attention_mask = torch.cat([a_part1, a_closs, a_part2], dim=0)
                    new_attention_masks.append(cur_new_attention_mask)
                
                closs_indices.append(video_position + video_len)
            else:
                closs_indices.append(-1)

            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                new_labels.append(cur_new_labels)
                new_time_labels.append(cur_time_labels)
                new_score_labels.append(cur_score_labels)
        
        # print("DEBUG: Loop finished.")
        # assert 1 < 0
        ##################################################################################
        # padding
        # max_len = 2048
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)
                
                ##################################################################################

                new_time_labels_align = []
                for cur_new_time_label in new_time_labels:
                    cur_new_time_label = torch.cat((cur_new_time_label, torch.full((max_len - cur_new_time_label.shape[0],), IGNORE_INDEX, dtype=cur_new_time_label.dtype, device=cur_new_time_label.device)), dim=0)
                    new_time_labels_align.append(cur_new_time_label)
                new_time_labels = torch.stack(new_time_labels_align, dim=0)

                new_score_labels_align = []
                for cur_new_score_label in new_score_labels:
                    cur_new_score_label = torch.cat((cur_new_score_label, torch.full((max_len - cur_new_score_label.shape[0],), IGNORE_INDEX, dtype=cur_new_score_label.dtype, device=cur_new_score_label.device)), dim=0)
                    new_score_labels_align.append(cur_new_score_label)
                new_score_labels = torch.stack(new_score_labels_align, dim=0)

                ##################################################################################

            if attention_mask is not None:
                new_attention_mask_align = []
                for cur_new_attn_mask in new_attention_masks:
                    cur_new_attn_mask = torch.cat((cur_new_attn_mask, torch.full((max_len - cur_new_attn_mask.shape[0],), False, dtype=cur_new_attn_mask.dtype, device=cur_new_attn_mask.device)), dim=0)
                    new_attention_mask_align.append(cur_new_attn_mask)
                attention_mask = torch.stack(new_attention_mask_align, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

                ##################################################################################

                new_time_labels = torch.stack(new_time_labels, dim=0)
                new_score_labels = torch.stack(new_score_labels, dim=0)

                ##################################################################################

            if attention_mask is not None:
                attention_mask = torch.stack(new_attention_masks, dim=0)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        # print(f'{new_input_embeds.shape}, {new_labels.shape}, {new_time_labels.shape}, {new_score_labels.shape}')
        # print(f'{new_labels.device}, {torch.where(new_labels != -100)}, {torch.where(new_time_labels != -100)}, {torch.where(new_score_labels != -100)}')
        # print(new_labels[new_labels != -100])
        # print(f'prepare end {new_labels.device}')

        return None, attention_mask, past_key_values, new_input_embeds, new_labels, new_time_labels, new_score_labels, closs_indices
        # return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings  = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg  = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:]  = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

    def initialize_MM_tokenizer(self, model_args, tokenizer):
        ##################################################################################
        if model_args.mm_use_im_patch_token:
            for modal in ['IMAGE', 'VIDEO', 'AUDIO', 'TIME', 'SCORE']:
                tokenizer.add_tokens([DEFAULT_MMODAL_PATCH_TOKEN[modal.upper()]], special_tokens=True)
            # tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = 0
            for modal in ['IMAGE', 'VIDEO', 'AUDIO', 'TIME', 'SCORE']:
                num_new_tokens += tokenizer.add_tokens([DEFAULT_MMODAL_START_TOKEN[modal.upper()], DEFAULT_MMODAL_END_TOKEN[modal.upper()]], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))


            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 6  # start/end tokens for image/video/audio
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
        ##################################################################################

