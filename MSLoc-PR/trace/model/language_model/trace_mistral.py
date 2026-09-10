# Adopted from: https://github.com/haotian-liu/LLaVA. Below is the original copyright:
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Any
from torch import Tensor

from transformers import AutoConfig, AutoModelForCausalLM, \
                         MistralConfig, MistralModel, MistralForCausalLM, AutoModelForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..trace_arch import TraceMetaModel, TraceMetaForCausalLM

class LMHeadBackward(torch.autograd.Function):
    # jump the sign operation as the sign operation does not have gradients

    @staticmethod
    def forward(ctx: Any, logits: Tensor):
        return logits

    @staticmethod
    def backward(ctx:Any, grad_output: Tensor):
        # print(grad_output.shape)
        grad_output[:, :, :-1] = 0
        # print(grad_output)
        return grad_output

class LMEmbedBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, embeddings: Tensor):
        return embeddings

    @staticmethod
    def backward(ctx:Any, grad_output: Tensor):
        print(grad_output.shape)
        grad_output[:, :-1] = 0
        # print(grad_output)
        return grad_output

def lm_embed_hook(module, input, output):
    output = LMEmbedBackward.apply(output)

class TraceMistralConfig(MistralConfig):
    model_type = "trace_mistral"


class TraceMistralModel(TraceMetaModel, MistralModel):
    config_class = TraceMistralConfig

    def __init__(self, config: MistralConfig):
        super(TraceMistralModel, self).__init__(config)


class TraceMistralForCausalLM(MistralForCausalLM, TraceMetaForCausalLM):
    config_class = TraceMistralConfig

    def __init__(self, config, **kwargs):
        super(MistralForCausalLM, self).__init__(config)

        # config.time_vocab_size = 12
        # config.score_vocab_size = 12

        self.model = TraceMistralModel(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
        self.swap_tokens = {config.vocab_size: 1,
                            config.vocab_size + 1: 2, 
                            config.vocab_size + config.time_vocab_size + 1: 0}
        # #####################################################################################################

        self.time_head = nn.Linear(config.hidden_size, config.time_vocab_size, bias=False)

        self.score_head = nn.Linear(config.hidden_size, config.score_vocab_size, bias=False)
        self.sync_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.time_vocab_size = config.time_vocab_size
        self.score_vocab_size = config.score_vocab_size

        # hook_handle = self.get_model().embed_tokens.register_forward_hook(lm_embed_hook)
        # #####################################################################################################

        # Initialize weights and apply final processing
        self.post_init()

    def initialize_embed_and_heads(self, model_args, tokenizer):

        tokenizer.add_tokens(['<sync>'], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))



    def get_model(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        times: Optional[List[List[List[float]]]] = None,
        scores: Optional[List[List[List[float]]]] = None,
        video_timestamps: Optional[List[List[List[float]]]] = None,
        heads: Optional[List[int]] = None,
        closs_input_ids: Optional[torch.LongTensor] = None,
        all_class_input_ids: Optional[torch.LongTensor] = None,
        closs_labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # print(input_ids)

        closs_indices = None
        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                time_labels, 
                score_labels,
                closs_indices
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                times,
                scores,
                video_timestamps=video_timestamps
            )

        # print(f'prepared {labels[labels > 0]} {labels > 0}')
        # print(self.time_head.weight)

        # return super().forward(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict
        # )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            # print(f"DEBUG: inputs_embeds shape: {inputs_embeds.shape}, dtype: {inputs_embeds.dtype}")
            if torch.isnan(inputs_embeds).any() or torch.isinf(inputs_embeds).any():
                print("CRITICAL ERROR: inputs_embeds contains NaN or Inf BEFORE model forward!")
            
            # Print stats
            inputs_min = inputs_embeds.min().item()
            inputs_max = inputs_embeds.max().item()
            inputs_mean = inputs_embeds.mean().item()
            inputs_std = inputs_embeds.std().item()
            # print(f"DEBUG: inputs_embeds stats: min={inputs_min}, max={inputs_max}, mean={inputs_mean}, std={inputs_std}")
        
        if attention_mask is not None:
            # print(f"DEBUG: attention_mask shape: {attention_mask.shape}, dtype: {attention_mask.dtype}")
            # Check if any row in attention_mask is all zeros (which would cause NaN in softmax)
            if (attention_mask.sum(dim=-1) == 0).any():
                print("CRITICAL ERROR: Found a row in attention_mask that is ALL ZEROS!")

        # print("DEBUG: Calling self.model (MistralModel)...")
        if torch.cuda.is_available(): torch.cuda.synchronize()

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        # print("DEBUG: self.model returned.")
        if torch.cuda.is_available(): torch.cuda.synchronize()

        hidden_states = outputs[0]
        
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            print("CRITICAL ERROR: hidden_states contains NaN or Inf!")
            # print(f"Hidden states stats: min={hidden_states.min()}, max={hidden_states.max()}, mean={hidden_states.mean()}")

        # print("DEBUG: Computing lm_head...")
        logits = self.lm_head(hidden_states)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        
        # print("DEBUG: Computing sync_head...")
        sync_logits = self.sync_head(hidden_states)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        
        logits = torch.cat([logits, sync_logits], dim=-1)
        logits = logits.float()

        #####################################################################################################
        # print("DEBUG: Computing time_head...")
        time_logits = self.time_head(hidden_states)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        time_logits = time_logits.float()
        
        # print("DEBUG: Computing score_head...")
        score_logits = self.score_head(hidden_states)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        score_logits = score_logits.float()

        # logits = torch.cat([logits, time_logits, score_logits], dim=-1)

        logits_list = [logits, time_logits, score_logits]
        if labels is not None:
            labels_list = [labels, time_labels, score_labels]
        # logits_list = [time_logits]
        # labels_list = [time_labels]
        # vocab_size_list = [self.config.time_vocab_size]
        # modals = ['time']
        vocab_size_list = [self.vocab_size + 1, self.config.time_vocab_size, self.config.score_vocab_size]
        # modals = ['text', 'time', 'score']
        # for m, l in zip(modals, labels_list):
        #     print(f'{m}, {torch.max(l)}')
        # print(vocab_size_list)
        # print(logits_list)

        loss = None
        if labels is not None:
            # print("DEBUG: Computing loss...")
            loss = []
            loss_fct = CrossEntropyLoss()

            for i, (cur_logits, cur_labels, cur_vocab_size) in enumerate(zip(logits_list, labels_list, vocab_size_list)):
                # print(f"DEBUG: Loss component {i}...")
                if torch.isnan(cur_logits).any() or torch.isinf(cur_logits).any():
                    print(f"CRITICAL ERROR: cur_logits {i} contains NaN or Inf!")

                shift_logits = cur_logits[..., :-1, :].contiguous()
                shift_labels = cur_labels[..., 1:].contiguous()
                # print(f'cur logits {shift_logits}, labels {shift_labels}, shapes, {shift_logits.shape}, {shift_labels.shape}, {cur_modal}')
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, cur_vocab_size)
                shift_labels = shift_labels.view(-1)
                # Ensure tensors are on the same device
                shift_labels = shift_labels.to(shift_logits.device)
                
                # Check for valid labels
                # valid_labels_mask = shift_labels != -100
                # if valid_labels_mask.sum() == 0:
                #     print(f"DEBUG: No valid labels for component {i}")
                
                cur_loss = loss_fct(shift_logits, shift_labels)
                # print(f"DEBUG: Loss component {i} computed: {cur_loss.item()}")
                loss.append(cur_loss)
            # print(loss)
                # print(f'{cur_labels[cur_labels != -100]} {cur_logits[cur_labels != 100]} {loss_fct(shift_logits, shift_labels)}')

            loss = sum(loss)
            # print(f"DEBUG: Total loss computed: {loss.item()}")

            # ==================== New CLoss Logic: Classification ====================
            # 使用分类头直接对特殊 token 进行分类训练
            if getattr(self.config, 'closs', False) and closs_labels is not None and closs_indices is not None:
                closs_head = self.get_model().get_closs_head()
                
                # 强制检查：CLoss 模块必须正确初始化
                if closs_head is None:
                    raise RuntimeError(
                        "CRITICAL ERROR: closs_head is None but closs is enabled!\\n"
                        "Please call model.initialize_closs_modules() before training."
                    )
                
                if closs_head is not None:
                    # 1. 提取 3 个特殊 Token 的 Hidden States
                    closs_features = []
                    valid_mask = []
                    
                    for b, idx in enumerate(closs_indices):
                        if idx != -1 and idx + 3 <= hidden_states.shape[1]:
                            feats = hidden_states[b, idx:idx+3, :]  # 3 x H
                            closs_features.append(feats)
                            valid_mask.append(True)
                        else:
                            # 记录错误但不崩溃，填充 0
                            print(f"CRITICAL WARNING: Invalid closs_index {idx} for batch {b}. hidden_states.shape={hidden_states.shape}")
                            closs_features.append(torch.zeros(3, hidden_states.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype))
                            valid_mask.append(False)
                    
                    closs_features = torch.stack(closs_features)  # B x 3 x H
                    
                    if torch.isnan(closs_features).any() or torch.isinf(closs_features).any():
                        print(f"CRITICAL ERROR: closs_features contains NaN or Inf!")
                    
                    # 2. 通过分类头计算 Logits
                    # (B, 3, H) -> (B, 3, num_classes)
                    logits_closs = closs_head(closs_features)
                    
                    if torch.isnan(logits_closs).any() or torch.isinf(logits_closs).any():
                        print(f"CRITICAL ERROR: logits_closs contains NaN or Inf!")
                    
                    # 3. 计算分类 Loss
                    B, N, C = logits_closs.shape
                    
                    # Flatten
                    logits_flat = logits_closs.view(-1, C) # (B*3, C)
                    targets_flat = closs_labels.view(-1).to(logits_closs.device) # (B*3)
                    
                    # 构建有效 mask
                    valid_mask_tensor = torch.tensor(valid_mask, device=logits_closs.device)
                    valid_mask_tensor = valid_mask_tensor.unsqueeze(1).expand(-1, 3).reshape(-1) # (B*3)
                    
                    # 过滤 -100 和 无效样本
                    # 注意：closs_labels 中 -100 表示忽略（例如 padding 或空类别）
                    target_mask = (targets_flat != -100) & valid_mask_tensor
                    
                    if target_mask.sum() > 0:
                        loss_fct = CrossEntropyLoss()
                        closs_loss = loss_fct(logits_flat[target_mask], targets_flat[target_mask])
                        
                        if torch.isnan(closs_loss).any() or torch.isinf(closs_loss).any():
                            print(f"CRITICAL ERROR: closs_loss is NaN or Inf!")
                            closs_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
                        
                        loss += closs_loss
                        self.last_closs = closs_loss.detach().item()
                        
                        # Logging
                        if hasattr(self, '_closs_log_step'):
                            self._closs_log_step += 1
                        else:
                            self._closs_log_step = 0
                        
                        if self._closs_log_step % 1 == 0:
                            with torch.no_grad():
                                pred = logits_flat[target_mask].argmax(dim=-1)
                                gt = targets_flat[target_mask]
                                acc = (pred == gt).float().mean().item()
                                print(f"[CLoss] step={self._closs_log_step}, loss={closs_loss.item():.4f}, acc={acc:.4f}, valid_samples={target_mask.sum().item()}")
                    else:
                        print(
                            f"CRITICAL WARNING: No valid samples for CLoss computation!\\n"
                            f"  - target_mask.sum() = 0\\n"
                            f"  - closs_labels: {closs_labels}\\n"
                            f"  - closs_indices: {closs_indices}\\n"
                        )
                        self.last_closs = 0.0
                else:
                    self.last_closs = 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        
        if heads is not None:
            assert len(heads) == logits.shape[0]
            logits = torch.cat([logits, time_logits, score_logits], dim=-1)
            # print(logits.shape)
            vocab_size_list = [(0, self.vocab_size + 1), (self.vocab_size + 1, self.vocab_size + self.config.time_vocab_size + 1), (self.vocab_size + self.config.time_vocab_size + 1, self.vocab_size + self.config.time_vocab_size + self.config.score_vocab_size + 1)]
            for batch_idx, head in enumerate(heads):
                cur_vocab_size = vocab_size_list[head]
                logits[batch_idx, ..., :cur_vocab_size[0]] = float('-inf')
                logits[batch_idx, ..., cur_vocab_size[1]:] = float('-inf')
        
        # print(logits.shape, logits)
        # print(loss)
        

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        #####################################################################################################

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images_or_videos: Optional[torch.Tensor] = None,
        times: Optional[List[List[List[float]]]] = None,
        scores: Optional[List[List[List[float]]]] = None,
        video_timestamps: Optional[List[List[List[float]]]] = None,
        modal_list: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # print(inputs)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images_or_videos is not None:
            if times is None:
                times = [[]] * len(images_or_videos)
            if scores is None:
                scores = [[]] * len(images_or_videos)
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                _, _, _, _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=inputs,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                X_modalities=[images_or_videos, modal_list],
                times=times,
                scores=scores,
                video_timestamps=video_timestamps
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # print(input_ids)
        images = kwargs.pop("images", None)
        scores = kwargs.pop("scores", None)
        times = kwargs.pop("times", None)
        heads = kwargs.pop("heads", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        # if 'input_ids' in _inputs:
        #     print(_inputs["input_ids"])
        # else:
        #     print('No inputs id here')
        if images is not None:
            _inputs['images'] = images
        if times is not None:
            _inputs['times'] = times
        if scores is not None:
            _inputs['scores'] = scores
        if heads is not None:
            # switch the lm head if encounter <sync>
            if "input_ids" in _inputs:
                new_tokens = _inputs["input_ids"][:, -1]
                for batch_idx, new_token in enumerate(new_tokens):
                    if int(new_token) in self.swap_tokens:
                        heads[batch_idx] = self.swap_tokens[int(new_token)]

            _inputs['heads'] = heads
        # print(_inputs['heads'])
        # print(_inputs['heads'])
        return _inputs


AutoConfig.register("trace_mistral", TraceMistralConfig)
AutoModelForCausalLM.register(TraceMistralConfig, TraceMistralForCausalLM)


