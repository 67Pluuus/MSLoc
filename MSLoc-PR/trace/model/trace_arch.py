
from abc import ABC, abstractmethod

import einops
import torch

from .multimodal_encoder.builder import build_vision_tower, build_time_tower, build_score_tower, build_sync_tower


from .multimodal_projector.builder import build_vision_projector
from ..constants import NUM_FRAMES, IGNORE_INDEX, MMODAL_TOKEN_INDEX


class TraceMetaModel:

    def __init__(self, config):
        super(TraceMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=False)
            self.mm_projector = build_vision_projector(config)

        self.time_tokenizer, self.time_tower = build_time_tower(None, None, 4096)
        self.score_tokenizer, self.score_tower = build_score_tower(None, None, 4096)
        self.sync_tower = build_sync_tower(4096)


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
            hw = 1

        time_features = None
        if video_timestamps is not None:
            video_time_tokens = self.encode_time(video_timestamps)

            all_tokens_list = []
            for batch_idx in range(batch_size):
                for f_idx in range(len(video_time_tokens[batch_idx])):
                    all_tokens_list.append(video_time_tokens[batch_idx][f_idx][:-1])

            if len(all_tokens_list) > 0:
                all_tokens_tensor = torch.stack(all_tokens_list).to(frames_features.device)

                all_features = self.get_time_tower()(all_tokens_tensor)

                t = len(video_time_tokens[0])
                time_features = einops.rearrange(all_features, '(b t) n h -> b t n h', b=batch_size, t=t)

            if torch.isnan(time_features).any() or torch.isinf(time_features).any():
                print("ERROR: time_features contains NaN or Inf!")

        if seperate_time_feature:
            if self.config.mm_projector_type == 'ref_projector':
                frames_features = self.temporal_aggregator(frames_features, hw, int(frames_features.shape[2] / hw), time_features=time_features)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if torch.isnan(frames_features).any() or torch.isinf(frames_features).any():
                    print("ERROR: frames_features after temporal_aggregator contains NaN or Inf!")
                return frames_features
            else:
                frames_features = self.temporal_aggregator(frames_features, hw, int(frames_features.shape[2] / hw))
                if torch.isnan(frames_features).any() or torch.isinf(frames_features).any():
                    print("ERROR: frames_features after temporal_aggregator contains NaN or Inf!")
        if time_features is not None:
            if not seperate_time_feature:
                time_features = time_features.view(time_features.shape[0], time_features.shape[1], -1, frames_features.shape[-1])

                frames_features = torch.cat([frames_features, time_features], dim=2)
                frames_features = self.temporal_aggregator(frames_features, hw, int(frames_features.shape[2] / hw))
            else:
                frames_features = torch.cat([frames_features, time_features], dim=2)

            if torch.isnan(frames_features).any() or torch.isinf(frames_features).any():
                print("ERROR: frames_features after merging time features contains NaN or Inf!")

            frames_features =  einops.rearrange(frames_features, 'b t n h -> b (t n) h')


        return frames_features


    def encode_time(self, times):

        time_tokens = []
        for batch_times in times:
            batch_time_tokens = [self.get_model().get_time_tower().encode(t) for t in batch_times]
            assert all([batch_time_token.shape == batch_time_tokens[0].shape for batch_time_token in batch_time_tokens]), f'{batch_times} {[batch_time_token.shape for batch_time_token in batch_time_tokens]}'
            time_tokens.append(batch_time_tokens)

        return time_tokens

    def encode_score(self, scores):

        score_tokens = []
        for batch_scores in scores:
            batch_score_tokens = [self.get_model().get_score_tower().encode(s) for s in batch_scores]
            score_tokens.append(batch_score_tokens)

        return score_tokens


    def temporal_aggregator(self, frames_features, h, w, time_features=None):
        """Aggregate frame features shaped (batch, time, tokens, hidden)."""
        if self.config.mm_projector_type == "mlp2x_gelu" or self.config.mm_projector_type == "linear":
            video_features = self.get_model().mm_projector(frames_features.mean(1))
        elif self.config.mm_projector_type == "spatial_conv":
            video_features = self.get_model().mm_projector(frames_features)
        elif self.config.mm_projector_type == "spatial_pool":
            video_features = self.get_model().mm_projector(frames_features)
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
        if vision_tower is None or X_modalities is None or input_ids.shape[1] == 1:
            new_input_embeds = []
            for batch_idx in range(input_ids.shape[0]):
                cur_input_ids = input_ids[batch_idx]

                cur_text_ids = cur_input_ids % self.vocab_size
                text_embeds = self.get_model().embed_tokens(cur_text_ids)
                sync_positions = (cur_input_ids == self.vocab_size)
                sync_embeds = self.get_sync_tower()(cur_input_ids[sync_positions])
                time_positions = cur_input_ids >= (self.vocab_size + 1) and cur_input_ids < (self.vocab_size + self.time_vocab_size + 1)
                cur_time_ids = cur_input_ids[time_positions] - self.vocab_size - 1
                time_embeds = self.get_time_tower()(cur_time_ids)
                score_positions = cur_input_ids >= (self.vocab_size + self.time_vocab_size + 1)
                cur_score_ids = cur_input_ids[score_positions] - self.vocab_size - self.time_vocab_size - 1
                score_embeds = self.get_score_tower()(cur_score_ids)

                text_embeds[time_positions] = time_embeds
                text_embeds[score_positions] = score_embeds
                text_embeds[sync_positions] = sync_embeds

                new_input_embeds.append(text_embeds)
            new_input_embeds = torch.stack(new_input_embeds, dim=0)

            return None, attention_mask, past_key_values, new_input_embeds, labels, None, None

        Xs, keys = X_modalities
        X_features = self.encode_images_or_videos(Xs, keys, video_timestamps)

        time_tokens = self.encode_time(times)
        score_tokens = self.encode_score(scores)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        new_time_labels = [] if labels is not None else None
        new_score_labels = [] if labels is not None else None
        new_attention_masks = [] if attention_mask is not None else None
        cur_X_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):

            cur_time_tokens = torch.cat(time_tokens[batch_idx], dim=0) if len(time_tokens[batch_idx]) > 0 else torch.tensor([], dtype=torch.int)
            cur_score_tokens =torch.cat(score_tokens[batch_idx], dim=0) if len(score_tokens[batch_idx]) > 0 else torch.tensor([], dtype=torch.int)
            cur_X_features = X_features[batch_idx]

            cur_time_features = self.get_time_tower()(cur_time_tokens.to(cur_X_features.device))
            cur_score_features = self.get_score_tower()(cur_score_tokens.to(cur_X_features.device))

            video_position = torch.where((cur_input_ids == MMODAL_TOKEN_INDEX['VIDEO']) + (cur_input_ids == MMODAL_TOKEN_INDEX['IMAGE']))[0]
            assert len(video_position) == 1, "only have one video inputs!"
            video_position = video_position[0]

            cur_new_input_ids = torch.cat([cur_input_ids[:video_position], torch.full((cur_X_features.shape[0],), MMODAL_TOKEN_INDEX['VIDEO'], device=cur_input_ids.device, dtype=cur_input_ids.dtype), cur_input_ids[video_position+1:]], dim=0)

            if attention_mask is not None:
                cur_attention_mask = attention_mask[batch_idx]
                cur_new_attention_mask = torch.cat([cur_attention_mask[:video_position], torch.full((cur_X_features.shape[0],), True, device=cur_attention_mask.device, dtype=cur_attention_mask.dtype), cur_attention_mask[video_position+1:]], dim=0)
                new_attention_masks.append(cur_new_attention_mask)

            sync_token_indices = cur_new_input_ids == MMODAL_TOKEN_INDEX['SYNC']
            cur_sync_features = self.get_sync_tower()(cur_new_input_ids[sync_token_indices].to(cur_X_features.device))

            cur_text_input_ids = torch.clamp(cur_new_input_ids, min=0)
            cur_new_input_embeds = self.get_model().embed_tokens(cur_text_input_ids)

            cur_new_input_embeds[cur_new_input_ids == MMODAL_TOKEN_INDEX['VIDEO']] = cur_X_features
            cur_new_input_embeds[cur_new_input_ids == MMODAL_TOKEN_INDEX['TIME']] = cur_time_features
            cur_new_input_embeds[cur_new_input_ids == MMODAL_TOKEN_INDEX['SCORE']] = cur_score_features
            cur_new_input_embeds[cur_new_input_ids == MMODAL_TOKEN_INDEX['SYNC']] = cur_sync_features

            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = torch.cat([cur_labels[:video_position], torch.full((cur_X_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype), cur_labels[video_position+1:]], dim=0)

                cur_new_labels[cur_new_input_ids < 0] = IGNORE_INDEX
                cur_new_labels[cur_new_input_ids == MMODAL_TOKEN_INDEX['SYNC']] = self.vocab_size

                cur_time_labels = torch.full(cur_new_labels.shape, IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)
                cur_time_labels[cur_new_input_ids == MMODAL_TOKEN_INDEX['TIME']] = cur_time_tokens.to(device=cur_labels.device, dtype=cur_labels.dtype)

                cur_score_labels = torch.full(cur_new_labels.shape, IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)
                cur_score_labels[cur_new_input_ids == MMODAL_TOKEN_INDEX['SCORE']] = cur_score_tokens.to(device=cur_labels.device, dtype=cur_labels.dtype)

            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                new_labels.append(cur_new_labels)
                new_time_labels.append(cur_time_labels)
                new_score_labels.append(cur_score_labels)

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


                new_time_labels = torch.stack(new_time_labels, dim=0)
                new_score_labels = torch.stack(new_score_labels, dim=0)


            if attention_mask is not None:
                attention_mask = torch.stack(new_attention_masks, dim=0)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels, new_time_labels, new_score_labels
