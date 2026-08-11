import copy
import os
import torch
import argparse
import traceback

import sys
current_file_path = os.path.abspath(__file__)
trace_eval_dir = os.path.dirname(current_file_path)
trace_pkg_dir = os.path.dirname(trace_eval_dir)
repo_root = os.path.dirname(trace_pkg_dir)

if repo_root in sys.path:
    sys.path.remove(repo_root)
sys.path.insert(0, repo_root)

from trace.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from trace.mm_utils import get_model_name_from_path, process_video_ref_split
from trace.model.builder import load_pretrained_model
from trace.conversation import conv_templates, SeparatorStyle
from trace.mm_utils import tokenizer_MMODAL_token_all

from math import ceil
import numpy as np
import torch.backends.cudnn as cudnn
import decord

import json
from pathlib import Path
import time
import datetime
from tqdm import tqdm
import random

random.seed(1234)


def safe_decode_text(tokenizer, token_ids):
    """Decode only base text-token ids; generated sync/time/score ids are not SentencePiece ids."""
    base_vocab_size = getattr(tokenizer, "vocab_size", None)
    filtered_ids = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id < 0:
            continue
        if base_vocab_size is not None and token_id >= base_vocab_size:
            continue
        filtered_ids.append(token_id)
    return tokenizer.decode(filtered_ids, skip_special_tokens=True) if filtered_ids else ""


def read_txt(path):
    with open(path, "r") as fin:
        data = fin.readline().strip()
    return data


def load_data(args):
    print(f"Loading proposals from {args.anno_file}")
    with open(args.anno_file, "r", encoding="utf-8") as file:
        proposals = json.load(file)

    if not isinstance(proposals, list):
        raise TypeError("Stage-1 predictions must be a JSON list")

    data = []
    for index, item in enumerate(proposals):
        video_path = item.get("video_path")
        if not video_path:
            raise ValueError(f"Proposal record {index} has no video_path")

        converted = item.copy()
        converted["image_id"] = video_path
        converted["id"] = item.get("id", index)
        converted.setdefault("caption", "")
        data.append(converted)

    if args.debug:
        data = data[:10]
    return data


def save_result(args, output_dir, results, split_name='test', format=False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    file_name = f'{args.dataset}_{split_name}_f{args.num_frames}_result.json'
    if args.num_chunks > 1:
        file_name = file_name.replace('.json', f'_chunk{args.chunk_idx}.json')
    if args.debug:
        file_name = 'debug_' + file_name
    if format:
        file_name = 'fmt_' + file_name

    with open(os.path.join(output_dir, file_name), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return


def main(args):
    num_beams = 1
    temperature = args.temperature
    top_p = args.top_p
    n_frms = args.num_frames
    eval_start_time = time.time()
    prompt = read_txt(args.prompt_file)

    is_master = (int(getattr(args, 'chunk_idx', 0)) == 0)
    if getattr(args, 'quiet_non_master', False) and not is_master:
        import builtins
        _orig_print = builtins.print

        def _quiet_print(*p_args, **p_kwargs):
            if p_kwargs.pop('force', False):
                return _orig_print(*p_args, **p_kwargs)
            return None

        builtins.print = _quiet_print

    device = torch.device(int(args.gpu_id))
    args.options = []

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


    model_name = get_model_name_from_path(args.model_path)
    load_kwargs = {'device_map': None}
    if args.vision_tower is not None:
        load_kwargs['vision_tower'] = args.vision_tower

    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, None, model_name, **load_kwargs)
    model = model.to(device)
    model.to(dtype=torch.float16)
    text_sync_id = model.vocab_size
    time_token_start = model.vocab_size + 1
    time_token_end = model.vocab_size + model.config.time_vocab_size
    time_sync_id = time_token_start + model.get_model().time_tokenizer.vocab['<sync>']
    time_sep_id = time_token_start + model.get_model().time_tokenizer.vocab['<sep>']

    conv_mode = 'llama_2'
    print('Initialization Finished')

    video_path = args.video_path
    anno_data = load_data(args)
    print('Load Annotation Finished')

    unique_data = {}
    unique_list = []
    seen_vids = set()

    print(f"Before deduplication: {len(anno_data)} samples")

    for item in anno_data:
        vid = item.get('video_path') or item.get('video')

        if not vid:
            iid = item.get('image_id')
            if iid and isinstance(iid, str) and ('.mp4' in iid or '.avi' in iid or '/' in iid):
                vid = iid
            else:
                vid = iid

        if vid:
            if vid not in seen_vids:
                seen_vids.add(vid)
                unique_list.append(item)
        else:
            unique_list.append(item)

    if len(unique_list) < len(anno_data):
        print(f"Deduplicated data: {len(anno_data)} -> {len(unique_list)} unique videos")
        anno_data = unique_list
    else:
        print(f"No duplicates found (based on video path). Total: {len(anno_data)}")

    if args.sample_num > 0:
        anno_data = random.sample(anno_data, args.sample_num)

    if args.num_chunks > 1:
        total_len = len(anno_data)
        chunk_size = ceil(total_len / args.num_chunks)
        start_idx = args.chunk_idx * chunk_size
        end_idx = min((args.chunk_idx + 1) * chunk_size, total_len)
        anno_data = anno_data[start_idx:end_idx]
        print(f"Processing chunk {args.chunk_idx}/{args.num_chunks} (indices {start_idx}-{end_idx}, {len(anno_data)} samples)")

    results = []

    tqdm_position = int(getattr(args, 'tqdm_position', -1))
    if tqdm_position < 0:
        tqdm_position = int(getattr(args, 'chunk_idx', 0))

    for item in tqdm(anno_data, position=tqdm_position, leave=True, dynamic_ncols=True):
        vname = item.get("image_id")
        vid_path = os.path.join(video_path, vname)
        if not os.path.exists(vid_path):
            if os.path.exists(item.get("video_path", "")):
                vid_path = item.get("video_path")
            else:
                vid_path = os.path.join(video_path, item.get("video_path", ""))
                if not os.path.exists(vid_path):
                    print(f"Video not found: {vname} / {item.get('video_path')}")
                    continue

        duration = item.get("duration")
        if duration is None:
            try:
                vr = decord.VideoReader(vid_path, ctx=decord.cpu(0))
                duration = len(vr) / vr.get_avg_fps()
            except Exception as e:
                print(f"Failed to read duration for {vid_path}: {e}")
                duration = 10000.0

        source_segments = []
        if "model_inference" in item and "segment" in item["model_inference"]:
            source_segments = item["model_inference"]["segment"]
        else:
            print(f"Skip {vid_path}: missing `model_inference.segment`")
            continue

        model_inference_segments = []
        model_inference_responses = []
        is_fake = (item.get('type') == 'fake')

        for segment in source_segments:
            if not segment:
                continue

            s, e = segment
            center = (s + e) / 2
            length = e - s
            expand_ratio = 1.0
            new_len = length * expand_ratio
            half_len = new_len / 2

            win_s = max(0.0, center - half_len)
            win_e = min(duration, center + half_len)

            if win_e <= win_s:
                print(f"Invalid window for {vid_path}: {win_s}-{win_e}")
                model_inference_segments.append([-99, -99])
                continue

            try:
                tensor, video_timestamps = process_video_ref_split(
                    vid_path,
                    processor,
                    model.config.image_aspect_ratio,
                    bnd_frames=args.bnd_frames,
                    seg_frames=args.seg_frames,
                    bnd_ratio=args.bnd_ratio,
                    start_time=win_s,
                    end_time=win_e
                )

                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"Warning: Tensor contains NaN or Inf for video {vid_path}")
                    raise ValueError("NaN/Inf tensor")

                tensor = tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)

                default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
                tensor = [tensor]
                video_timestamps = [video_timestamps]
                heads = [1]
                modal_list = ['video']


                final_prompt = copy.deepcopy(prompt)
                question = default_mm_token + "\n" + final_prompt


                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                cur_prompt = conv.get_prompt()

                cur_prompt += '<sync>'

                input_ids = tokenizer_MMODAL_token_all(cur_prompt, tokenizer, return_tensors='pt').unsqueeze(0).to('cuda')
                attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

                stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE] else conv.sep2
                keywords = [stop_str]
                do_sample = False

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_masks,
                        images_or_videos=tensor,
                        modal_list=modal_list,
                        do_sample=do_sample,
                        temperature=0.2 if do_sample else 0.0,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                        video_timestamps=video_timestamps,
                        heads=heads
                    )

                parsed_segments = []
                cur_timestamps = []
                cur_timestamp = []
                cur_caption = []
                last_timestamps = None

                for idx in output_ids[0]:
                    token_id = int(idx)
                    if token_id < text_sync_id:
                        cur_caption.append(token_id)
                    elif token_id == text_sync_id:
                        pass
                    elif token_id <= time_token_end:
                        if len(cur_caption) > 0:
                             full_caption = safe_decode_text(tokenizer, cur_caption)
                             if last_timestamps is not None:
                                 parsed_segments.append({"timestamp": last_timestamps, "caption": full_caption})
                             last_timestamps = None
                             cur_caption = []

                        if token_id == time_sync_id:
                            if len(cur_timestamp) > 0:
                                cur_timestamps.append(float(''.join(cur_timestamp)))
                            last_timestamps = cur_timestamps
                            cur_timestamps = []
                            cur_timestamp = []
                        elif token_id == time_sep_id:
                            if len(cur_timestamp) > 0:
                                cur_timestamps.append(float(''.join(cur_timestamp)))
                            cur_timestamp = []
                        else:
                            cur_timestamp.append(model.get_model().time_tokenizer.decode(token_id - time_token_start))
                    else:
                        pass

                if len(cur_caption) > 0:
                    full_caption = safe_decode_text(tokenizer, cur_caption)
                    if last_timestamps is not None:
                        parsed_segments.append({"timestamp": last_timestamps, "caption": full_caption})
                elif last_timestamps is not None:
                    parsed_segments.append({"timestamp": last_timestamps, "caption": ""})

                abs_s, abs_e = -99, -99
                response_text = ""

                if len(parsed_segments) > 0:
                    item_res = parsed_segments[0]
                    ts = item_res["timestamp"]
                    response_text = item_res["caption"]
                    if len(ts) >= 2:
                        rel_s, rel_e = ts[0], ts[1]
                        abs_s = win_s + rel_s
                        abs_e = win_s + rel_e

                if abs_s != -99 and abs_e != -99:
                    model_inference_segments.append([abs_s, abs_e])
                    model_inference_responses.append(response_text)
                else:
                    model_inference_segments.append([-99, -99])
                    model_inference_responses.append("")

            except Exception:
                traceback.print_exc()
                print(f'generate for video {vid_path} segment {segment} failed')
                model_inference_segments.append([-99, -99])

        final_output = item.copy()
        final_output['annotations'] = item.get("annotations", [])

        valid_segments = []
        valid_responses = []
        for s, r in zip(model_inference_segments, model_inference_responses):
            if s != [-99, -99]:
                valid_segments.append(s)
                valid_responses.append(r)

        model_inference = {
            "segment": valid_segments,
            "response": valid_responses
        }

        if len(valid_segments) == 0:
            model_inference["type"] = "real"
        else:
            model_inference["type"] = "fake"

        final_output['model_inference'] = model_inference

        if 'video_path' not in final_output:
             final_output['video_path'] = item.get("image_id")

        results.append(final_output)

    save_result(args, args.output_dir, results, args.split, format=True)

    total_time = time.time() - eval_start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluate time {}'.format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', required=True)
    parser.add_argument('--dataset', default='youcook')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--split', default='val')
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--debug', action='store_true', help='the debug mode will only use 10 data samples')
    parser.add_argument('--prompt_file', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--vision_tower', type=str, default=None, help='Override the vision tower path saved in model config')
    parser.add_argument('--sample_num', type=int, default=-1, help='fast inference by sampling N instances to evaluate')
    parser.add_argument('--num_chunks', type=int, default=1)
    parser.add_argument('--chunk_idx', type=int, default=0)
    parser.add_argument('--anno_file', required=True, help='Stage-1 predictions.json path')
    parser.add_argument('--quiet_non_master', action='store_true', help='Only chunk0 prints normal logs (progress bars still show)')
    parser.add_argument('--tqdm_position', type=int, default=-1, help='tqdm position for multi-progress display; default uses chunk_idx')

    parser.add_argument('--bnd_ratio', type=float, default=0.2, help='Ratio of boundary region (each side).')
    parser.add_argument('--bnd_frames', type=int, default=16, help='Number of frames to sample from each boundary region.')
    parser.add_argument('--seg_frames', type=int, default=8, help='Number of frames to sample from the event region.')
    args = parser.parse_args()
    main(args)
