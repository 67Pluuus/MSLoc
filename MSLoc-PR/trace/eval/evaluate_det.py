import copy
import os
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    print('Using NPU')
except:
    print('Using GPU')
import argparse
import traceback

import sys
# Adjust sys.path so that the `trace` package (located at MSLoc/Trace/trace) is importable.
current_file_path = os.path.abspath(__file__)
trace_eval_dir = os.path.dirname(current_file_path)
trace_pkg_dir = os.path.dirname(trace_eval_dir)
repo_root = os.path.dirname(trace_pkg_dir)

if repo_root not in sys.path:
    sys.path.append(repo_root)
if trace_pkg_dir not in sys.path:
    sys.path.append(trace_pkg_dir)
sys.path.append('./')

from trace.conversation import conv_templates
from trace.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from trace.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, process_video, process_image
from trace.model.builder import load_pretrained_model
from trace.conversation import conv_templates, SeparatorStyle
from trace.mm_utils import tokenizer_MMODAL_token_all, KeywordsStoppingCriteria
from transformers import StoppingCriteria, StoppingCriteriaList
from math import ceil
from PIL import Image
import numpy as np
import torch.backends.cudnn as cudnn
import decord

# decord.bridge.set_bridge('torch')
import logging
from torchvision.transforms.functional import InterpolationMode

from torchvision import transforms
import pdb
import json
from pathlib import Path
import time
import datetime
from tqdm import tqdm
import random

random.seed(1234)


def read_txt(path):
    with open(path, "r") as fin:
        data = fin.readline().strip()
    return data


def load_data(args):
    print(f"Loading annotations from {args.anno_file}")
    with open(args.anno_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise TypeError(f"Expected list of videos, got {type(data)}")
        
    if args.debug:
        data = data[:10]
    return data


def save_result(args, output_dir, results, split_name='test', format=False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Output filename
    file_name = f'fmt_{args.dataset}_{split_name}_f{args.num_frames}_result.json'
    
    if args.num_chunks > 1:
        file_name = file_name.replace('.json', f'_chunk{args.chunk_idx}.json')
    if args.debug:
        file_name = 'debug_' + file_name

    # Save the results list as-is.
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

    # Suppress non-essential logs on every chunk except chunk 0.
    is_master = (int(getattr(args, 'chunk_idx', 0)) == 0)
    if getattr(args, 'quiet_non_master', False) and not is_master:
        import builtins
        _orig_print = builtins.print

        def _quiet_print(*p_args, **p_kwargs):
            if p_kwargs.pop('force', False):
                return _orig_print(*p_args, **p_kwargs)
            return None

        builtins.print = _quiet_print

    # load model
    device = torch.device(int(args.gpu_id))
    args.options = []

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, None, model_name)
    model = model.to(device)
    model.to(dtype=torch.float32)
    conv_mode = 'llama_2'
    print('Initialization Finished')

    # load data
    anno_data = load_data(args)
    print(f'Load Annotation Finished, total {len(anno_data)} videos')

    if args.sample_num > 0:
        anno_data = random.sample(anno_data, args.sample_num)
    
    # Data chunking for parallel evaluation (by video)
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

    video_root_path = args.video_path

    for item in tqdm(anno_data, position=tqdm_position, leave=True, dynamic_ncols=True):
        # Copy the item before mutating it.
        output_item = copy.deepcopy(item)
        
        # Resolve the video path.
        vname = item.get("video_path")
        if not vname:
            # video_path is required.
            print(f"Warning: Missing video_path in item: {item}")
            results.append(output_item)
            continue

        # Build the absolute path under video_root_path.
        vid_path = os.path.join(video_root_path, vname)
        if not os.path.exists(vid_path):
            # Fall back to treating `vname` as an absolute path.
            if os.path.exists(vname):
                vid_path = vname
            else:
                print(f"Video not found: {vid_path}")
                results.append(output_item)
                continue

        # Iterate over annotations.
        if 'annotations' not in output_item:
            results.append(output_item)
            continue

        for ann in output_item['annotations']:
            segment = ann.get('segment')
            if not segment:
                continue
            
            s, e = segment
            # Use the segment as-is, no padding.
            win_s, win_e = s, e
            
            # Window must be valid.
            if win_e <= win_s:
                print(f"Invalid window for {vid_path}: {win_s}-{win_e}")
                continue

            try:
                tensor, video_timestamps = process_video(
                    vid_path, processor, model.config.image_aspect_ratio, 
                    n_frms, 
                    start_time=win_s, 
                    end_time=win_e
                )
                
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"Warning: Tensor contains NaN or Inf for video {vid_path}")
                    continue

                tensor = tensor.to(dtype=torch.float32, device='cuda', non_blocking=True)

                default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
                tensor = [tensor]
                video_timestamps = [video_timestamps]
                heads = [1]
                modal_list = ['video']

                # Use the same detection prompt for every segment.
                final_prompt = copy.deepcopy(prompt)
                question = default_mm_token + "\n" + final_prompt

                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                cur_prompt = conv.get_prompt()

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
                    if idx <= 32000:
                        if idx == 32000:
                            cur_caption.append(idx)
                        else:
                            cur_caption.append(idx)
                    elif idx <= 32013:  # 32001 <sync>; 32002 <sep>
                        if len(cur_caption) > 0:
                             full_caption = tokenizer.decode(cur_caption, skip_special_tokens=True)
                             ts_to_save = last_timestamps if last_timestamps is not None else []
                             parsed_segments.append({"timestamp": ts_to_save, "caption": full_caption})
                             last_timestamps = None
                             cur_caption = []

                        if idx == 32001:
                            if len(cur_timestamp) > 0:
                                cur_timestamps.append(float(''.join(cur_timestamp)))
                            last_timestamps = cur_timestamps
                            cur_timestamps = []
                            cur_timestamp = []
                        elif idx == 32002:
                            if len(cur_timestamp) > 0:
                                cur_timestamps.append(float(''.join(cur_timestamp)))
                            cur_timestamp = []
                        else:
                            cur_timestamp.append(model.get_model().time_tokenizer.decode(idx - 32001))
                    else:
                        pass

                if len(cur_caption) > 0:
                    full_caption = tokenizer.decode(cur_caption, skip_special_tokens=True)
                    ts_to_save = last_timestamps if last_timestamps is not None else []
                    parsed_segments.append({"timestamp": ts_to_save, "caption": full_caption})
                elif last_timestamps is not None:
                    parsed_segments.append({"timestamp": last_timestamps, "caption": ""})

                # Decision rule: any predicted timestamp -> fake.
                has_timestamp = False
                fake_responses = []
                all_responses = []
                
                for item in parsed_segments:
                    ts = item["timestamp"]
                    cap = item["caption"]
                    all_responses.append(cap)
                    
                    if len(ts) >= 2:
                        has_timestamp = True
                        fake_responses.append(cap)
                
                predict_label = "fake" if has_timestamp else "real"
                
                if has_timestamp:
                    response = " ".join(fake_responses).strip()
                else:
                    response = " ".join(all_responses).strip()
                
                # Update the annotation in place.
                ann['predict_label'] = predict_label
                ann['response'] = response

            except Exception:
                traceback.print_exc()
                print(f'generate for video {vid_path} segment {segment} failed')
                # On error, leave predict_label / response unset.

        results.append(output_item)

    save_result(args, args.output_dir, results, args.split, format=True)

    total_time = time.time() - eval_start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluate time {}'.format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_path', type=str, default='')  # kept for back-compat; --anno_file is preferred
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--task', default='det')
    parser.add_argument('--dataset', default='aigc')
    parser.add_argument('--output_dir', default='debug')
    parser.add_argument('--split', default='test')
    parser.add_argument('--num_frames', type=int, default=32)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=1)  # we process one segment at a time
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--prompt_file', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--sample_num', type=int, default=-1)
    parser.add_argument('--num_chunks', type=int, default=1)
    parser.add_argument('--chunk_idx', type=int, default=0)
    parser.add_argument('--anno_file', type=str, required=True)
    parser.add_argument('--quiet_non_master', action='store_true')
    parser.add_argument('--tqdm_position', type=int, default=-1)
    
    args = parser.parse_args()
    main(args)


