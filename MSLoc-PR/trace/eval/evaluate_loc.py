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
from trace.mm_utils import process_video, tokenizer_MMODAL_token_all, get_model_name_from_path, KeywordsStoppingCriteria

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


def load_data(args, anno_path, split=None):
    '''
    anno data example:
    {"annotations":
        [
            {
                "image_id": "xHr8X2Wpmno.mp4"
                ...
            },
            ...
        ]
    }
    
    Or training data format:
    [
        {
            "video": "path/to/video.mp4",
            "id": 1,
            "conversations": [...],
            "times": [[start, end], ...],
            ...
        },
        ...
    ]
    '''
    if args.anno_file:
        file_path = args.anno_file
    else:
        file_path = os.path.join(anno_path, f'{split}.caption_coco_format.json')
    
    print(f"Loading annotations from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # Accept three input shapes:
    # 1) {"annotations": [...]}  (COCO-style)
    # 2) [...]                  (annotations array directly)
    # 3) Training-data format    (uses `video` instead of `image_id`)
    if isinstance(raw, dict):
        data = raw.get("annotations", [])
    elif isinstance(raw, list):
        data = raw
    else:
        raise TypeError(f"Unsupported annotation json root type: {type(raw)}")

    # Convert raw / training annotations into the eval format.
    # Accept both `video_path` (custom) and `video` (training).
    if data and isinstance(data[0], dict):
        needs_conversion = False
        if "image_id" not in data[0]:
            if "video" in data[0] or "video_path" in data[0]:
                needs_conversion = True
        
        if needs_conversion:
            print("Detected training/raw data format, converting to evaluation format...")
            converted_data = []
            for idx, item in enumerate(data):
                # Copy the original item to keep all fields (source, duration, ...).
                converted_item = item.copy()
                
                # Prefer video_path, fall back to video.
                vid = item.get("video_path") or item.get("video")
                
                # Use the index as fallback id.
                item_id = item.get("id")
                if item_id is None:
                    item_id = idx

                # Fields required by the model.
                converted_item["image_id"] = vid
                converted_item["id"] = item_id
                if "caption" not in converted_item:
                    converted_item["caption"] = ""  # caption is unused at inference time
                
                # Carry segments through if available.
                if "times" in item and item["times"]:
                    converted_item["segments"] = item["times"]
                elif "segments" in item:
                    converted_item["segments"] = item["segments"]
                
                converted_data.append(converted_item)
            data = converted_data
            print(f"Converted {len(data)} samples from raw format to evaluation format")

    if args.debug:
        data = data[:10]
    return data


def save_result(args, output_dir, results, split_name='test', format=False):
    """Persist `results` to disk as a list (the caller has already built the final shape)."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Output filename
    file_name = f'{args.dataset}_{split_name}_f{args.num_frames}_result.json'
    if args.timestamp:
        if args.timestamp_file != '':
            file_name = f'{args.dataset}_{split_name}_f{args.num_frames}_result_with_pred_timestamp.json'
        else:
            file_name = f'{args.dataset}_{split_name}_f{args.num_frames}_result_with_gt_timestamp.json'
    if args.num_chunks > 1:
        file_name = file_name.replace('.json', f'_chunk{args.chunk_idx}.json')
    if args.debug:
        file_name = 'debug_' + file_name
    if format:
        file_name = 'fmt_' + file_name

    # Save the results list as-is.
    with open(os.path.join(output_dir, file_name), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return


def get_timestamp_from_file(timestamp_file):
    timestamp = {}
    with open(timestamp_file, 'r') as f:
        data = json.load(f)
        for vid, vlist in data.items():
            timestamp[vid] = []
            for vterm in vlist:
                timestamp[vid].append(vterm["timestamp"])
    return timestamp


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
            # Allow explicit force=True to bypass quiet mode.
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

    # set after init_distributed_mode() to only log on master.

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, None, model_name)
    model = model.to(device)
    model.to(dtype=torch.float32)
    conv_mode = 'llama_2'
    print('Initialization Finished')

    # load data
    video_path = args.video_path
    anno_path = args.anno_path
    anno_data = load_data(args, anno_path, split=args.split)
    print('Load Annotation Finished')

    # Deduplicate by video path to avoid redundant inference
    unique_data = {}
    # Preserve original order.
    unique_list = []
    seen_vids = set()
    
    print(f"Before deduplication: {len(anno_data)} samples")
    
    for item in anno_data:
        # Prefer video_path / video as the dedup key; fall back to image_id when it looks like a path.
        vid = item.get('video_path') or item.get('video')
        
        if not vid:
            iid = item.get('image_id')
            if iid and isinstance(iid, str) and ('.mp4' in iid or '.avi' in iid or '/' in iid):
                vid = iid
            else:
                # When image_id is not a path (e.g. a numeric id) we still use it as the
                # dedup key. It may not perfectly deduplicate but there is no better key.
                vid = iid
        
        # Keep the item if vid is set and unseen.
        if vid:
            if vid not in seen_vids:
                seen_vids.add(vid)
                unique_list.append(item)
        else:
            # Extremely rare: neither video_path nor image_id is set. Keep the item to be safe.
            unique_list.append(item)
    
    if len(unique_list) < len(anno_data):
        print(f"Deduplicated data: {len(anno_data)} -> {len(unique_list)} unique videos")
        anno_data = unique_list
    else:
        print(f"No duplicates found (based on video path). Total: {len(anno_data)}")

    if args.timestamp_file != '':
        pred_timestamps = get_timestamp_from_file(args.timestamp_file)
    vids = []
    vnames = []
    captions = []
    qids = []
    missing_videos = []
    if args.sample_num > 0:
        # sample part data to evaluate
        anno_data = random.sample(anno_data, args.sample_num)
    
    # Data chunking for parallel evaluation
    if args.num_chunks > 1:
        total_len = len(anno_data)
        chunk_size = ceil(total_len / args.num_chunks)
        start_idx = args.chunk_idx * chunk_size
        end_idx = min((args.chunk_idx + 1) * chunk_size, total_len)
        anno_data = anno_data[start_idx:end_idx]
        print(f"Processing chunk {args.chunk_idx}/{args.num_chunks} (indices {start_idx}-{end_idx}, {len(anno_data)} samples)")

    for jterm in anno_data:
        vname = jterm["image_id"]
        vid_path = os.path.join(video_path, vname)
        if not os.path.exists(vid_path):
            missing_videos.append(vid_path)
            continue
        if args.timestamp:
            duration = int(jterm["duration"])
            if args.timestamp_file == '':  # input the gt timestamps
                timestamp = jterm["segments"]
            else:  # input the pred timestamps
                timestamp = pred_timestamps[vname]
            for (start_time, end_time) in timestamp:
                # process anno timestamp error
                if start_time >= end_time or end_time > duration or start_time >= duration:
                    continue
                vids.append(vid_path)
                vnames.append(vname + "_" + str(start_time) + "_" + str(end_time))
                # image_emb, _ = model.encode_img(video)
                # img_lists.append([image_emb])
        else:
            vids.append(vid_path)
            vnames.append(vname)
            captions.append(jterm["caption"])
            qids.append(jterm["id"])

    print(f'Parse Data Done, {len(vnames)} videos')

    results = []
    bz = args.batch_size
    # evaluate using batch
    epoch = ceil(len(vnames) / bz)
    # epoch = 1

    # Concurrent tqdm bars: one position per chunk.
    tqdm_position = int(getattr(args, 'tqdm_position', -1))
    if tqdm_position < 0:
        tqdm_position = int(getattr(args, 'chunk_idx', 0))

    last_video_path = None

    for i in tqdm(range(epoch), position=tqdm_position, leave=True, dynamic_ncols=True):
        # The original implementation indexed `vids[i]` per epoch. That works for bz=1 but
        # would otherwise reuse only the first sample of every batch, making it look like
        # all GPUs are processing the same video. We slice strictly per batch instead.
        start = i * bz
        end = min((i + 1) * bz, len(vnames))

        batch_vids = vids[start:end]
        batch_vnames = vnames[start:end]
        batch_caps = captions[start:end] if captions else []
        batch_qids = qids[start:end] if qids else []

        for b, path in enumerate(batch_vids):
            # Check for duplicate video processing
            if last_video_path == path:
                raise RuntimeError(f"Error: Duplicate video processing detected! Video path: {path}. "
                                   "This indicates a potential bug in data loading or batch processing.")
            else:
                print(last_video_path, path)

            last_video_path = path

            question = ''
            image_id = batch_qids[b] if b < len(batch_qids) else None
            final_prompt = copy.deepcopy(prompt)

            if args.task in ["tvg", "vhd"]:
                # tvg / vhd: prompt requires .format(query / caption)
                cap = batch_caps[b] if b < len(batch_caps) else ''
                question = final_prompt.format(cap.strip())
            else:
                question = final_prompt

            # Print only the first few samples per chunk to verify chunking works.
            if start + b < 3:
                print(f"[chunk={args.chunk_idx}] sample_idx={start + b} video={path}", flush=True)

            try:
                print(f"Processing video: {path}")
                tensor, video_timestamps = process_video(path, processor, model.config.image_aspect_ratio, n_frms)
                print("Video processed successfully")

                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"Warning: Tensor contains NaN or Inf for video {path}")
                    continue

                tensor = tensor.to(dtype=torch.float32, device='cuda', non_blocking=True)

                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"Warning: Tensor contains NaN or Inf AFTER casting to float32 for video {path}")
                    print(f"Tensor min: {tensor.min()}, max: {tensor.max()}, mean: {tensor.mean()}")
                    continue

                default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
                tensor = [tensor]
                video_timestamps = [video_timestamps]
                heads = [1]
                modal_list = ['video']

                question = default_mm_token + "\n" + question
                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                cur_prompt = conv.get_prompt()
                cur_prompt += '<sync>'

                input_ids = tokenizer_MMODAL_token_all(cur_prompt, tokenizer, return_tensors='pt').unsqueeze(0).to('cuda')
                attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

                stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE] else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                do_sample = False

                print("Starting generation...")
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
                        # stopping_criteria=[stopping_criteria],
                        pad_token_id=tokenizer.eos_token_id,
                        video_timestamps=video_timestamps,
                        heads=heads
                    )
                print("Generation finished")

                parsed_timestamps = []
                parsed_captions = []

                cur_timestamps = []
                cur_timestamp = []
                cur_caption = []

                for idx in output_ids[0]:
                    if idx <= 32000:
                        if idx == 32000:
                            new_caption = tokenizer.decode(cur_caption, skip_special_tokens=True)
                            parsed_captions.append(new_caption)
                            cur_caption = []
                            if stop_str in new_caption:
                                break
                        else:
                            cur_caption.append(idx)
                    elif idx <= 32013:  # 32001 <sync>; 32002 <sep>
                        if idx == 32001:
                            if len(cur_timestamp) > 0:
                                cur_timestamps.append(float(''.join(cur_timestamp)))
                            parsed_timestamps.append(cur_timestamps)
                            cur_timestamps = []
                            cur_timestamp = []
                        elif idx == 32002:
                            if len(cur_timestamp) > 0:
                                cur_timestamps.append(float(''.join(cur_timestamp)))
                            cur_timestamp = []
                        else:
                            cur_timestamp.append(model.get_model().time_tokenizer.decode(idx - 32001))
                    else:
                        # Skip score tokens.
                        pass

                if len(cur_caption):
                    parsed_captions.append(tokenizer.decode(cur_caption, skip_special_tokens=True))

                # Build the output for this sample.
                final_output = {}

                # Recover the original video name from batch_vnames[b]; fall back to `path`.
                # `video_path` is taken from the JSON-provided relative path (image_id) when present,
                # otherwise from the absolute `path`.
                vname = batch_vnames[b] if b < len(batch_vnames) else None

                # Output field for downstream consumers.
                final_output['video_path'] = vname if vname is not None else path

                pred_annotations = []
                num_preds = min(len(parsed_timestamps), len(parsed_captions))

                for k in range(num_preds):
                    ts = parsed_timestamps[k]
                    cap = parsed_captions[k]
                    if len(ts) >= 2:
                        st, ed = ts[0], ts[1]
                        if st >= ed:
                            continue

                        anno = {
                            "segment": [st, ed],
                            "obj_cot": [
                                {
                                    "obj": "t",
                                    "bnd_class": "",
                                    "bnd_sub_class": "",
                                    "obj_caption": cap
                                }
                            ],
                            "bnd_cot_st": [],
                            "bnd_cot_ed": []
                        }
                        pred_annotations.append(anno)

                final_output['annotations'] = pred_annotations
                final_output['type'] = 'fake' if len(pred_annotations) > 0 else 'real'

                results.append(final_output)

            except Exception:
                traceback.print_exc()
                print(f'generate for video {path} failed')
                break

    save_result(args, args.output_dir, results, args.split, format=True)

    total_time = time.time() - eval_start_time
    # convert seconds to date
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluate time {}'.format(total_time_str))

    # with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
    #     f.write(json.dumps(cfg.to_dict(), indent=4) + "\n")
    #     f.write(message + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_path', type=str, default='data/YouCook2-BB/YouCook2_asr_denseCap/')
    parser.add_argument('--video_path', type=str, default='data/YouCook2-BB/YouCook2_asr_denseCap/youcook2_6fps_224/')
    parser.add_argument('--task',
                        default='dvc')  # dvc for dense video captioning; tvg for temporal video grounding; vhd for video highlight detection
    parser.add_argument('--dataset', default='youcook')
    parser.add_argument('--output_dir', default='debug')
    parser.add_argument('--split', default='val')
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--timestamp', action='store_true', help='input the gt/predicted timestamps to the model')
    parser.add_argument('--timestamp_file', type=str, default='', help='the predcited timestamps file')
    parser.add_argument('--debug', action='store_true', help='the debug mode will only use 10 data samples')
    parser.add_argument('--prompt_file', default='prompts/dvc_description.txt')
    parser.add_argument('--model_path',
                        default='ckpt/vtgllm/train_stage2_llama2_7b_time64k_valley72k_bz32_f96_epoch3_open_i_instruct_qformer_lora_bind_time_ws32_mfp96_mtl2048/20231026060/checkpoint_2.pth')
    parser.add_argument('--sample_num', type=int, default=-1, help='fast inference by sampling N instances to evaluate')
    parser.add_argument('--num_chunks', type=int, default=1)
    parser.add_argument('--chunk_idx', type=int, default=0)
    parser.add_argument('--anno_file', type=str, default=None, help='Direct path to annotation file')
    parser.add_argument('--quiet_non_master', action='store_true', help='Only chunk0 prints normal logs (progress bars still show)')
    parser.add_argument('--tqdm_position', type=int, default=-1, help='tqdm position for multi-progress display; default uses chunk_idx')
    args = parser.parse_args()
    main(args)


