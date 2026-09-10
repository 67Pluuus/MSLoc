import copy
import os
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    print('Using NPU')
except:
    # print(e)
    print('Using GPU')
import argparse
import traceback

import sys
# Adjust paths for Tasle_Model
current_file_path = os.path.abspath(__file__)
# .../trace/eval/evaluate_ref.py -> .../trace/eval -> .../trace -> .../Tasle_Model
trace_eval_dir = os.path.dirname(current_file_path)
trace_dir = os.path.dirname(trace_eval_dir)
tasle_model_inner_dir = os.path.dirname(trace_dir)
tasle_model_outer_dir = os.path.dirname(tasle_model_inner_dir)

if tasle_model_outer_dir not in sys.path:
    sys.path.append(tasle_model_outer_dir)
if tasle_model_inner_dir not in sys.path:
    sys.path.append(tasle_model_inner_dir)
if trace_dir not in sys.path:
    sys.path.append(trace_dir)
sys.path.append('./')

from trace.conversation import conv_templates
from trace.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from trace.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, process_video, process_image, process_video_ref_split
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

    # 兼容三类输入：
    # 1) {"annotations": [...]}  (COCO风格)
    # 2) [...]                  (直接为annotations数组)
    # 3) 训练数据格式 (包含video字段而非image_id)
    if isinstance(raw, dict):
        if "video_path" in raw or "video" in raw:
            data = [raw]
        else:
            data = raw.get("annotations", [])
    elif isinstance(raw, list):
        data = raw
    else:
        raise TypeError(f"Unsupported annotation json root type: {type(raw)}")

    # 转换训练数据格式/原始格式为评测格式
    # 兼容 video_path 字段 (用户自定义格式) 以及 video 字段 (训练格式)
    if data and isinstance(data[0], dict):
        needs_conversion = False
        if "image_id" not in data[0]:
            if "video" in data[0] or "video_path" in data[0]:
                needs_conversion = True
        
        if needs_conversion:
            print("Detected training/raw data format, converting to evaluation format...")
            converted_data = []
            for idx, item in enumerate(data):
                # 复制原始 item 以保留所有字段，确保输出时能获取到 source, duration 等
                converted_item = item.copy()
                
                # 优先取 video_path，其次 video
                vid = item.get("video_path") or item.get("video")
                
                # 确保有 id，如果没有则使用索引
                item_id = item.get("id")
                if item_id is None:
                    item_id = idx

                # 补充模型需要的字段
                converted_item["image_id"] = vid
                converted_item["id"] = item_id
                if "caption" not in converted_item:
                    converted_item["caption"] = ""  # 推理时不需要caption
                
                # 如果有times字段，可以添加segments
                if "times" in item and item["times"]:
                    converted_item["segments"] = item["times"]
                elif "segments" in item:
                    converted_item["segments"] = item["segments"]
                
                converted_data.append(converted_item)
            data = converted_data
            print(f"Converted {len(data)} samples from raw format to evaluation format")

    # Filter for 'fake' type
    # filtered_data = []
    # for item in data:
    #     if item.get('type') == 'fake':
    #         filtered_data.append(item)
    
    # print(f"Filtered {len(data)} -> {len(filtered_data)} samples (type='fake')")
    # data = filtered_data

    if args.debug:
        data = data[:10]
    return data


def save_result(args, output_dir, results, split_name='test', format=False):
    """保存结果：直接保存 results 列表，因为 main 函数已经构建好了最终格式。"""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 输出文件名
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

    # 直接保存 results 列表
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

    # quiet控制：仅在chunk0输出普通日志
    is_master = (int(getattr(args, 'chunk_idx', 0)) == 0)
    if getattr(args, 'quiet_non_master', False) and not is_master:
        import builtins
        _orig_print = builtins.print

        def _quiet_print(*p_args, **p_kwargs):
            # 允许显式force打印
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
    load_kwargs = {}
    if args.closs is not None:
        load_kwargs['closs'] = args.closs
    
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, None, model_name, **load_kwargs)
    model = model.to(device)
    model.to(dtype=torch.float32)
    
    # Check and print closs status
    closs_status = getattr(model.config, 'closs', False)
    has_closs_tokens = hasattr(model, 'closs_tokens') or hasattr(model.get_model(), 'closs_tokens')
    print(f"Model Config CLOSS: {closs_status}")
    print(f"Model has closs_tokens: {has_closs_tokens}")
    
    # conv_mode = 'mistral_instruct'
    conv_mode = 'llama_2'
    # conv_mode = 'v1'
    print('Initialization Finished')

    # load data
    video_path = args.video_path
    anno_path = args.anno_path
    anno_data = load_data(args, anno_path, split=args.split)
    print('Load Annotation Finished')

    # Deduplicate by video path to avoid redundant inference
    unique_data = {}
    # 保持原始顺序
    unique_list = []
    seen_vids = set()
    
    print(f"Before deduplication: {len(anno_data)} samples")
    
    for item in anno_data:
        # 优先使用 video_path 或 video 作为去重键
        # 如果没有，再尝试使用 image_id (如果它看起来像文件路径)
        vid = item.get('video_path') or item.get('video')
        
        if not vid:
            iid = item.get('image_id')
            if iid and isinstance(iid, str) and ('.mp4' in iid or '.avi' in iid or '/' in iid):
                vid = iid
            else:
                # 如果 image_id 也是流水号，可能无法正确去重，但在没有 video_path 的情况下只能如此
                vid = iid
        
        # 确保 vid 存在且未处理过
        if vid:
            if vid not in seen_vids:
                seen_vids.add(vid)
                unique_list.append(item)
        else:
            # 极其罕见的情况：没有 video_path 也没有 image_id，保留以防万一
            unique_list.append(item)
    
    if len(unique_list) < len(anno_data):
        print(f"Deduplicated data: {len(anno_data)} -> {len(unique_list)} unique videos")
        anno_data = unique_list
    else:
        print(f"No duplicates found (based on video path). Total: {len(anno_data)}")

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

    results = []
    
    # tqdm并存显示：每个chunk用不同position
    tqdm_position = int(getattr(args, 'tqdm_position', -1))
    if tqdm_position < 0:
        tqdm_position = int(getattr(args, 'chunk_idx', 0))

    for item in tqdm(anno_data, position=tqdm_position, leave=True, dynamic_ncols=True):
        # if item.get('type') != 'fake':
        #     results.append(item)
        #     continue

        vname = item.get("image_id")
        # 尝试构建完整路径
        vid_path = os.path.join(video_path, vname)
        if not os.path.exists(vid_path):
            # 尝试直接使用 video_path 字段（如果是绝对路径）
            if os.path.exists(item.get("video_path", "")):
                vid_path = item.get("video_path")
            else:
                # 尝试拼接 video_path
                vid_path = os.path.join(video_path, item.get("video_path", ""))
                if not os.path.exists(vid_path):
                    print(f"Video not found: {vname} / {item.get('video_path')}")
                    continue

        duration = item.get("duration")
        # 如果 duration 缺失，尝试读取
        if duration is None:
            try:
                vr = decord.VideoReader(vid_path, ctx=decord.cpu(0))
                duration = len(vr) / vr.get_avg_fps()
            except Exception as e:
                print(f"Failed to read duration for {vid_path}: {e}")
                duration = 10000.0 # Fallback

        # 确定待微调的 segments
        # 优先使用 model_inference 中的 segment
        # 如果不存在，回退到 annotations (兼容原始数据输入)
        source_segments = []
        if "model_inference" in item and "segment" in item["model_inference"]:
            source_segments = item["model_inference"]["segment"]
        else:
            print(item)
            exit()
        # elif "annotations" in item:
        #     for ann in item["annotations"]:
        #         if "segment" in ann:
        #             source_segments.append(ann["segment"])
        # elif "segments" in item:
        #     source_segments = item["segments"]

        # print(item)
        # exit()
        
        if not source_segments:
            # print(f"No segments found for {item.get('image_id', 'unknown')}")
            # 如果没有 segment，可能是没有检测到任何 proposal，此时应输出空结果或保留原样
            pass
        
        # 如果是 Real 数据且没有 segment，通常不需要处理，但为了保持逻辑一致性，
        
        # 如果是 Real 数据且没有 segment，通常不需要处理，但为了保持逻辑一致性，
        # 如果 source_segments 为空且是 fake 数据，可能意味着 Loc 没检测到，Ref 也无法微调。
        # 如果是 real 数据，source_segments 通常为空。
        
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
            
            # 确保窗口有效
            if win_e <= win_s:
                print(f"Invalid window for {vid_path}: {win_s}-{win_e}")
                model_inference_segments.append([-99, -99])
                continue

            try:
                # print(f"Processing video: {vid_path} window: {win_s}-{win_e}")
                
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

                tensor = tensor.to(dtype=torch.float32, device='cuda', non_blocking=True)

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
                # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
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
                             if last_timestamps is not None:
                                 parsed_segments.append({"timestamp": last_timestamps, "caption": full_caption})
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
                    if last_timestamps is not None:
                        parsed_segments.append({"timestamp": last_timestamps, "caption": full_caption})
                elif last_timestamps is not None:
                    parsed_segments.append({"timestamp": last_timestamps, "caption": ""})

                # 解析结果
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
                    # 记录失败或无预测，但在最终输出时会过滤
                    model_inference_segments.append([-99, -99])
                    model_inference_responses.append("")
                
                print(win_s, win_e, abs_s, abs_e)

                # if abs_s != -99 and abs_e != -99:
                #     if abs_e - abs_s < 1:
                #         model_inference_segments.append([win_s, win_e])
                #     else:
                #         model_inference_segments.append([abs_s, abs_e])
                # else:
                #     # 记录失败或无预测，但在最终输出时会过滤
                #     model_inference_segments.append([win_s, win_e])
                #     # model_inference_segments.append([-99, -99])
                
                print(win_s, win_e, abs_s, abs_e)
            except Exception:
                traceback.print_exc()
                print(f'generate for video {vid_path} segment {segment} failed')
                model_inference_segments.append([-99, -99])

        # 构建输出 item
        final_output = item.copy()
        # 保持原始 annotations 不变
        final_output['annotations'] = item.get("annotations", [])
        
        # 过滤无效片段
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

        # 确保 video_path 存在
        if 'video_path' not in final_output:
             final_output['video_path'] = item.get("image_id")
        
        results.append(final_output)

    save_result(args, args.output_dir, results, args.split, format=True)

    total_time = time.time() - eval_start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluate time {}'.format(total_time_str))


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
    
    # 新增参数
    parser.add_argument('--bnd_ratio', type=float, default=0.2, help='Ratio of boundary region (each side).')
    parser.add_argument('--bnd_frames', type=int, default=16, help='Number of frames to sample from each boundary region.')
    parser.add_argument('--seg_frames', type=int, default=8, help='Number of frames to sample from the event region.')
    parser.add_argument('--closs', type=lambda x: (str(x).lower() == 'true'), default=None, help='Whether to use closs tokens')

    args = parser.parse_args()
    main(args)


