import json
import os
import argparse

def convert_aigc_to_coco(anno_file, output_dir, format='aigc'):
    with open(anno_file, 'r') as f:
        data = json.load(f)
    
    val_annotations = []
    
    if format == 'tasle':
        for idx, item in enumerate(data):
            video_path = item.get('video_path', '')
            duration = item.get('duration', 0)
            recipe_type = item.get('type', 'unknown')
            
            segments = []
            captions = []
            
            for anno in item.get('annotations', []):
                segments.append(anno.get('segment', []))
                
                # Extract caption from obj_cot
                obj_cot = anno.get('obj_cot', [])
                anno_caption = ""
                if obj_cot and len(obj_cot) > 0:
                    # Try to get obj_caption, fallback to obj
                    anno_caption = obj_cot[0].get('obj_caption', obj_cot[0].get('obj', ''))
                
                captions.append(anno_caption)
            
            entry = {
                "image_id": video_path,
                "duration": duration,
                "segments": segments,
                "pure_cap": ". ".join(captions),
                "caption": ". ".join(captions),
                "recipe_type": recipe_type,
                "id": idx
            }
            val_annotations.append(entry)
    else:
        database = data
        
        train_annotations = []
        
        # If database is a dict, iterate over items
        iterator = database.items() if isinstance(database, dict) else database
        
        for vid, info in iterator:
            duration = info['duration']
            recipe_type = info['type']
            
            # aigc annotations format:
            # "annotations": [
            #     {
            #         "segment": [start, end],
            #         "id": "...",
            #         "sentence": "..."
            #     }
            # ]
            
            segments = []
            captions = []
            
            for anno in info['annotations']:
                segments.append(anno['segment'])
                captions.append(anno['sentence'])
                
            entry = {
                "image_id": vid, # evaluate.py expects image_id to be the video name/id
                "duration": duration,
                "segments": segments,
                "pure_cap": ". ".join(captions), # evaluate.py uses pure_cap for aigc
                "caption": ". ".join(captions), # fallback
                "recipe_type": recipe_type
            }
            
            val_annotations.append(entry)
            
    # Save validation set
    val_output = {
        "annotations": val_annotations
    }
    val_path = os.path.join(output_dir, 'val.caption_coco_format.json')
    with open(val_path, 'w') as f:
        json.dump(val_output, f)
    print(f"Saved {len(val_annotations)} validation annotations to {val_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_file', type=str,
                        default='../../../data/Tasle-CoT-10K/annos/train_all_1209.json',
                        help='Annotation JSON file (relative to MSLoc/Trace/scripts/eval).')
    parser.add_argument('--output_dir', type=str,
                        default='./',
                        help='Where to write the converted COCO-format JSON.')
    parser.add_argument('--format', type=str, default='tasle', choices=['aigc', 'tasle'], help='Input annotation format')
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    convert_aigc_to_coco(args.anno_file, args.output_dir, args.format)


