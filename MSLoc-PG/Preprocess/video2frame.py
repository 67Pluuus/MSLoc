import os
import argparse
from glob import glob
import fnmatch
from moviepy.editor import VideoFileClip
import multiprocessing
import math
import subprocess

# Defaults are RELATIVE to the MSLoc repository root.
# You can also pass --input_root / --output_root from the CLI.
DEFAULT_INPUT_ROOT = './data/Tasle-CoT-10K/videos'
DEFAULT_OUTPUT_ROOT = './data/Tasle-CoT-10K/video_frames'

INPUT_ROOT = DEFAULT_INPUT_ROOT
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT

def find_video_files(root_dir, extensions=['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']):
    """Recursively collect every video file under root_dir."""
    video_files = []
    for extension in extensions:
        pattern = f"**/{extension}" if os.name != 'nt' else f"**\\{extension}"
        video_files.extend(glob(os.path.join(root_dir, pattern), recursive=True))
    return video_files

def get_video_length(file_path):
    video = VideoFileClip(file_path)
    try:
        return video.duration
    finally:
        video.close()

def process_video(video_path):
    # Video name without extension.
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Path of `video_path` relative to INPUT_ROOT.
    relative_path = os.path.relpath(os.path.dirname(video_path), INPUT_ROOT)

    # Mirror the directory structure under OUTPUT_ROOT.
    image_path = os.path.join(OUTPUT_ROOT, relative_path, video_name)

    # Skip if already processed.
    if os.path.exists(image_path) and len(os.listdir(image_path)) > 0:
        return

    os.makedirs(image_path, exist_ok=True)

    video_length = get_video_length(video_path)
    if video_length is None or not math.isfinite(video_length) or video_length <= 0:
        raise ValueError(f"Invalid video duration: {video_path}")

    inter_val = 8
    ffmpeg_cmd = [
        'ffmpeg', '-nostdin', '-loglevel', 'error', '-i', video_path,
        '-r', str(inter_val), os.path.join(image_path, 'frame_%d.jpg'),
    ]
    result = subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path} (exit code {result.returncode}):\n{result.stderr}"
        )
    print(f"Frames extracted: {video_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=str, default=DEFAULT_INPUT_ROOT,
                        help='Root directory of input videos (relative or absolute).')
    parser.add_argument('--output_root', type=str, default=DEFAULT_OUTPUT_ROOT,
                        help='Root directory to save extracted frames.')
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    INPUT_ROOT = os.path.abspath(args.input_root)
    OUTPUT_ROOT = os.path.abspath(args.output_root)
    if not os.path.isdir(INPUT_ROOT):
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_ROOT}")

    print(f"Extracting video frames... input={INPUT_ROOT} output={OUTPUT_ROOT}")

    video_files = find_video_files(INPUT_ROOT)
    if not video_files:
        raise ValueError(f"No videos found in {INPUT_ROOT}")
    print(f"Found {len(video_files)} video file(s)")

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        pool.map(process_video, video_files)
    print("All videos processed.")

