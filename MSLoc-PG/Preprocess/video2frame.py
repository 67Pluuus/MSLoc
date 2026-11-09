"""Extract the released DeMamba 8-fps JPEG input, with per-video resume."""
import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio_ffmpeg
from tqdm import tqdm


def extract_video(task):
    source, destination, ffmpeg, clean = task
    source, destination = Path(source), Path(destination)
    stat = source.stat()
    identity = dict(source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns, fps=8)
    marker = destination / 'complete.json'
    if marker.is_file() and not clean:
        saved = json.loads(marker.read_text())
        files = list(destination.glob('frame_*.jpg'))
        if all(saved[k] == v for k, v in identity.items()) and len(files) == saved['frames']:
            if all((destination / f'frame_{i}.jpg').stat().st_size > 0
                   for i in range(1, saved['frames'] + 1)):
                return 'reused', saved['frames'], saved['bytes']
        raise ValueError(f'Changed source or incomplete committed frames: {destination}; use --clean')
    destination.mkdir(parents=True, exist_ok=True)
    # Remove only extractor-owned files for this exact video.
    for path in destination.glob('frame_*.jpg'):
        path.unlink()
    marker.unlink(missing_ok=True)
    result = subprocess.run(
        [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
         '-threads', '1', '-i', str(source), '-r', '8', '-threads', '1',
         str(destination / 'frame_%d.jpg')],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode:
        raise RuntimeError(f'Extraction failed for {source}: {result.stderr}')
    files = list(destination.glob('frame_*.jpg'))
    if not files:
        raise ValueError(f'No frames decoded: {source}')
    saved = dict(identity, frames=len(files), bytes=sum(p.stat().st_size for p in files))
    temporary = marker.with_suffix('.tmp')
    temporary.write_text(json.dumps(saved), encoding='utf-8')
    os.replace(temporary, marker)
    return 'extracted', saved['frames'], saved['bytes']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_root', type=Path, required=True)
    parser.add_argument('--output_root', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, nargs='+', required=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max-videos', type=int, default=0)
    parser.add_argument('--clean', action='store_true')
    args = parser.parse_args()
    if args.num_workers <= 0 or args.max_videos < 0:
        parser.error('Workers must be positive and video limit nonnegative')
    input_root, output_root = args.input_root.resolve(), args.output_root.resolve()
    records = {}
    for annotation in args.annotations:
        for item in json.loads(annotation.read_text(encoding='utf-8')):
            records[item['video_path']] = item
    items = list(records.values())
    if args.max_videos:
        step = max(1, len(items) // args.max_videos)
        items = items[::step][:args.max_videos]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tasks = []
    for item in items:
        relative = Path(item['video_path'])
        source = (input_root / relative).resolve()
        destination = (output_root / relative.with_suffix('')).resolve()
        if not source.is_relative_to(input_root) or not destination.is_relative_to(output_root):
            raise ValueError(f'Path escapes dataset root: {relative}')
        if not source.is_file():
            raise FileNotFoundError(source)
        tasks.append((str(source), str(destination), ffmpeg, args.clean))
    output_root.mkdir(parents=True, exist_ok=True)
    print(f'Videos={len(tasks)}, free disk={shutil.disk_usage(output_root).free / 2**30:.2f} GiB', flush=True)
    counts = dict(reused=0, extracted=0, frames=0, bytes=0)
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        with tqdm(pool.map(extract_video, tasks), total=len(tasks), desc='Extract 8-fps frames') as bar:
            for status, frames, size in bar:
                counts[status] += 1
                counts['frames'] += frames
                counts['bytes'] += size
                bar.set_postfix(GiB=f"{counts['bytes'] / 2**30:.2f}")
    seconds = sum(item['duration'] for item in items)
    print(json.dumps(dict(counts, seconds=seconds,
                         estimated_full_GiB=counts['bytes'] / seconds * sum(x['duration'] for x in records.values()) / 2**30), indent=2))


if __name__ == '__main__':
    main()
