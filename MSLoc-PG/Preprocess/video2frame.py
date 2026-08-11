import argparse
import math
import multiprocessing
import subprocess
from functools import partial
from pathlib import Path

from moviepy.editor import VideoFileClip


DEFAULT_INPUT_ROOT = "../../MSLoc_assets/data/Tasle-CoT-10K/videos"
DEFAULT_OUTPUT_ROOT = "../../MSLoc_assets/data/Tasle-CoT-10K/video_frames"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


def find_video_files(root_dir: Path):
    return sorted(
        path for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def get_video_length(file_path: Path) -> float:
    try:
        with VideoFileClip(str(file_path)) as video:
            return float(video.duration)
    except Exception as exc:
        print(f"Failed to read duration for {file_path}: {exc}")
        return 0.0


def frame_files(image_path: Path):
    return list(image_path.glob("frame_*.jpg")) if image_path.exists() else []


def process_video(video_path: Path, input_root: Path, output_root: Path, fps: float, force: bool):
    """Extract one video with absolute paths and verify the resulting frame count."""
    relative_video = video_path.relative_to(input_root)
    image_path = output_root / relative_video.with_suffix("")
    image_path.mkdir(parents=True, exist_ok=True)

    duration = get_video_length(video_path)
    if duration <= 0:
        return "failed", f"invalid video: {video_path}"

    expected_frames = max(1, int(round(duration * fps)))
    existing_frames = frame_files(image_path)
    if not force and abs(len(existing_frames) - expected_frames) <= 1:
        return "skipped", str(video_path)

    for frame_path in existing_frames:
        frame_path.unlink()

    output_pattern = image_path / "frame_%d.jpg"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps:g}",
        str(output_pattern),
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return "failed", f"{video_path}: {exc}"

    extracted_frames = len(frame_files(image_path))
    if abs(extracted_frames - expected_frames) > 1:
        return (
            "failed",
            f"{video_path}: expected about {expected_frames} frames, found {extracted_frames}",
        )
    return "extracted", str(video_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", default=DEFAULT_INPUT_ROOT,
                        help="Root directory of input videos (relative or absolute).")
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT,
                        help="Root directory for extracted frames (relative or absolute).")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--force", action="store_true",
                        help="Re-extract videos even when the existing frame count is complete.")
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not input_root.is_dir():
        raise SystemExit(f"Input root does not exist: {input_root}")
    if args.num_workers < 1:
        raise SystemExit("--num_workers must be at least 1")
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise SystemExit("--fps must be a positive finite number")
    output_root.mkdir(parents=True, exist_ok=True)

    video_files = find_video_files(input_root)
    print(f"Extracting at {args.fps:g} fps")
    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    print(f"Found {len(video_files)} video file(s)")

    worker = partial(
        process_video,
        input_root=input_root,
        output_root=output_root,
        fps=args.fps,
        force=args.force,
    )
    counts = {"extracted": 0, "skipped": 0, "failed": 0}
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for status, message in pool.imap_unordered(worker, video_files):
            counts[status] += 1
            if status == "failed":
                print(f"Failed: {message}")

    print(
        "Finished: "
        f"{counts['extracted']} extracted, {counts['skipped']} skipped, "
        f"{counts['failed']} failed"
    )
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
