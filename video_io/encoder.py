"""Video frame encoding using OpenCV."""

from __future__ import annotations

import os
import sys
from typing import List

import cv2
from tqdm import tqdm

from models import FrameEntry


def encode_frames(frames: List[FrameEntry], output_path: str, fps: int) -> None:
    """Write a list of FrameEntry objects to a video file.

    Frames are written in ascending ``frame_id`` order using OpenCV's VideoWriter.
    The prototype uses the mp4v codec (no hardware encoding dependency).

    Args:
        frames: List of :class:`~models.FrameEntry` objects to encode.
        output_path: Path where the output video file will be written.
        fps: Frames per second for the output video.

    Raises:
        RuntimeError: If no frames are provided or if VideoWriter initialization fails.
        IOError: If the output directory does not exist or is not writable.
    """
    if not frames:
        raise RuntimeError("No frames provided for encoding.")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Sort by frame_id to ensure correct order
    sorted_frames = sorted(frames, key=lambda f: f.frame_id)

    # Get dimensions from the first frame
    first_frame = sorted_frames[0].data
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to initialize VideoWriter for: {output_path!r}")

    with tqdm(total=len(sorted_frames), desc="Encoding frames", unit="frame") as pbar:
        for entry in sorted_frames:
            writer.write(entry.data)
            pbar.update(1)

    writer.release()

    duration = len(sorted_frames) / fps if fps > 0 else 0.0
    print(
        f"\nEncoded {len(sorted_frames)} frames to {output_path!r} | "
        f"Duration: {duration:.3f}s"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m video_io.encoder <video_path>")
        sys.exit(1)

    from video_io.extractor import extract_frames

    input_path = sys.argv[1]
    output_path = "/tmp/reencoded.mp4"

    frames = extract_frames(input_path)
    fps = 30  # Default fps for smoke test

    encode_frames(frames, output_path, fps)

    if os.path.exists(output_path):
        print("Round-trip OK")
    else:
        print(f"Round-trip FAILED: output file not found at {output_path!r}")
