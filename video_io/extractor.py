"""Video frame extraction using OpenCV."""

from __future__ import annotations

import sys
from typing import List

import cv2
from tqdm import tqdm

from models import FrameEntry, FrameStatus, ProblemType


def extract_frames(video_path: str) -> List[FrameEntry]:
    """Read a video file and return all frames as a sorted list of FrameEntry objects.

    Each frame is stored as a raw BGR uint8 numpy array (no compression).
    PTS is computed as ``frame_index / fps`` in seconds.

    Args:
        video_path: Absolute or relative path to the input video file.

    Returns:
        A list of :class:`~models.FrameEntry` objects sorted ascending by
        ``frame_id``, each with ``status=ORIGINAL`` and ``problem=NONE``.

    Raises:
        RuntimeError: If the video file cannot be opened by OpenCV.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path!r}")

    fps: float = cap.get(cv2.CAP_PROP_FPS) or 1.0
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration: float = total_frames / fps if fps > 0 else 0.0

    frames: List[FrameEntry] = []
    frame_index = 0

    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            pts = frame_index / fps
            entry = FrameEntry(
                frame_id=frame_index,
                pts=pts,
                data=bgr,
                status=FrameStatus.ORIGINAL,
                problem=ProblemType.NONE,
            )
            frames.append(entry)
            frame_index += 1
            pbar.update(1)

    cap.release()

    print(
        f"\nExtracted {len(frames)} frames | "
        f"FPS: {fps:.3f} | "
        f"Duration: {duration:.3f}s | "
        f"Resolution: {width}x{height}"
    )

    # Ensure ordering by frame_id (should already be sorted, but be explicit)
    frames.sort(key=lambda f: f.frame_id)
    return frames


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m io.extractor <video_path>")
        sys.exit(1)

    extracted = extract_frames(sys.argv[1])

    print("\nFirst 3 frames:")
    for frame in extracted[:3]:
        print(f"  frame_id={frame.frame_id}  pts={frame.pts:.6f}s")
