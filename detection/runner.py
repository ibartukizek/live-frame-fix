"""Detection runner: orchestrates all detectors and generates reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from config import PipelineConfig
from io import extract_frames
from models import FrameEntry, ProblemSegment, ProblemType

from .artifact_detector import ArtifactDetector
from .drop_detector import DropDetector
from .frozen_detector import FrozenDetector


def run_detection(frames: List[FrameEntry], cfg: PipelineConfig) -> List[ProblemSegment]:
    """Run all quality detectors on a frame list and return sorted problems.

    Args:
        frames: List of :class:`~models.FrameEntry` objects in frame_id order.
        cfg: :class:`~config.PipelineConfig` with detection parameters.

    Returns:
        Sorted list of :class:`~models.ProblemSegment` objects, ordered by
        start_frame_id.
    """
    # Instantiate detectors with config parameters
    frozen_detector = FrozenDetector(
        hash_threshold=cfg.frozen_hash_threshold,
        min_duration=cfg.frozen_min_frames,
    )
    drop_detector = DropDetector(
        fps=cfg.fps,
        tolerance=cfg.drop_pts_tolerance,
    )
    artifact_detector = ArtifactDetector(
        block_ratio_threshold=cfg.artifact_block_threshold,
        block_size=8,
    )

    segments: List[ProblemSegment] = []

    # Run all detectors on each frame
    with tqdm(total=len(frames), desc="Running detection", unit="frame") as pbar:
        for entry in frames:
            # FrozenDetector
            seg = frozen_detector.feed(entry)
            if seg is not None:
                segments.append(seg)

            # DropDetector
            seg = drop_detector.feed(entry)
            if seg is not None:
                segments.append(seg)

            # ArtifactDetector
            seg = artifact_detector.feed(entry)
            if seg is not None:
                segments.append(seg)

            pbar.update(1)

    # Handle end-of-stream for FrozenDetector
    seg = frozen_detector.finalize()
    if seg is not None:
        segments.append(seg)

    # Sort by start_frame_id
    segments.sort(key=lambda s: s.start_frame_id)

    # Compute summary statistics
    total_frames = len(frames)
    frozen_segs = [s for s in segments if s.problem_type == ProblemType.FROZEN]
    drop_segs = [s for s in segments if s.problem_type == ProblemType.DROP]
    artifact_segs = [s for s in segments if s.problem_type == ProblemType.ARTIFACT]

    frozen_frames = sum(s.end_frame_id - s.start_frame_id + 1 for s in frozen_segs)
    drop_frames = sum(s.end_frame_id - s.start_frame_id + 1 for s in drop_segs)
    artifact_count = len(artifact_segs)

    # Print summary table
    print(
        f"\n{'─' * 60}\n"
        f"Detection Summary\n"
        f"{'─' * 60}\n"
        f"Total frames scanned:  {total_frames}\n"
        f"Frozen segments:       {len(frozen_segs):3d}  (total frames: {frozen_frames})\n"
        f"Drop segments:         {len(drop_segs):3d}  (total frames: {drop_frames})\n"
        f"Artifact frames:       {artifact_count:3d}\n"
        f"{'─' * 60}\n"
    )

    return segments


def save_detection_report(
    segments: List[ProblemSegment],
    frames_count: int,
    output_path: str,
) -> None:
    """Save a detection report to JSON.

    Args:
        segments: List of detected problem segments.
        frames_count: Total frames scanned.
        output_path: Output video path (used to derive report filename).
    """
    # Compute statistics
    frozen_segs = [s for s in segments if s.problem_type == ProblemType.FROZEN]
    drop_segs = [s for s in segments if s.problem_type == ProblemType.DROP]
    artifact_segs = [s for s in segments if s.problem_type == ProblemType.ARTIFACT]

    frozen_frames = sum(s.end_frame_id - s.start_frame_id + 1 for s in frozen_segs)
    drop_frames = sum(s.end_frame_id - s.start_frame_id + 1 for s in drop_segs)

    report: Dict = {
        "summary": {
            "total_frames_scanned": frames_count,
            "frozen_segments": len(frozen_segs),
            "frozen_frames_affected": frozen_frames,
            "drop_segments": len(drop_segs),
            "drop_frames_missing": drop_frames,
            "artifact_frames": len(artifact_segs),
        },
        "segments": [
            {
                "problem_type": s.problem_type.value,
                "start_frame_id": s.start_frame_id,
                "end_frame_id": s.end_frame_id,
                "frame_a_id": s.frame_a_id,
                "frame_b_id": s.frame_b_id,
            }
            for s in segments
        ],
    }

    report_path = f"{output_path}.detection_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m detection.runner <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]

    # Create a minimal config for testing
    cfg = PipelineConfig(
        input_path=video_path,
        output_path="/tmp/out.mp4",
        fps=30,
        frozen_hash_threshold=5,
        frozen_min_frames=3,
        drop_pts_tolerance=0.5,
        artifact_block_threshold=1.8,
    )

    # Extract frames
    print(f"Extracting frames from {video_path!r}...")
    frames = extract_frames(video_path)

    # Run detection
    print("\nRunning detection...")
    segments = run_detection(frames, cfg)

    # Save report
    save_detection_report(segments, len(frames), cfg.output_path)

    print(f"\nDetected {len(segments)} problem segments total.")
