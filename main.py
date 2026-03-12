"""Video quality correction pipeline CLI."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

from config import PipelineConfig
from correction import run_correction
from detection import run_detection, save_detection_report
from video_io import encode_frames, extract_frames


def load_config_from_json(path: str) -> Dict[str, Any]:
    """Load config overrides from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def main() -> int:
    """Run the video quality correction pipeline."""
    parser = argparse.ArgumentParser(
        description="Video quality correction pipeline for frozen frames, drops, and artifacts."
    )
    parser.add_argument("--input", required=True, help="Path to input MP4 video")
    parser.add_argument("--output", required=True, help="Path to output MP4 video")
    parser.add_argument("--config", default=None, help="Optional JSON config file to override defaults")
    parser.add_argument("--skip-frozen", action="store_true", help="Skip frozen frame correction")
    parser.add_argument("--skip-drops", action="store_true", help="Skip drop correction")
    parser.add_argument("--skip-artifacts", action="store_true", help="Skip artifact correction")
    parser.add_argument(
        "--detection-only",
        action="store_true",
        help="Run detection and save report, skip correction and encoding",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input!r}", file=sys.stderr)
        return 1

    # Record overall pipeline start
    pipeline_start = time.perf_counter()

    # Stage 1: Load config
    print("=" * 60)
    print("STAGE 1: Loading configuration")
    print("=" * 60)
    t0 = time.perf_counter()

    cfg_overrides: Dict[str, Any] = {}
    if args.config:
        print(f"Loading config overrides from: {args.config!r}")
        cfg_overrides = load_config_from_json(args.config)

    cfg = PipelineConfig(
        input_path=args.input,
        output_path=args.output,
        fps=cfg_overrides.get("fps", 30),
        frozen_hash_threshold=cfg_overrides.get("frozen_hash_threshold", 5),
        frozen_min_frames=cfg_overrides.get("frozen_min_frames", 3),
        drop_pts_tolerance=cfg_overrides.get("drop_pts_tolerance", 0.5),
        artifact_block_threshold=cfg_overrides.get("artifact_block_threshold", 1.8),
        drop_short_max_frames=cfg_overrides.get("drop_short_max_frames", 5),
        film_model_path=cfg_overrides.get("film_model_path", "weights/film_net.pt"),
        ifrnet_model_path=cfg_overrides.get("ifrnet_model_path", "weights/ifrnet_l.pth"),
        fbcnn_model_path=cfg_overrides.get("fbcnn_model_path", "weights/fbcnn_color.pth"),
        device=cfg_overrides.get("device", "cuda"),
        half_precision=cfg_overrides.get("half_precision", True),
    )

    config_time = time.perf_counter() - t0
    print(f"Configuration loaded in {config_time:.2f}s\n")

    # Stage 2: Extract frames
    print("=" * 60)
    print("STAGE 2: Extracting frames from input video")
    print("=" * 60)
    t0 = time.perf_counter()

    frames = extract_frames(cfg.input_path)

    extraction_time = time.perf_counter() - t0
    print(f"Extraction completed in {extraction_time:.2f}s\n")

    # Stage 3: Run detection
    print("=" * 60)
    print("STAGE 3: Running detection")
    print("=" * 60)
    t0 = time.perf_counter()

    segments = run_detection(frames, cfg)

    detection_time = time.perf_counter() - t0
    print(f"Detection completed in {detection_time:.2f}s\n")

    # Save detection report
    save_detection_report(segments, len(frames), cfg.output_path)

    if args.detection_only:
        print("=" * 60)
        print("DETECTION-ONLY MODE: Skipping correction and encoding")
        print("=" * 60)
        total_time = time.perf_counter() - pipeline_start
        print(f"\nTotal pipeline time: {total_time:.2f}s")
        return 0

    # Stage 4: Filter segments based on skip flags
    print("=" * 60)
    print("STAGE 4: Filtering segments")
    print("=" * 60)

    from models import ProblemType

    original_count = len(segments)

    if args.skip_frozen:
        segments = [s for s in segments if s.problem_type != ProblemType.FROZEN]
        print("Skipping FROZEN frame correction")

    if args.skip_drops:
        segments = [s for s in segments if s.problem_type != ProblemType.DROP]
        print("Skipping DROP frame correction")

    if args.skip_artifacts:
        segments = [s for s in segments if s.problem_type != ProblemType.ARTIFACT]
        print("Skipping ARTIFACT frame correction")

    filtered_count = len(segments)
    print(f"Segments to correct: {filtered_count} (filtered from {original_count})\n")

    if not segments:
        print("No segments to correct. Proceeding directly to encoding.\n")
        corrected_frames = frames
        correction_time = 0.0
    else:
        # Stage 5: Run correction
        print("=" * 60)
        print("STAGE 5: Running correction")
        print("=" * 60)
        t0 = time.perf_counter()

        corrected_frames = run_correction(frames, segments, cfg)

        correction_time = time.perf_counter() - t0
        print(f"Correction completed in {correction_time:.2f}s\n")

    # Stage 6: Encode output video
    print("=" * 60)
    print("STAGE 6: Encoding output video")
    print("=" * 60)
    t0 = time.perf_counter()

    encode_frames(corrected_frames, cfg.output_path, cfg.fps)

    encoding_time = time.perf_counter() - t0
    print(f"Encoding completed in {encoding_time:.2f}s\n")

    # Total wall-clock time
    total_pipeline_time = time.perf_counter() - pipeline_start
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Config:       {config_time:.2f}s")
    print(f"  Extraction:   {extraction_time:.2f}s")
    print(f"  Detection:    {detection_time:.2f}s")
    print(f"  Correction:   {correction_time:.2f}s")
    print(f"  Encoding:     {encoding_time:.2f}s")
    print(f"  ─────────────────────────────────")
    print(f"  TOTAL:        {total_pipeline_time:.2f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
