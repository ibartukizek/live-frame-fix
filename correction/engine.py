"""Correction engine: applies models to fix detected problems."""

from __future__ import annotations

from typing import Dict, List

from tqdm import tqdm

from config import PipelineConfig
from models import FrameEntry, FrameStatus, ProblemSegment, ProblemType

from .fbcnn_reducer import FBCNNReducer
from .film_interpolator import FILMInterpolator
from .ifrnet_interpolator import IFRNetInterpolator


def run_correction(
    frames: List[FrameEntry],
    segments: List[ProblemSegment],
    cfg: PipelineConfig,
) -> List[FrameEntry]:
    """Apply correction models to fix detected problems.

    Args:
        frames: List of :class:`~models.FrameEntry` objects (may contain gaps).
        segments: List of :class:`~models.ProblemSegment` to correct.
        cfg: :class:`~config.PipelineConfig` with model paths and parameters.

    Returns:
        Updated list of :class:`~models.FrameEntry` objects with synthetic
        frames inserted and corrected data, sorted by ``frame_id``.
    """
    if not segments:
        print("No segments to correct.")
        return frames

    # Create a lookup map for frames by frame_id
    frame_map: Dict[int, FrameEntry] = {f.frame_id: f for f in frames}

    # Lazy model loading: only instantiate if needed
    film_model: FILMInterpolator | None = None
    ifrnet_model: IFRNetInterpolator | None = None
    fbcnn_model: FBCNNReducer | None = None

    frozen_count = 0
    drop_count = 0
    artifact_count = 0

    with tqdm(total=len(segments), desc="Correcting segments", unit="seg") as pbar:
        for seg in segments:
            problem_type = seg.problem_type

            if problem_type == ProblemType.FROZEN:
                # Load FILM if not yet loaded
                if film_model is None:
                    film_model = FILMInterpolator(
                        model_path=cfg.film_model_path,
                        device=cfg.device,
                        half=cfg.half_precision,
                    )

                # Get anchor frames
                frame_a = frame_map.get(seg.frame_a_id)
                frame_b = frame_map.get(seg.frame_b_id)

                if frame_a is None or frame_b is None:
                    pbar.write(f"Warning: missing anchor frames for FROZEN segment {seg}")
                    pbar.update(1)
                    continue

                # Generate synthetic frames
                n = seg.end_frame_id - seg.start_frame_id + 1
                synthetic_frames = film_model.generate(
                    frame_a.data, frame_b.data, n_frames=n
                )

                # Update existing frozen frame entries with synthetic data
                for i, syn_data in enumerate(synthetic_frames):
                    frame_id = seg.start_frame_id + i
                    if frame_id in frame_map:
                        entry = frame_map[frame_id]
                        entry.data = syn_data
                        entry.status = FrameStatus.SYNTHETIC
                        entry.problem = ProblemType.FROZEN
                        frozen_count += 1

            elif problem_type == ProblemType.DROP:
                # Load FILM or IFRNet based on segment length
                n = seg.end_frame_id - seg.start_frame_id + 1
                use_ifrnet = n <= cfg.drop_short_max_frames

                if use_ifrnet:
                    if ifrnet_model is None:
                        ifrnet_model = IFRNetInterpolator(
                            model_path=cfg.ifrnet_model_path,
                            device=cfg.device,
                            half=cfg.half_precision,
                        )
                    interpolator = ifrnet_model
                else:
                    if film_model is None:
                        film_model = FILMInterpolator(
                            model_path=cfg.film_model_path,
                            device=cfg.device,
                            half=cfg.half_precision,
                        )
                    interpolator = film_model

                # Get anchor frames
                frame_a = frame_map.get(seg.frame_a_id)
                frame_b = frame_map.get(seg.frame_b_id)

                if frame_a is None or frame_b is None:
                    pbar.write(f"Warning: missing anchor frames for DROP segment {seg}")
                    pbar.update(1)
                    continue

                # Generate synthetic frames
                synthetic_frames = interpolator.generate(
                    frame_a.data, frame_b.data, n_frames=n
                )

                # Insert new FrameEntry objects (drop frames don't exist yet)
                for i, syn_data in enumerate(synthetic_frames):
                    frame_id = seg.start_frame_id + i

                    # Compute interpolated PTS
                    t = float(i + 1) / float(n + 1)
                    pts_a = frame_a.pts
                    pts_b = frame_b.pts
                    pts = pts_a + t * (pts_b - pts_a)

                    entry = FrameEntry(
                        frame_id=frame_id,
                        pts=pts,
                        data=syn_data,
                        status=FrameStatus.SYNTHETIC,
                        problem=ProblemType.DROP,
                    )
                    frame_map[frame_id] = entry
                    drop_count += 1

            elif problem_type == ProblemType.ARTIFACT:
                # Load FBCNN if not yet loaded
                if fbcnn_model is None:
                    fbcnn_model = FBCNNReducer(
                        model_path=cfg.fbcnn_model_path,
                        device=cfg.device,
                        half=cfg.half_precision,
                    )

                # Get the frame to clean
                frame = frame_map.get(seg.frame_a_id)
                if frame is None:
                    pbar.write(f"Warning: missing frame for ARTIFACT segment {seg}")
                    pbar.update(1)
                    continue

                # Apply artifact reduction
                cleaned_data = fbcnn_model.reduce(frame.data)
                frame.data = cleaned_data
                frame.status = FrameStatus.CORRECTED
                frame.problem = ProblemType.ARTIFACT
                artifact_count += 1

            pbar.update(1)

    # Re-sort by frame_id and return
    updated_frames = sorted(frame_map.values(), key=lambda f: f.frame_id)

    print(
        f"\n{'─' * 60}\n"
        f"Correction Summary\n"
        f"{'─' * 60}\n"
        f"Frozen frames filled:     {frozen_count}\n"
        f"Drop frames filled:       {drop_count}\n"
        f"Artifact frames cleaned:  {artifact_count}\n"
        f"{'─' * 60}\n"
    )

    return updated_frames
