"""Blocking artifact detection based on edge analysis."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from models import FrameEntry, ProblemSegment, ProblemType


class ArtifactDetector:
    """Detect blocking artifacts in frames by analyzing high-frequency edge patterns.

    Frames compressed with block-based codecs (e.g., H.264) often show visible
    block edges. This detector measures the ratio of edge strength at block
    boundaries vs. interior regions.
    """

    def __init__(self, block_ratio_threshold: float = 1.8, block_size: int = 8):
        """Initialize the artifact detector.

        Args:
            block_ratio_threshold: Ratio threshold above which blocking artifacts
                are flagged. Higher values require stronger evidence.
            block_size: Expected compression block size (default 8 for most codecs).
        """
        self.block_ratio_threshold = block_ratio_threshold
        self.block_size = block_size

    def feed(self, entry: FrameEntry) -> Optional[ProblemSegment]:
        """Analyze a frame for blocking artifacts and emit a segment if detected.

        Args:
            entry: The frame to analyze.

        Returns:
            A :class:`~models.ProblemSegment` if blocking artifacts are detected,
            or ``None`` otherwise. Artifact segments span a single frame
            (start_frame_id == end_frame_id).
        """
        # Convert to grayscale float32
        bgr = entry.data
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        height, width = gray.shape

        boundary_list: list[float] = []
        interior_list: list[float] = []

        # Analyze horizontal edges (differences between adjacent columns)
        for x in range(width - 1):
            # Column x and x+1: calculate mean absolute difference across rows
            col_diff = np.abs(gray[:, x + 1] - gray[:, x])
            mean_diff = float(np.mean(col_diff))

            # Classify as boundary or interior based on block alignment
            if (x + 1) % self.block_size == 0:
                boundary_list.append(mean_diff)
            else:
                interior_list.append(mean_diff)

        # Compute blocking score
        if not boundary_list or not interior_list:
            # Edge case: not enough columns to categorize
            return None

        mean_boundary = np.mean(boundary_list)
        mean_interior = np.mean(interior_list)

        score = mean_boundary / (mean_interior + 1e-6)

        if score > self.block_ratio_threshold:
            # Artifact detected: emit a single-frame segment
            segment = ProblemSegment(
                problem_type=ProblemType.ARTIFACT,
                start_frame_id=entry.frame_id,
                end_frame_id=entry.frame_id,
                frame_a_id=entry.frame_id,
                frame_b_id=entry.frame_id,
            )
            return segment

        return None


if __name__ == "__main__":
    # Smoke test: one frame with hard-edged 8x8 blocks, one with smooth gradient

    # Test 1: Create a frame with 8x8 blocks (should have high blocking score)
    blocked_frame = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(0, 256, 8):
        for j in range(0, 256, 8):
            # Alternate block colors
            value = (((i // 8) + (j // 8)) % 2) * 128 + 64
            blocked_frame[i : i + 8, j : j + 8] = value

    entry_blocked = FrameEntry(
        frame_id=0, pts=0.0, data=blocked_frame, status=None, problem=None
    )

    # Test 2: Create a smooth gradient frame (should have low blocking score)
    gradient_frame = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        gradient_frame[i, :] = int(i)  # Vertical gradient, no block edges

    entry_gradient = FrameEntry(
        frame_id=1, pts=1.0 / 30.0, data=gradient_frame, status=None, problem=None
    )

    detector = ArtifactDetector(block_ratio_threshold=1.8, block_size=8)

    # Test blocked frame: should emit a segment
    seg_blocked = detector.feed(entry_blocked)
    assert seg_blocked is not None, "Expected artifact detection on blocked frame"
    assert seg_blocked.problem_type == ProblemType.ARTIFACT
    assert seg_blocked.start_frame_id == 0
    assert seg_blocked.end_frame_id == 0

    # Test gradient frame: should NOT emit a segment
    seg_gradient = detector.feed(entry_gradient)
    assert seg_gradient is None, f"Unexpected artifact detection on gradient: {seg_gradient}"

    print("Smoke test passed!")
    print("  Blocked frame correctly flagged as artifact")
    print("  Gradient frame correctly passed as artifact-free")
