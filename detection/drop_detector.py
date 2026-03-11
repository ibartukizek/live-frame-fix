"""Frame drop detection based on PTS gap analysis."""

from __future__ import annotations

from typing import Optional

import numpy as np

from models import FrameEntry, ProblemSegment, ProblemType


class DropDetector:
    """Detect missing (dropped) frames by analyzing PTS gaps between frames.

    A drop is detected when the PTS gap between consecutive frames exceeds
    ``expected_gap * (1 + tolerance)``, where expected_gap = 1.0 / fps.

    Note:
        Dropped frames do not exist in the ``FrameEntry`` list. The emitted
        ``ProblemSegment`` uses *synthetic* frame IDs for the dropped range,
        derived from the IDs of the surrounding real frames.
    """

    def __init__(self, fps: int = 30, tolerance: float = 0.5):
        """Initialize the frame drop detector.

        Args:
            fps: Expected frames per second of the source video.
            tolerance: Fractional tolerance above the expected gap before a
                drop is declared. For example, 0.5 means a gap 50% longer
                than expected triggers detection.
        """
        self.fps = fps
        self.tolerance = tolerance
        self.expected_gap: float = 1.0 / fps

        # State tracking
        self._prev_pts: Optional[float] = None
        self._prev_frame_id: Optional[int] = None

    def feed(self, entry: FrameEntry) -> Optional[ProblemSegment]:
        """Process a frame and emit a ProblemSegment if a drop is detected.

        Args:
            entry: The current frame to analyze.

        Returns:
            A :class:`~models.ProblemSegment` if a gap large enough to indicate
            dropped frames was detected between the previous and current frames,
            or ``None`` otherwise.
        """
        current_pts = entry.pts
        current_id = entry.frame_id

        segment: Optional[ProblemSegment] = None

        if self._prev_pts is None:
            # First frame: initialize state
            self._prev_pts = current_pts
            self._prev_frame_id = current_id
            return None

        gap = current_pts - self._prev_pts
        threshold = self.expected_gap * (1.0 + self.tolerance)

        if gap > threshold:
            # Drop detected: compute the range of missing synthetic frame IDs
            missing_count = round(gap / self.expected_gap) - 1
            start_frame_id = self._prev_frame_id + 1
            end_frame_id = current_id - 1

            segment = ProblemSegment(
                problem_type=ProblemType.DROP,
                start_frame_id=start_frame_id,
                end_frame_id=end_frame_id,
                frame_a_id=self._prev_frame_id,
                frame_b_id=current_id,
            )

        # Update state for next iteration
        self._prev_pts = current_pts
        self._prev_frame_id = current_id

        return segment


if __name__ == "__main__":
    # Smoke test:
    # 1. Create 100 FrameEntry objects with correct PTS (i/30.0)
    # 2. Remove entries 50-54 (simulate 5 dropped frames)
    # 3. Renumber remaining frames but keep original PTS values
    # 4. Assert one ProblemSegment with start=50, end=54, frame_a=49, frame_b=55

    # Build all 100 frames with original PTS
    dummy_data = np.zeros((4, 4, 3), dtype=np.uint8)
    all_frames = [
        FrameEntry(frame_id=i, pts=float(i) / 30.0, data=dummy_data)
        for i in range(100)
    ]

    # Remove frames 50-54 to simulate drops
    # Note: surviving frames keep their original frame_ids and PTS values
    surviving_frames = [f for f in all_frames if not (50 <= f.frame_id <= 54)]

    # Run DropDetector
    detector = DropDetector(fps=30, tolerance=0.5)
    segments = []

    for entry in surviving_frames:
        seg = detector.feed(entry)
        if seg is not None:
            segments.append(seg)

    # Assertions
    assert len(segments) == 1, f"Expected 1 segment, got {len(segments)}: {segments}"
    seg = segments[0]
    assert seg.problem_type == ProblemType.DROP, f"Expected DROP, got {seg.problem_type}"
    assert seg.start_frame_id == 50, f"Expected start=50, got {seg.start_frame_id}"
    assert seg.end_frame_id == 54, f"Expected end=54, got {seg.end_frame_id}"
    assert seg.frame_a_id == 49, f"Expected frame_a=49, got {seg.frame_a_id}"
    assert seg.frame_b_id == 55, f"Expected frame_b=55, got {seg.frame_b_id}"

    print("Smoke test passed!")
    print(
        f"  Segment: start={seg.start_frame_id}, end={seg.end_frame_id}, "
        f"frame_a={seg.frame_a_id}, frame_b={seg.frame_b_id}"
    )
