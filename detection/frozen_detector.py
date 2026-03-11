"""Frozen frame detection using perceptual hashing."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from models import FrameEntry, ProblemSegment, ProblemType


def _compute_phash(frame: np.ndarray) -> int:
    """Compute a 64-bit perceptual hash for a BGR frame.

    The algorithm:
    1. Convert to grayscale
    2. Resize to 32x32 pixels
    3. Apply 2D DCT on float32 version
    4. Extract the top-left 8x8 block (low frequencies)
    5. Compute median of the 8x8 block
    6. Generate 64-bit hash: bit=1 if value > median, else 0

    Args:
        frame: BGR uint8 numpy array (H, W, 3)

    Returns:
        64-bit integer hash.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize to 32x32
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)

    # Convert to float32 and apply DCT
    resized_float = resized.astype(np.float32)
    dct = cv2.dct(resized_float)

    # Extract top-left 8x8 block
    block = dct[:8, :8]

    # Compute median
    median = np.median(block)

    # Generate 64-bit hash
    hash_bits = (block > median).flatten().astype(np.uint8)
    hash_int = int("".join(str(bit) for bit in hash_bits), 2)

    return hash_int


def _hamming_distance(hash1: int, hash2: int) -> int:
    """Compute Hamming distance between two 64-bit hashes.

    Args:
        hash1: First hash as integer.
        hash2: Second hash as integer.

    Returns:
        Number of differing bits.
    """
    xor = hash1 ^ hash2
    return bin(xor).count("1")


class FrozenDetector:
    """Detect frozen frames by tracking perceptual hash similarity.

    A freeze is detected when consecutive frames have similar hashes
    (Hamming distance <= hash_threshold) for at least min_duration frames.
    """

    def __init__(self, hash_threshold: int = 5, min_duration: int = 3):
        """Initialize the frozen frame detector.

        Args:
            hash_threshold: Maximum Hamming distance to consider frames identical.
            min_duration: Minimum number of consecutive frozen frames to trigger a segment.
        """
        self.hash_threshold = hash_threshold
        self.min_duration = min_duration

        # State tracking
        self._prev_hash: Optional[int] = None
        self._prev_frame_id: Optional[int] = None
        self._freeze_start_frame_id: Optional[int] = None
        self._frame_before_freeze_id: Optional[int] = None
        self._frozen_count: int = 0

    def feed(self, entry: FrameEntry) -> Optional[ProblemSegment]:
        """Process a frame and emit a ProblemSegment if a freeze segment ends.

        Args:
            entry: The current frame to analyze.

        Returns:
            A :class:`~models.ProblemSegment` if a frozen segment just ended,
            or ``None`` otherwise.
        """
        current_hash = _compute_phash(entry.data)
        current_id = entry.frame_id

        segment: Optional[ProblemSegment] = None

        if self._prev_hash is None:
            # First frame: initialize state
            self._prev_hash = current_hash
            self._prev_frame_id = current_id
            return None

        # Compute similarity with previous frame
        distance = _hamming_distance(current_hash, self._prev_hash)
        is_frozen = distance <= self.hash_threshold

        if is_frozen:
            # Current frame is frozen (similar to previous)
            if self._freeze_start_frame_id is None:
                # Start of a new freeze
                self._freeze_start_frame_id = self._prev_frame_id
                self._frame_before_freeze_id = self._prev_frame_id - 1
                self._frozen_count = 1
            else:
                # Continuation of existing freeze
                self._frozen_count += 1
        else:
            # Current frame is NOT frozen (different from previous)
            if self._freeze_start_frame_id is not None:
                # A freeze was ongoing and just ended
                freeze_end_id = self._prev_frame_id
                if self._frozen_count >= self.min_duration:
                    segment = ProblemSegment(
                        problem_type=ProblemType.FROZEN,
                        start_frame_id=self._freeze_start_frame_id,
                        end_frame_id=freeze_end_id,
                        frame_a_id=self._frame_before_freeze_id,
                        frame_b_id=current_id,  # Current frame is the first recovered frame
                    )

                # Reset freeze state
                self._freeze_start_frame_id = None
                self._frame_before_freeze_id = None
                self._frozen_count = 0

        # Update state for next iteration
        self._prev_hash = current_hash
        self._prev_frame_id = current_id

        return segment

    def finalize(self) -> Optional[ProblemSegment]:
        """Call at end-of-stream to emit a freeze that is still ongoing.

        Returns:
            A :class:`~models.ProblemSegment` if a freeze was ongoing at EOF,
            with ``frame_b_id=-1`` to indicate no recovery frame.
        """
        if self._freeze_start_frame_id is not None and self._frozen_count >= self.min_duration:
            return ProblemSegment(
                problem_type=ProblemType.FROZEN,
                start_frame_id=self._freeze_start_frame_id,
                end_frame_id=self._prev_frame_id,  # Last frame we saw
                frame_a_id=self._frame_before_freeze_id,
                frame_b_id=-1,  # No recovery frame (still frozen at EOF)
            )
        return None


if __name__ == "__main__":
    # Smoke test: create 100 fake frames, make frames 40-70 identical
    frames = []
    for i in range(100):
        # Create different images by using seeded random noise
        if 40 <= i <= 70:
            # Make frames 40-70 identical (same seed)
            np.random.seed(42)
        else:
            # Each unique frame has unique content (different seed)
            np.random.seed(i)

        data = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        entry = FrameEntry(
            frame_id=i,
            pts=float(i) / 30.0,
            data=data,
        )
        frames.append(entry)

    detector = FrozenDetector(hash_threshold=5, min_duration=3)
    segments = []

    for entry in frames:
        segment = detector.feed(entry)
        if segment is not None:
            segments.append(segment)

    # Handle end-of-stream
    final_segment = detector.finalize()
    if final_segment is not None:
        segments.append(final_segment)

    # Assertions
    assert len(segments) == 1, f"Expected 1 segment, got {len(segments)}"
    seg = segments[0]
    assert seg.start_frame_id == 40, f"Expected start=40, got {seg.start_frame_id}"
    assert seg.end_frame_id == 70, f"Expected end=70, got {seg.end_frame_id}"
    assert seg.frame_a_id == 39, f"Expected frame_a=39, got {seg.frame_a_id}"
    assert seg.frame_b_id == 71, f"Expected frame_b=71, got {seg.frame_b_id}"

    print("Smoke test passed!")
    print(f"  Segment: start={seg.start_frame_id}, end={seg.end_frame_id}, "
          f"frame_a={seg.frame_a_id}, frame_b={seg.frame_b_id}")
