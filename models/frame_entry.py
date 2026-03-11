"""FrameEntry dataclass for representing individual video frames."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class FrameStatus(str, Enum):
    """Status of a frame in the pipeline."""
    ORIGINAL = "ORIGINAL"
    PENDING = "PENDING"
    CORRECTED = "CORRECTED"
    SYNTHETIC = "SYNTHETIC"


class ProblemType(str, Enum):
    """Type of quality problem detected in a frame."""
    NONE = "NONE"
    FROZEN = "FROZEN"
    DROP = "DROP"
    ARTIFACT = "ARTIFACT"


@dataclass
class FrameEntry:
    """Represents a single frame with metadata for video processing.
    
    Attributes:
        frame_id: Unique identifier for the frame (0-indexed)
        pts: Presentation timestamp in seconds
        data: Frame image data as BGR uint8 numpy array (H, W, 3)
        status: Current processing status of the frame
        problem: Type of quality problem detected, if any
    """
    frame_id: int
    pts: float
    data: np.ndarray
    status: FrameStatus = FrameStatus.ORIGINAL
    problem: ProblemType = ProblemType.NONE
