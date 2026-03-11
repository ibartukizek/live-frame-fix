"""ProblemSegment dataclass for describing contiguous problem regions in video."""

from dataclasses import dataclass

from .frame_entry import ProblemType


@dataclass
class ProblemSegment:
    """Represents a contiguous segment of frames with a detected quality problem.
    
    A ProblemSegment spans from start_frame_id to end_frame_id (inclusive) and
    holds references to the anchor frames (frame_a_id and frame_b_id) that
    bracket the segment and can be used for interpolation or correction.
    
    Attributes:
        problem_type: The category of quality issue affecting this segment
        start_frame_id: First frame_id in the problematic segment (inclusive)
        end_frame_id: Last frame_id in the problematic segment (inclusive)
        frame_a_id: Reference frame id before the segment (left anchor)
        frame_b_id: Reference frame id after the segment (right anchor)
    """
    problem_type: ProblemType
    start_frame_id: int
    end_frame_id: int
    frame_a_id: int
    frame_b_id: int
