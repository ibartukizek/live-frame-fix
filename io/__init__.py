"""I/O utilities for video reading and writing."""

from .encoder import encode_frames
from .extractor import extract_frames

__all__ = ["extract_frames", "encode_frames"]
