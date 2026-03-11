"""Frame quality detection logic (frozen, drop, artifact detectors)."""

from .drop_detector import DropDetector
from .frozen_detector import FrozenDetector

__all__ = ["FrozenDetector", "DropDetector"]
