"""Frame quality detection logic (frozen, drop, artifact detectors)."""

from .artifact_detector import ArtifactDetector
from .drop_detector import DropDetector
from .frozen_detector import FrozenDetector

__all__ = ["FrozenDetector", "DropDetector", "ArtifactDetector"]
