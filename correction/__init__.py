"""Frame correction strategies (interpolation, artifact removal)."""

from .film_interpolator import FILMInterpolator
from .ifrnet_interpolator import IFRNetInterpolator

__all__ = ["FILMInterpolator", "IFRNetInterpolator"]
