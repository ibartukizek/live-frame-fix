"""Frame correction strategies (interpolation, artifact removal)."""

from .fbcnn_reducer import FBCNNReducer
from .film_interpolator import FILMInterpolator
from .ifrnet_interpolator import IFRNetInterpolator

__all__ = ["FILMInterpolator", "IFRNetInterpolator", "FBCNNReducer"]
