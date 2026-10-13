"""Joint RGB/NIR network and strict loader used by the legacy generation CLI."""

from .yolo26 import Yolo26DepthDetRoad
from .checkpoint import load_model

__all__ = ["Yolo26DepthDetRoad", "load_model"]
