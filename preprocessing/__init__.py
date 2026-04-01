"""Preprocessing package for Deepfake Detection"""

from .face_detection import FaceDetectionPipeline
from .frame_extractor import FrameExtractor

__all__ = ["FaceDetectionPipeline", "FrameExtractor"]
