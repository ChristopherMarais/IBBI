# src/ibbi/models/__init__.py

from .classification import yolov10x_bb_classify_model
from .detection import yolov10x_bb_detect_model

__all__ = ["yolov10x_bb_detect_model", "yolov10x_bb_classify_model"]
