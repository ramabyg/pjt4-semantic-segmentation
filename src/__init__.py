"""
Semantic Segmentation Project with Detectron2

A modular implementation for semantic segmentation using Detectron2.
Designed for local development and Kaggle training.
"""

from .dataset import SemanticSegmentationDataset, get_default_transforms
from .model import Detectron2Config, SemanticSegmentationConfig
from .train import SemanticSegmentationTrainer, train_detectron2_model
from .inference import SemanticSegmentationPredictor, create_submission_csv, rle_encode, rle_decode
from .utils import SegmentationMetrics, VisualizationUtils, PathUtils
from .config_loader import ConfigLoader, PerformanceTracker, ExperimentConfig, ModelConfig, TrainingConfig, DataConfig, create_sample_configs

__version__ = "1.0.0"

__all__ = [
    "SemanticSegmentationDataset",
    "get_default_transforms",
    "Detectron2Config",
    "SemanticSegmentationConfig",
    "SemanticSegmentationTrainer",
    "train_detectron2_model",
    "SemanticSegmentationPredictor",
    "create_submission_csv",
    "rle_encode",
    "rle_decode",
    "SegmentationMetrics",
    "VisualizationUtils",
    "PathUtils",
    "ConfigLoader",
    "PerformanceTracker",
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "create_sample_configs",
]
