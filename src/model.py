"""
Model configuration and utilities for semantic segmentation with Detectron2.
"""

import torch
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.modeling import build_model
from typing import Optional


class Detectron2Config:
    """
    Wrapper for Detectron2 configuration.

    Attributes:
        cfg: Detectron2 config object.
    """

    def __init__(
        self,
        model_name: str = "mask_rcnn_R_50_FPN_3x",
        num_classes: int = 13,
        num_gpus: int = 1,
        batch_size_per_image: int = 512,
        learning_rate: float = 0.02,
        max_iter: int = 10000,
        checkpoint_period: int = 500,
    ):
        """
        Initialize Detectron2 config.

        Args:
            model_name: Name of the model from Model Zoo.
            num_classes: Number of classes in the dataset.
            num_gpus: Number of GPUs to use.
            batch_size_per_image: RPN batch size.
            learning_rate: Initial learning rate.
            max_iter: Maximum training iterations.
            checkpoint_period: Save checkpoint every N iterations.
        """
        self.cfg = get_cfg()

        # Load from model zoo
        config_file = f"COCO-InstanceSegmentation/{model_name}.yaml"
        self.cfg.merge_from_file(model_zoo.get_config_file(config_file))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)

        # Dataset config
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

        # Training config
        self.cfg.SOLVER.IMS_PER_BATCH = 2 * num_gpus
        self.cfg.SOLVER.BASE_LR = learning_rate
        self.cfg.SOLVER.MAX_ITER = max_iter
        self.cfg.SOLVER.STEPS = (max_iter // 2, max_iter * 3 // 4)
        self.cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period

        # RPN config
        self.cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = batch_size_per_image

        # Other configs
        self.cfg.DATALOADER.NUM_WORKERS = 4
        self.cfg.INPUT.MASK_FORMAT = "bitmask"
        self.cfg.VIS_PERIOD = 0

        self.cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def set_dataset_paths(self, train_dir: str, val_dir: str, test_dir: Optional[str] = None):
        """
        Set dataset directory paths.

        Args:
            train_dir: Path to training dataset.
            val_dir: Path to validation dataset.
            test_dir: Path to test dataset (optional).
        """
        self.cfg.DATASETS.TRAIN = tuple([train_dir])
        self.cfg.DATASETS.VAL = tuple([val_dir])
        if test_dir:
            self.cfg.DATASETS.TEST = tuple([test_dir])

    def set_output_dir(self, output_dir: str):
        """Set output directory for checkpoints and logs."""
        self.cfg.OUTPUT_DIR = output_dir

    def get_cfg(self):
        """Get the config object."""
        return self.cfg


class SemanticSegmentationConfig:
    """
    Configuration for semantic segmentation training.

    This is a simpler alternative to Detectron2's default config,
    useful for custom implementations.
    """

    def __init__(
        self,
        num_classes: int = 13,
        image_size: int = 512,
        learning_rate: float = 0.001,
        batch_size: int = 8,
        num_epochs: int = 50,
        weight_decay: float = 1e-5,
        patience: int = 10,
    ):
        self.num_classes = num_classes
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.patience = patience  # For early stopping

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            "num_classes": self.num_classes,
            "image_size": self.image_size,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
        }
