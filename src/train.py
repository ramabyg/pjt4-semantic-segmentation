"""
Training script for semantic segmentation with Detectron2.
Can be used as a standalone script or imported in Jupyter notebook.
"""

import os
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import time

from detectron2.engine import DefaultTrainer, default_argument_parser, launch
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer

from .dataset import SemanticSegmentationDataset
from .model import Detectron2Config


logger = logging.getLogger(__name__)


class SemanticSegmentationTrainer:
    """
    Trainer class for semantic segmentation using Detectron2.
    """

    def __init__(
        self,
        cfg: Detectron2Config,
        data_path: str,
        train_ids: list,
        val_ids: list,
        output_dir: str = "./outputs",
    ):
        """
        Initialize trainer.

        Args:
            cfg: Detectron2Config instance.
            data_path: Path to dataset.
            train_ids: List of training image IDs.
            val_ids: List of validation image IDs.
            output_dir: Output directory for checkpoints.
        """
        self.cfg = cfg.get_cfg()
        self.data_path = data_path
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.output_dir = output_dir

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.cfg.OUTPUT_DIR = output_dir

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def register_datasets(self):
        """Register train and validation datasets."""
        # Register training dataset
        train_dataset_name = "train_semseg"
        if train_dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(train_dataset_name)

        DatasetCatalog.register(
            train_dataset_name,
            lambda: self.get_dataset_dicts(self.train_ids),
        )
        MetadataCatalog.get(train_dataset_name).thing_classes = [
            "background", "building", "car", "clutter", "dog", "fence",
            "grass", "person", "sky", "tree", "truck", "void", "water"
        ][:self.cfg.MODEL.ROI_HEADS.NUM_CLASSES]

        # Register validation dataset
        val_dataset_name = "val_semseg"
        if val_dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(val_dataset_name)

        DatasetCatalog.register(
            val_dataset_name,
            lambda: self.get_dataset_dicts(self.val_ids),
        )
        MetadataCatalog.get(val_dataset_name).thing_classes = \
            MetadataCatalog.get(train_dataset_name).thing_classes

        # Set dataset names in config
        self.cfg.DATASETS.TRAIN = (train_dataset_name,)
        self.cfg.DATASETS.TEST = (val_dataset_name,)

    def get_dataset_dicts(self, image_ids: list) -> list:
        """
        Get dataset dictionaries for Detectron2.

        Args:
            image_ids: List of image IDs.

        Returns:
            List of dataset dictionaries.
        """
        dataset_dicts = []

        for image_id in image_ids:
            image_path = Path(self.data_path) / "images" / f"{image_id}.jpg"
            mask_path = Path(self.data_path) / "masks" / f"{image_id}.png"

            if not image_path.exists() or not mask_path.exists():
                continue

            height, width = self.get_image_size(str(image_path))

            record = {
                "file_name": str(image_path),
                "image_id": image_id,
                "height": height,
                "width": width,
                "sem_seg_file_name": str(mask_path),
            }
            dataset_dicts.append(record)

        return dataset_dicts

    @staticmethod
    def get_image_size(image_path: str) -> tuple:
        """Get image dimensions."""
        import cv2
        img = cv2.imread(image_path)
        return img.shape[:2]

    def train(self):
        """Train the model."""
        self.register_datasets()

        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)

        logger.info("Starting training...")
        start_time = time.time()

        trainer.train()

        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time:.2f} seconds")

    def evaluate(self):
        """Evaluate the model on validation set."""
        self.register_datasets()

        evaluator = COCOEvaluator(
            "val_semseg",
            self.cfg,
            False,
            output_dir=self.output_dir,
        )

        val_loader = build_detection_test_loader(self.cfg, "val_semseg")

        from detectron2.modeling import build_model
        model = build_model(self.cfg)

        logger.info("Evaluating model...")
        results = inference_on_dataset(model, val_loader, evaluator)

        return results


def train_detectron2_model(
    config: Detectron2Config,
    data_path: str,
    train_ids: list,
    val_ids: list,
    output_dir: str = "./outputs",
    num_gpus: int = 1,
):
    """
    Train a Detectron2 model for semantic segmentation.

    Args:
        config: Detectron2Config instance.
        data_path: Path to dataset.
        train_ids: List of training image IDs.
        val_ids: List of validation image IDs.
        output_dir: Output directory for checkpoints.
        num_gpus: Number of GPUs to use.
    """
    trainer = SemanticSegmentationTrainer(
        config,
        data_path,
        train_ids,
        val_ids,
        output_dir,
    )

    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    # Example usage as standalone script
    import sys

    # Configure your paths and settings here
    data_path = "./data"
    output_dir = "./outputs"

    # Load config
    config = Detectron2Config(
        num_classes=13,
        learning_rate=0.02,
        max_iter=10000,
    )

    # Split dataset
    from .dataset import SemanticSegmentationDataset
    dataset = SemanticSegmentationDataset(data_path)
    train_ids, val_ids = SemanticSegmentationDataset.train_val_split(
        dataset.image_files,
        train_ratio=0.8,
    )

    # Train
    train_detectron2_model(
        config,
        data_path,
        train_ids,
        val_ids,
        output_dir,
    )
