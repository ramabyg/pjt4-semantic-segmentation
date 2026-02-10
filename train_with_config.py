#!/usr/bin/env python
"""
Example script showing how to use configuration and performance tracking
for training semantic segmentation models.

Usage:
    python train_with_config.py --config configs/config_baseline.yaml
    python train_with_config.py --config configs/config_large_model.yaml
"""

import argparse
import logging
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.config_loader import ConfigLoader, PerformanceTracker
from src.dataset import SemanticSegmentationDataset, get_default_transforms
from src.train import SemanticSegmentationTrainer
from src.model import Detectron2Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(config_path: str):
    """
    Train model using configuration file

    Args:
        config_path: Path to YAML configuration file
    """

    # Load configuration
    logger.info(f"Loading configuration from: {config_path}")
    experiment_config = ConfigLoader.load_experiment_config(config_path)

    logger.info(f"Experiment: {experiment_config.experiment_name}")
    logger.info(f"Model: {experiment_config.model.model_name}")
    logger.info(f"Max Iterations: {experiment_config.training.max_iter}")

    # Set random seeds
    np.random.seed(experiment_config.seed)
    torch.manual_seed(experiment_config.seed)

    # Initialize performance tracker
    tracker = PerformanceTracker(
        output_dir=experiment_config.output_dir,
        experiment_name=experiment_config.experiment_name,
    )

    # Save configuration for reference
    ConfigLoader.save_config(
        experiment_config,
        tracker.metrics_dir / "experiment_config.yaml"
    )

    logger.info(f"Metrics directory: {tracker.metrics_dir}")

    # Load dataset
    logger.info(f"Loading dataset from: {experiment_config.data.data_path}")
    dataset = SemanticSegmentationDataset(
        data_path=experiment_config.data.data_path,
        images_folder=experiment_config.data.images_folder,
        masks_folder=experiment_config.data.masks_folder,
    )

    # Split into train and validation
    train_ids, val_ids = SemanticSegmentationDataset.train_val_split(
        dataset.image_files,
        train_ratio=experiment_config.data.train_ratio,
        random_state=experiment_config.seed,
    )

    logger.info(f"Train samples: {len(train_ids)}")
    logger.info(f"Val samples: {len(val_ids)}")

    # Create Detectron2 config
    logger.info("Creating Detectron2 configuration...")
    detectron2_config = Detectron2Config(
        model_name=experiment_config.model.model_name,
        num_classes=experiment_config.model.num_classes,
        num_gpus=experiment_config.model.num_gpus,
        batch_size_per_image=experiment_config.model.batch_size_per_image,
        learning_rate=experiment_config.training.learning_rate,
        max_iter=experiment_config.training.max_iter,
        checkpoint_period=experiment_config.training.checkpoint_period,
    )

    detectron2_config.set_output_dir(str(tracker.metrics_dir))

    # Create trainer
    logger.info("Initializing trainer...")
    trainer = SemanticSegmentationTrainer(
        cfg=detectron2_config,
        data_path=experiment_config.data.data_path,
        train_ids=train_ids,
        val_ids=val_ids,
        output_dir=str(tracker.metrics_dir),
    )

    # Train
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # Evaluate
    logger.info("Evaluating on validation set...")
    try:
        results = trainer.evaluate()
        logger.info(f"Evaluation results: {results}")
    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")

    # Save summary
    logger.info("Saving metrics and summary...")
    tracker.save_summary()

    logger.info(f"✓ Experiment '{experiment_config.experiment_name}' completed!")
    logger.info(f"Results saved to: {tracker.metrics_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train semantic segmentation model with YAML configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_baseline.yaml",
        help="Path to YAML configuration file",
    )

    args = parser.parse_args()

    # Verify config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    main(str(config_path))
