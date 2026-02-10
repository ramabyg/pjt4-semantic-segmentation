"""
Configuration loader for Detectron2 models and training parameters.
Supports YAML-based configuration for easy model and hyperparameter management.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration dataclass"""
    name: str
    model_name: str  # e.g., "mask_rcnn_R_50_FPN_3x"
    num_classes: int
    num_gpus: int
    batch_size_per_image: int
    backbone: Optional[str] = None
    pretrained: bool = True


@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    learning_rate: float
    max_iter: int
    batch_size: int
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_iters: int = 1000
    lr_decay_milestones: list = None
    lr_decay_gamma: float = 0.1
    checkpoint_period: int = 500
    num_workers: int = 4
    patience: int = 10  # For early stopping


@dataclass
class DataConfig:
    """Data configuration dataclass"""
    data_path: str
    images_folder: str = "images"
    masks_folder: str = "masks"
    train_ratio: float = 0.8
    image_size: int = 512
    num_classes: int = 13
    class_names: list = None


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    experiment_name: str
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    output_dir: str = "./outputs"
    seed: int = 42
    device: str = "cuda"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'experiment_name': self.experiment_name,
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'output_dir': self.output_dir,
            'seed': self.seed,
            'device': self.device,
        }


class ConfigLoader:
    """Load configurations from YAML files"""

    @staticmethod
    def load_yaml(yaml_path: str) -> Dict[str, Any]:
        """
        Load YAML configuration file

        Args:
            yaml_path: Path to YAML file

        Returns:
            Dictionary with configuration
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded config from: {yaml_path}")
        return config

    @staticmethod
    def load_experiment_config(yaml_path: str) -> ExperimentConfig:
        """
        Load experiment configuration from YAML

        Args:
            yaml_path: Path to YAML file

        Returns:
            ExperimentConfig object
        """
        config_dict = ConfigLoader.load_yaml(yaml_path)

        # Parse nested configs
        model_cfg = ModelConfig(**config_dict['model'])

        training_cfg_dict = config_dict['training']
        if 'lr_decay_milestones' not in training_cfg_dict:
            training_cfg_dict['lr_decay_milestones'] = []
        training_cfg = TrainingConfig(**training_cfg_dict)

        data_cfg_dict = config_dict['data']
        if 'class_names' not in data_cfg_dict:
            data_cfg_dict['class_names'] = [
                "background", "building", "car", "clutter", "dog", "fence",
                "grass", "person", "sky", "tree", "truck", "void", "water"
            ]
        data_cfg = DataConfig(**data_cfg_dict)

        experiment = ExperimentConfig(
            experiment_name=config_dict.get('experiment_name', 'default'),
            model=model_cfg,
            training=training_cfg,
            data=data_cfg,
            output_dir=config_dict.get('output_dir', './outputs'),
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', 'cuda'),
        )

        return experiment

    @staticmethod
    def save_config(config: ExperimentConfig, output_path: str):
        """
        Save configuration to YAML file

        Args:
            config: ExperimentConfig object
            output_path: Path to save YAML
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = config.to_dict()

        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        logger.info(f"Saved config to: {output_path}")


class PerformanceTracker:
    """
    Track and save performance metrics across training iterations.
    Supports saving to JSON and CSV for analysis.
    """

    def __init__(self, output_dir: str, experiment_name: str):
        """
        Initialize performance tracker

        Args:
            output_dir: Directory to save metrics
            experiment_name: Name of the experiment
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.metrics_dir = self.output_dir / "metrics" / experiment_name
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"PerformanceTracker initialized at: {self.metrics_dir}")

    def log_iteration(self, iteration: int, metrics: Dict[str, float], stage: str = "train"):
        """
        Log metrics for a single iteration

        Args:
            iteration: Iteration/epoch number
            metrics: Dictionary of metric names and values
            stage: "train", "val", or "test"
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'iteration': iteration,
            'stage': stage,
            **metrics
        }
        self.metrics_history.append(record)

        logger.debug(f"Logged metrics for {stage} iteration {iteration}")

    def log_batch(self, iteration: int, batch_idx: int, metrics: Dict[str, float]):
        """
        Log metrics for a batch (during training)

        Args:
            iteration: Epoch number
            batch_idx: Batch index
            metrics: Dictionary of metric names and values
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'epoch': iteration,
            'batch': batch_idx,
            'stage': 'batch',
            **metrics
        }
        self.metrics_history.append(record)

    def save_metrics_json(self) -> Path:
        """
        Save all metrics to JSON file

        Returns:
            Path to saved JSON file
        """
        output_file = self.metrics_dir / f"metrics_{self.timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        logger.info(f"Saved metrics to JSON: {output_file}")
        return output_file

    def save_metrics_csv(self) -> Path:
        """
        Save metrics to CSV file

        Returns:
            Path to saved CSV file
        """
        import csv

        if not self.metrics_history:
            logger.warning("No metrics to save")
            return None

        output_file = self.metrics_dir / f"metrics_{self.timestamp}.csv"

        # Get all unique keys
        keys = set()
        for record in self.metrics_history:
            keys.update(record.keys())
        keys = sorted(list(keys))

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.metrics_history)

        logger.info(f"Saved metrics to CSV: {output_file}")
        return output_file

    def get_best_metrics(self, metric_name: str, stage: str = "val") -> Dict[str, Any]:
        """
        Get best metrics for a specific metric

        Args:
            metric_name: Name of the metric to find best value
            stage: Stage to filter ("train", "val", "test")

        Returns:
            Dictionary with best metrics
        """
        filtered = [m for m in self.metrics_history
                   if m.get('stage') == stage and metric_name in m]

        if not filtered:
            return None

        # Find best (maximize by default, minimize for loss)
        if 'loss' in metric_name.lower():
            best = min(filtered, key=lambda x: x[metric_name])
        else:
            best = max(filtered, key=lambda x: x[metric_name])

        return best

    def save_summary(self) -> Path:
        """
        Save summary of best metrics

        Returns:
            Path to summary file
        """
        output_file = self.metrics_dir / f"summary_{self.timestamp}.txt"

        with open(output_file, 'w') as f:
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Total records: {len(self.metrics_history)}\n\n")

            # Get stages
            stages = set(m.get('stage') for m in self.metrics_history)

            for stage in sorted(stages):
                f.write(f"\n{'='*60}\n")
                f.write(f"Stage: {stage.upper()}\n")
                f.write(f"{'='*60}\n")

                stage_metrics = [m for m in self.metrics_history
                               if m.get('stage') == stage]

                if not stage_metrics:
                    continue

                # Get metric names (exclude metadata fields)
                exclude_keys = {'timestamp', 'iteration', 'epoch', 'batch', 'stage'}
                metric_names = set()
                for m in stage_metrics:
                    metric_names.update(k for k in m.keys() if k not in exclude_keys)

                for metric_name in sorted(metric_names):
                    values = [m[metric_name] for m in stage_metrics
                             if metric_name in m]

                    if values:
                        f.write(f"\n{metric_name}:\n")
                        f.write(f"  Min: {min(values):.6f}\n")
                        f.write(f"  Max: {max(values):.6f}\n")
                        f.write(f"  Avg: {sum(values) / len(values):.6f}\n")
                        f.write(f"  Latest: {values[-1]:.6f}\n")

        logger.info(f"Saved summary to: {output_file}")
        return output_file

    def plot_metrics(self, metrics_to_plot: list = None):
        """
        Plot metrics

        Args:
            metrics_to_plot: List of metric names to plot
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        if not self.metrics_history:
            logger.warning("No metrics to plot")
            return

        df = pd.DataFrame(self.metrics_history)

        # Get stages
        stages = df['stage'].unique()

        # Filter metrics
        if metrics_to_plot is None:
            exclude_keys = {'timestamp', 'iteration', 'epoch', 'batch', 'stage'}
            metrics_to_plot = [col for col in df.columns
                             if col not in exclude_keys]

        # Create subplots
        n_plots = len(metrics_to_plot)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots))

        if n_plots == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics_to_plot):
            if metric not in df.columns:
                continue

            for stage in stages:
                stage_df = df[df['stage'] == stage]
                if metric in stage_df.columns:
                    x_col = 'iteration' if 'iteration' in stage_df.columns else 'epoch'
                    axes[idx].plot(stage_df[x_col], stage_df[metric],
                                 label=stage, marker='o', alpha=0.7)

            axes[idx].set_title(f"Metric: {metric}")
            axes[idx].set_xlabel("Iteration")
            axes[idx].set_ylabel(metric)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        fig_path = self.metrics_dir / f"metrics_plot_{self.timestamp}.png"
        plt.savefig(fig_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved plot to: {fig_path}")

        plt.show()

        return fig


def create_sample_configs():
    """Create sample configuration files"""

    configs = {
        "config_baseline.yaml": {
            "experiment_name": "baseline_maskrcnn",
            "model": {
                "name": "Mask R-CNN R50-FPN",
                "model_name": "mask_rcnn_R_50_FPN_3x",
                "num_classes": 13,
                "num_gpus": 1,
                "batch_size_per_image": 512,
            },
            "training": {
                "learning_rate": 0.02,
                "max_iter": 10000,
                "batch_size": 4,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "warmup_iters": 1000,
                "lr_decay_milestones": [5000, 7500],
                "lr_decay_gamma": 0.1,
                "checkpoint_period": 500,
                "num_workers": 4,
                "patience": 10,
            },
            "data": {
                "data_path": "./data",
                "images_folder": "images",
                "masks_folder": "masks",
                "train_ratio": 0.8,
                "image_size": 512,
                "num_classes": 13,
            },
            "output_dir": "./outputs",
            "seed": 42,
            "device": "cuda",
        },

        "config_large_model.yaml": {
            "experiment_name": "large_maskrcnn_r101",
            "model": {
                "name": "Mask R-CNN R101-FPN",
                "model_name": "mask_rcnn_R_101_FPN_3x",
                "num_classes": 13,
                "num_gpus": 1,
                "batch_size_per_image": 512,
            },
            "training": {
                "learning_rate": 0.01,
                "max_iter": 15000,
                "batch_size": 2,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "warmup_iters": 1000,
                "lr_decay_milestones": [7500, 11250],
                "lr_decay_gamma": 0.1,
                "checkpoint_period": 500,
                "num_workers": 4,
                "patience": 15,
            },
            "data": {
                "data_path": "./data",
                "images_folder": "images",
                "masks_folder": "masks",
                "train_ratio": 0.8,
                "image_size": 768,
                "num_classes": 13,
            },
            "output_dir": "./outputs",
            "seed": 42,
            "device": "cuda",
        },

        "config_fast_train.yaml": {
            "experiment_name": "fast_training",
            "model": {
                "name": "Mask R-CNN R50-FPN",
                "model_name": "mask_rcnn_R_50_FPN_3x",
                "num_classes": 13,
                "num_gpus": 1,
                "batch_size_per_image": 256,
            },
            "training": {
                "learning_rate": 0.02,
                "max_iter": 5000,
                "batch_size": 8,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "warmup_iters": 500,
                "lr_decay_milestones": [2500, 3750],
                "lr_decay_gamma": 0.1,
                "checkpoint_period": 250,
                "num_workers": 2,
                "patience": 5,
            },
            "data": {
                "data_path": "./data",
                "images_folder": "images",
                "masks_folder": "masks",
                "train_ratio": 0.8,
                "image_size": 384,
                "num_classes": 13,
            },
            "output_dir": "./outputs",
            "seed": 42,
            "device": "cuda",
        },
    }

    config_dir = Path("./configs")
    config_dir.mkdir(exist_ok=True)

    for config_name, config_dict in configs.items():
        config_path = config_dir / config_name
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        print(f"Created: {config_path}")

    return config_dir
