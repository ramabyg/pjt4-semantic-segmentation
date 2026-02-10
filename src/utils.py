"""
Utility functions for semantic segmentation.
Includes metrics, visualization, and helper functions.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.metrics import jaccard_score, f1_score, confusion_matrix


class SegmentationMetrics:
    """Calculate various metrics for semantic segmentation."""

    @staticmethod
    def iou_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
        """
        Calculate Intersection over Union (IoU) score.

        Args:
            y_true: Ground truth mask (H, W).
            y_pred: Predicted mask (H, W).
            num_classes: Number of classes.

        Returns:
            Mean IoU score across all classes.
        """
        iou_scores = []
        for class_id in range(num_classes):
            gt = (y_true == class_id).astype(np.uint8)
            pred = (y_pred == class_id).astype(np.uint8)

            intersection = np.logical_and(gt, pred).sum()
            union = np.logical_or(gt, pred).sum()

            if union == 0:
                iou_scores.append(1.0)  # If both are empty, IoU is 1
            else:
                iou_scores.append(intersection / union)

        return np.mean(iou_scores)

    @staticmethod
    def dice_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
        """
        Calculate Dice coefficient.

        Args:
            y_true: Ground truth mask (H, W).
            y_pred: Predicted mask (H, W).
            num_classes: Number of classes.

        Returns:
            Mean Dice score across all classes.
        """
        dice_scores = []
        for class_id in range(num_classes):
            gt = (y_true == class_id).astype(np.uint8)
            pred = (y_pred == class_id).astype(np.uint8)

            intersection = np.logical_and(gt, pred).sum()
            total = gt.sum() + pred.sum()

            if total == 0:
                dice_scores.append(1.0)
            else:
                dice_scores.append(2 * intersection / total)

        return np.mean(dice_scores)

    @staticmethod
    def pixel_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate pixel-wise accuracy.

        Args:
            y_true: Ground truth mask (H, W).
            y_pred: Predicted mask (H, W).

        Returns:
            Accuracy score.
        """
        return np.sum(y_true == y_pred) / y_true.size

    @staticmethod
    def class_wise_iou(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
        """
        Calculate IoU for each class.

        Args:
            y_true: Ground truth mask (H, W).
            y_pred: Predicted mask (H, W).
            num_classes: Number of classes.

        Returns:
            Dictionary with IoU for each class.
        """
        class_iou = {}
        for class_id in range(num_classes):
            gt = (y_true == class_id).astype(np.uint8)
            pred = (y_pred == class_id).astype(np.uint8)

            intersection = np.logical_and(gt, pred).sum()
            union = np.logical_or(gt, pred).sum()

            if union == 0:
                class_iou[class_id] = 1.0
            else:
                class_iou[class_id] = intersection / union

        return class_iou


class VisualizationUtils:
    """Visualization utilities for segmentation results."""

    @staticmethod
    def plot_image_and_mask(
        image: np.ndarray,
        mask: np.ndarray,
        title: str = "",
        figsize: Tuple[int, int] = (12, 6),
    ):
        """
        Plot image and its mask side by side.

        Args:
            image: Input image (H, W, 3).
            mask: Segmentation mask (H, W).
            title: Title for the plot.
            figsize: Figure size.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].imshow(image.astype(np.uint8))
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="tab20")
        axes[1].set_title("Mask")
        axes[1].axis("off")

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_predictions(
        image: np.ndarray,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        title: str = "",
        figsize: Tuple[int, int] = (15, 5),
    ):
        """
        Plot image, ground truth mask, and prediction side by side.

        Args:
            image: Input image (H, W, 3).
            gt_mask: Ground truth mask (H, W).
            pred_mask: Predicted mask (H, W).
            title: Title for the plot.
            figsize: Figure size.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        axes[0].imshow(image.astype(np.uint8))
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(gt_mask, cmap="tab20")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred_mask, cmap="tab20")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_batch(
        images: List[np.ndarray],
        masks: List[np.ndarray],
        figsize: Tuple[int, int] = (16, 10),
    ):
        """
        Plot a batch of images and masks.

        Args:
            images: List of images.
            masks: List of masks.
            figsize: Figure size.
        """
        n_samples = len(images)
        fig, axes = plt.subplots(n_samples, 2, figsize=figsize)

        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            axes[i, 0].imshow(images[i].astype(np.uint8))
            axes[i, 0].set_title(f"Image {i}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(masks[i], cmap="tab20")
            axes[i, 1].set_title(f"Mask {i}")
            axes[i, 1].axis("off")

        plt.tight_layout()
        return fig


class PathUtils:
    """Utility functions for path handling."""

    @staticmethod
    def ensure_dir(path: str) -> Path:
        """
        Ensure directory exists, create if it doesn't.

        Args:
            path: Directory path.

        Returns:
            Path object.
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def get_all_images(folder: str, extensions: List[str] = [".jpg", ".png"]) -> List[Path]:
        """
        Get all image files in a folder.

        Args:
            folder: Folder path.
            extensions: List of file extensions to look for.

        Returns:
            List of image paths.
        """
        folder_path = Path(folder)
        images = []
        for ext in extensions:
            images.extend(folder_path.glob(f"*{ext}"))
        return sorted(images)


def create_color_map(num_classes: int) -> np.ndarray:
    """
    Create a color map for visualization.

    Args:
        num_classes: Number of classes.

    Returns:
        Color map of shape (num_classes, 3).
    """
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3))
    colors[0] = [0, 0, 0]  # Background as black
    return colors


def apply_color_map(mask: np.ndarray, color_map: np.ndarray) -> np.ndarray:
    """
    Apply color map to a single-channel mask.

    Args:
        mask: Single-channel mask (H, W).
        color_map: Color map (num_classes, 3).

    Returns:
        Colored mask (H, W, 3).
    """
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id in range(len(color_map)):
        colored_mask[mask == class_id] = color_map[class_id]
    return colored_mask
