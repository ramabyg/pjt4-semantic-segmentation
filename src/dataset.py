"""
Custom Dataset class for semantic segmentation using Detectron2.
Supports custom data loading, augmentation, and train/val splitting.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Callable, Tuple
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SemanticSegmentationDataset(Dataset):
    """
    Generic Dataset class for semantic segmentation.

    Args:
        data_path (str): Path to the dataset folder.
        images_folder (str): Name of the folder containing the images.
        masks_folder (str): Name of the folder containing the masks.
        image_ids (List[str]): List of image IDs to use (e.g., for train/val split).
        transforms (Optional[Callable]): Albumentations transforms to apply.
        image_extension (str): Extension of image files (default: '.jpg').
        mask_extension (str): Extension of mask files (default: '.png').

    Attributes:
        image_files (List[str]): List of image file paths.
        mask_files (List[str]): List of mask file paths.
        class_names (Optional[List[str]]): Names of the classes.
    """

    def __init__(
        self,
        data_path: str,
        images_folder: str = "images",
        masks_folder: str = "masks",
        image_ids: Optional[List[str]] = None,
        transforms: Optional[Callable] = None,
        image_extension: str = ".jpg",
        mask_extension: str = ".png",
        class_names: Optional[List[str]] = None,
    ):
        self.data_path = Path(data_path)
        self.images_folder = self.data_path / images_folder
        self.masks_folder = self.data_path / masks_folder
        self.transforms = transforms
        self.class_names = class_names

        # Get all image files
        if image_ids is None:
            self.image_files = sorted([
                f.stem for f in self.images_folder.glob(f"*{image_extension}")
            ])
        else:
            self.image_files = image_ids

        self.image_extension = image_extension
        self.mask_extension = mask_extension

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get image and mask at given index.

        Returns:
            Tuple of (image, mask) as tensors
        """
        image_id = self.image_files[idx]

        # Load image
        image_path = self.images_folder / f"{image_id}{self.image_extension}"
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_path = self.masks_folder / f"{image_id}{self.mask_extension}"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Apply transforms
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            # Convert to tensor if no transforms
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask

    @staticmethod
    def train_val_split(
        image_ids: List[str],
        train_ratio: float = 0.8,
        random_state: int = 42,
    ) -> Tuple[List[str], List[str]]:
        """
        Split image IDs into train and validation sets.

        Args:
            image_ids: List of image IDs.
            train_ratio: Fraction to use for training (default: 0.8).
            random_state: Random seed for reproducibility.

        Returns:
            Tuple of (train_ids, val_ids)
        """
        import random
        random.seed(random_state)
        
        # Use plain Python list to preserve string types
        image_ids_list = list(image_ids)
        random.shuffle(image_ids_list)

        split_idx = int(len(image_ids_list) * train_ratio)
        train_ids = image_ids_list[:split_idx]
        val_ids = image_ids_list[split_idx:]

        return train_ids, val_ids


def get_default_transforms(image_size: int = 512) -> Tuple[A.Compose, A.Compose]:
    """
    Get default augmentation transforms for train and validation.

    Args:
        image_size: Size to resize images to.

    Returns:
        Tuple of (train_transforms, val_transforms)
    """
    train_transforms = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(p=0.8),
                A.RandomBrightnessContrast(p=0.2),
            ], p=0.3),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ],
        is_check_shapes=False,
    )

    val_transforms = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(),
            ToTensorV2(),
        ],
        is_check_shapes=False,
    )

    return train_transforms, val_transforms
