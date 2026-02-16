"""
Inference script for semantic segmentation with Detectron2.
Used for making predictions on test data and generating submissions.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Tuple
import logging

from detectron2.config import get_cfg, CfgNode
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import transforms as T

logger = logging.getLogger(__name__)


class SemanticSegmentationPredictor:
    """
    Predictor class for semantic segmentation.

    Can load a trained Detectron2 model and make predictions on images.
    """

    def __init__(
        self,
        cfg_or_model_path: Union[str, CfgNode],
        model_weights: Optional[str] = None,
        confidence_threshold: float = 0.5,
        device: str = "cuda",
    ):
        """
        Initialize predictor.

        Args:
            cfg_or_model_path: Either path to config file or a Detectron2 config object.
            model_weights: Path to model weights file.
            confidence_threshold: Confidence threshold for predictions.
            device: Device to use ('cuda' or 'cpu').
        """
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Load config
        if isinstance(cfg_or_model_path, str):
            self.cfg = get_cfg()
            self.cfg.merge_from_file(cfg_or_model_path)
        else:
            self.cfg = cfg_or_model_path

        self.cfg.MODEL.DEVICE = device

        # Load model
        self.model = build_model(self.cfg)
        self.model.eval()

        if model_weights:
            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(model_weights)
            logger.info(f"Loaded model weights from {model_weights}")

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Make prediction on a single image.

        Args:
            image: Input image (H, W, 3) in RGB format.

        Returns:
            Tuple of (predicted_mask, metadata)
        """
        # Handle grayscale images
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        # Prepare input
        height, width = image.shape[:2]
        aug_input = T.AugInput(image)
        aug_input.apply_augmentations(
            [T.ResizeShortestEdge(
                [self.cfg.INPUT.MIN_SIZE_TEST],
                self.cfg.INPUT.MAX_SIZE_TEST,
                sample_style="choice",
            )]
        )
        image_tensor = torch.as_tensor(
            aug_input.image.astype("float32").transpose(2, 0, 1)
        )

        # Make prediction
        inputs = {"image": image_tensor, "height": height, "width": width}
        outputs = self.model([inputs])[0]

        # Extract segmentation results
        if "sem_seg" in outputs:
            # Semantic segmentation output
            sem_seg = outputs["sem_seg"].argmax(dim=0).cpu().numpy()
            metadata = {
                "sem_seg": sem_seg,
                "height": height,
                "width": width,
            }
        elif "instances" in outputs:
            # Instance segmentation output
            instances = outputs["instances"]
            pred_masks = instances.pred_masks.cpu().numpy()
            pred_classes = instances.pred_classes.cpu().numpy()
            scores = instances.scores.cpu().numpy()

            # Create semantic segmentation from instances
            sem_seg = np.zeros((height, width), dtype=np.int32)
            for i, (mask, cls, score) in enumerate(zip(pred_masks, pred_classes, scores)):
                if score >= self.confidence_threshold:
                    sem_seg[mask > 0.5] = cls + 1  # +1 to avoid background class 0

            metadata = {
                "sem_seg": sem_seg,
                "pred_masks": pred_masks,
                "pred_classes": pred_classes,
                "scores": scores,
                "height": height,
                "width": width,
            }
        else:
            raise ValueError(f"Unknown output format: {outputs.keys()}")

        return sem_seg, metadata

    def predict_batch(
        self,
        images: List[np.ndarray],
        return_metadata: bool = False,
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[dict]]]:
        """
        Make predictions on a batch of images.

        Args:
            images: List of input images.
            return_metadata: Whether to return metadata.

        Returns:
            List of predicted masks, optionally with metadata.
        """
        predictions = []
        metadatas = []

        for image in images:
            pred_mask, metadata = self.predict(image)
            predictions.append(pred_mask)
            metadatas.append(metadata)

        if return_metadata:
            return predictions, metadatas
        return predictions

    def predict_on_folder(
        self,
        folder_path: str,
        output_folder: Optional[str] = None,
        extensions: List[str] = [".jpg", ".png"],
    ) -> dict:
        """
        Make predictions on all images in a folder.

        Args:
            folder_path: Path to image folder.
            output_folder: Path to save predictions. If None, don't save.
            extensions: Image extensions to look for.

        Returns:
            Dictionary mapping image names to predictions.
        """
        folder = Path(folder_path)
        predictions = {}

        # Get all images
        image_paths = []
        for ext in extensions:
            image_paths.extend(folder.glob(f"*{ext}"))

        logger.info(f"Found {len(image_paths)} images in {folder_path}")

        for image_path in image_paths:
            # Load image
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Predict
            pred_mask, _ = self.predict(image)

            # Save if output folder is specified
            if output_folder:
                Path(output_folder).mkdir(parents=True, exist_ok=True)
                output_path = Path(output_folder) / f"{image_path.stem}.png"
                cv2.imwrite(str(output_path), pred_mask.astype(np.uint8))

            predictions[image_path.stem] = pred_mask

        logger.info(f"Predictions completed. Processed {len(predictions)} images.")

        return predictions


def create_submission_csv(
    predictions: dict,
    output_path: str = "submission.csv",
    class_mapping: Optional[dict] = None,
):
    """
    Create submission CSV from predictions.

    Args:
        predictions: Dictionary mapping image names to predicted masks.
        output_path: Path to save submission CSV.
        class_mapping: Optional mapping of class indices to class names.
    """
    import csv

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ImageId", "EncodedPixels"])

        for image_id, mask in predictions.items():
            # RLE encode the mask
            encoded = rle_encode(mask)
            writer.writerow([image_id, encoded])

    logger.info(f"Submission CSV saved to {output_path}")


def rle_encode(mask: np.ndarray) -> str:
    """
    Encode mask as RLE (Run Length Encoding).

    Args:
        mask: Binary or multi-class mask.

    Returns:
        RLE encoded string.
    """
    # Flatten the mask
    mask = mask.astype(np.uint8)
    pixels = mask.flatten()

    # Find transitions
    pads = np.concatenate(([0], np.diff(pixels), [0]))
    runs = np.where(pads != 0)[0] + 1

    # Calculate run lengths
    runs[1::2] -= runs[::2]

    return ' '.join(map(str, runs))


def rle_decode(encoded_str: str, shape: Tuple[int, int]) -> np.ndarray:
    """
    Decode RLE string back to mask.

    Args:
        encoded_str: RLE encoded string.
        shape: Original shape (height, width).

    Returns:
        Decoded mask.
    """
    if not encoded_str or encoded_str == '':
        return np.zeros(np.prod(shape), dtype=np.uint8).reshape(shape)

    s = encoded_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1

    ends = starts + lengths
    img = np.zeros(np.prod(shape), dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape)
