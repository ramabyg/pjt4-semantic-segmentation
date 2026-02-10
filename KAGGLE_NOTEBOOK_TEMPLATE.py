# Kaggle Notebook Setup Cell (Copy-Paste Ready)

"""
KAGGLE NOTEBOOK: Semantic Segmentation Project
This notebook clones the project from GitHub and sets up the environment for Kaggle GPU training.

Steps:
1. Copy this entire file
2. Create a new Kaggle notebook
3. Paste each cell into the Kaggle notebook
4. Replace YOUR_USERNAME with your GitHub username
5. Update DATA_PATH to point to your Kaggle dataset
6. Run cells sequentially
"""

# ============================================================================
# CELL 1: Clone Repository and Install Dependencies
# ============================================================================

import subprocess
import os
import sys
from pathlib import Path

# Clone from GitHub
REPO_URL = "https://github.com/YOUR_USERNAME/pjt4-semantic-segmentation.git"
PROJECT_DIR = "/kaggle/working/project"

# Remove if already exists
if os.path.exists(PROJECT_DIR):
    import shutil
    shutil.rmtree(PROJECT_DIR)

print("🔄 Cloning repository from GitHub...")
subprocess.run(["git", "clone", REPO_URL, PROJECT_DIR], check=True, capture_output=True)

os.chdir(PROJECT_DIR)
print("✓ Repository cloned successfully!")

print("\n🔧 Installing dependencies...")
subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)
print("✓ Dependencies installed!")

# ============================================================================
# CELL 2: Setup Environment and Verify Installation
# ============================================================================

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to Python path
sys.path.insert(0, str(Path.cwd() / "src"))

# Setup paths for Kaggle environment
PROJECT_ROOT = Path("/kaggle/working/project")
DATA_PATH = Path("/kaggle/input/aeroscapes-semantic-segmentation")  # Update with your dataset name
OUTPUT_DIR = Path("/kaggle/working/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Verify environment
print("=" * 60)
print("KAGGLE ENVIRONMENT SETUP")
print("=" * 60)
print(f"✓ Python Version: {sys.version.split()[0]}")
print(f"✓ PyTorch Version: {torch.__version__}")
print(f"✓ NumPy Version: {np.__version__}")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
print(f"✓ Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
print(f"✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# Verify project structure
print(f"\n✓ Project Root: {PROJECT_ROOT}")
print(f"✓ Data Path: {DATA_PATH}")
print(f"✓ Output Directory: {OUTPUT_DIR}")

# ============================================================================
# CELL 3: Import All Modules
# ============================================================================

from src.dataset import SemanticSegmentationDataset, get_default_transforms
from src.model import Detectron2Config, SemanticSegmentationConfig
from src.train import SemanticSegmentationTrainer
from src.inference import SemanticSegmentationPredictor, rle_encode
from src.utils import SegmentationMetrics, PathUtils
from src.config_loader import ConfigLoader, PerformanceTracker

import pandas as pd
import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("✓ All modules imported successfully!")

# ============================================================================
# CELL 4: Load Dataset
# ============================================================================

# Get image IDs from train.csv
image_ids_df = pd.read_csv(DATA_PATH / "train.csv")
imageIDs = image_ids_df["ImageID"].tolist()

class_names = [
    "Background", "Person", "Bike", "Car", "Drone", "Boat",
    "Animal", "Obstacle", "Construction", "Vegetation", "Road", "Sky"
]

# Create dataset instance
dataset = SemanticSegmentationDataset(
    data_path=DATA_PATH,
    images_folder="imgs/imgs",
    masks_folder="masks/masks",
    image_ids=imageIDs,
    class_names=class_names
)

print(f"✓ Total images in dataset: {len(dataset)}")

# Split into train and validation
train_ids, val_ids = SemanticSegmentationDataset.train_val_split(
    dataset.image_files,
    train_ratio=0.8,
    random_state=42
)

print(f"✓ Train set: {len(train_ids)} images")
print(f"✓ Validation set: {len(val_ids)} images")

# Create train and validation datasets with transforms
train_transforms, val_transforms = get_default_transforms(image_size=512)

train_dataset = SemanticSegmentationDataset(
    data_path=DATA_PATH,
    image_ids=train_ids,
    transforms=train_transforms,
    images_folder="imgs/imgs",
    masks_folder="masks/masks",
    class_names=class_names
)

val_dataset = SemanticSegmentationDataset(
    data_path=DATA_PATH,
    image_ids=val_ids,
    transforms=val_transforms,
    images_folder="imgs/imgs",
    masks_folder="masks/masks",
    class_names=class_names
)

print(f"✓ Train dataset: {len(train_dataset)}")
print(f"✓ Val dataset: {len(val_dataset)}")

# ============================================================================
# CELL 5: Configure Model
# ============================================================================

# Initialize Detectron2 Config for semantic segmentation
config = Detectron2Config(
    model_name="mask_rcnn_R_50_FPN_3x",
    num_classes=13,  # 12 classes + background
    num_gpus=1,
    batch_size_per_image=512,
    learning_rate=0.02,
    max_iter=10000,  # Adjust based on your needs
    checkpoint_period=500,
)

cfg = config.get_cfg()

print("Model Configuration:")
print(f"  ✓ Model: {cfg.MODEL.BACKBONE.NAME}")
print(f"  ✓ Num Classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
print(f"  ✓ Learning Rate: {cfg.SOLVER.BASE_LR}")
print(f"  ✓ Max Iterations: {cfg.SOLVER.MAX_ITER}")
print(f"  ✓ output directory: {OUTPUT_DIR}")

# ============================================================================
# CELL 6: Train Model
# ============================================================================

print("Starting training on Kaggle GPU...")
print("This may take a while. Monitor progress below.")

try:
    trainer = SemanticSegmentationTrainer(
        config=cfg,
        data_path=DATA_PATH,
        train_ids=train_ids,
        val_ids=val_ids,
        output_dir=str(OUTPUT_DIR)
    )

    # Train the model
    trainer.train()

    print("✓ Training complete!")
    print(f"✓ Model saved to: {OUTPUT_DIR}")

except Exception as e:
    print(f"❌ Training error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# CELL 7: Inference and Submission
# ============================================================================

# Load trained model for inference
model_path = OUTPUT_DIR / "model_final.pth"

if model_path.exists():
    print(f"✓ Loading trained model from: {model_path}")

    try:
        predictor = SemanticSegmentationPredictor(
            cfg_or_model_path=cfg,
            model_weights=str(model_path),
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        )

        print("✓ Model loaded successfully!")

        # Test on a few validation images
        print("\nTesting on validation samples...")
        sample_indices = range(min(3, len(val_dataset)))

        fig, axes = plt.subplots(len(sample_indices), 3, figsize=(15, 5*len(sample_indices)))
        if len(sample_indices) == 1:
            axes = axes.reshape(1, -1)

        for idx in sample_indices:
            image, gt_mask = val_dataset[idx]

            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy()
                image = (image * 255).astype(np.uint8)

            # Predict
            pred_mask, _ = predictor.predict(image)

            # Calculate metrics
            iou = SegmentationMetrics.iou_score(gt_mask.numpy() if torch.is_tensor(gt_mask) else gt_mask,
                                               pred_mask, num_classes=13)

            # Plot
            axes[idx, 0].imshow(image)
            axes[idx, 0].set_title("Input Image")
            axes[idx, 0].axis('off')

            axes[idx, 1].imshow(gt_mask if not torch.is_tensor(gt_mask) else gt_mask.numpy(), cmap='tab20')
            axes[idx, 1].set_title("Ground Truth")
            axes[idx, 1].axis('off')

            axes[idx, 2].imshow(pred_mask, cmap='tab20')
            axes[idx, 2].set_title(f"Prediction (IoU: {iou:.3f})")
            axes[idx, 2].axis('off')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "validation_samples.png", dpi=100, bbox_inches='tight')
        plt.show()

        print("✓ Validation visualization saved!")

    except Exception as e:
        print(f"❌ Inference error: {e}")
        import traceback
        traceback.print_exc()

else:
    print(f"❌ Model not found at: {model_path}")
    print("Please train the model first (run Cell 6)")

# ============================================================================
# CELL 8: Generate Submission CSV (if test data available)
# ============================================================================

# Prepare submission
test_data_path = DATA_PATH / "test_images"  # Update with your test folder

if test_data_path.exists():
    print(f"Generating submission from test set: {test_data_path}")

    test_images = sorted(test_data_path.glob("*.jpg"))
    submission_rows = []

    for img_path in test_images:
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Predict
        pred_mask, _ = predictor.predict(image_rgb)

        # RLE encode
        rle = rle_encode(pred_mask.astype(np.uint8))

        # Add to submission
        submission_rows.append({
            'ImageId': img_path.stem,
            'EncodedPixels': rle
        })

    # Save submission CSV
    submission_df = pd.DataFrame(submission_rows)
    submission_path = OUTPUT_DIR / "submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"✓ Submission CSV saved to: {submission_path}")
    print(f"✓ Total predictions: {len(submission_df)}")
    print(submission_df.head())

else:
    print(f"⚠ Test data not found at: {test_data_path}")
    print("Please upload your test images to the Kaggle dataset")

print("\n" + "="*60)
print("KAGGLE NOTEBOOK EXECUTION COMPLETE")
print("="*60)
