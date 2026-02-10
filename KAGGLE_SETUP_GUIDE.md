# Kaggle Notebook Setup Guide

## Quick Start on Kaggle

### Option A: Clone from GitHub (Recommended)

1. **Create a new Kaggle notebook** in your competition
2. **Add this setup cell** as the first code cell:

```python
# Cell 1: Clone repository and install dependencies
import subprocess
import os

# Clone from GitHub
subprocess.run(["git", "clone", "https://github.com/YOUR_USERNAME/pjt4-semantic-segmentation.git", "/kaggle/working/project"], check=True)
os.chdir("/kaggle/working/project")

# Install requirements
subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)

print("✓ Repository cloned and dependencies installed!")
```

3. **Add this import cell** right after:

```python
# Cell 2: Setup paths and imports
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / "src"))

# Import all modules
from src.dataset import SemanticSegmentationDataset, get_default_transforms
from src.model import Detectron2Config
from src.train import SemanticSegmentationTrainer
from src.inference import SemanticSegmentationPredictor
from src.utils import SegmentationMetrics, PathUtils
import torch

print(f"✓ All imports successful!")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
print(f"✓ Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

### Option B: Upload Files Directly

If you don't want to use GitHub:

1. Download all files from local folder
2. Upload to Kaggle as a dataset
3. Extract and import in notebook

---

## Key Differences: Local vs Kaggle

| Aspect | Local (Debugging) | Kaggle (Training) |
|--------|-------------------|-------------------|
| **GPU** | Not available | V100 available |
| **Working Dir** | `c:\rama\learn\pjt4-...` | `/kaggle/working/` |
| **Data Path** | `../data/...` | `/kaggle/input/...` |
| **Output Path** | `./outputs/` | `/kaggle/working/outputs/` |
| **Python Path** | Auto-added | Need to add manually |

---

## Path Handling Code

Use this in your notebook (works on both local and Kaggle):

```python
import os
from pathlib import Path

def get_env_paths():
    """Auto-detect environment and set paths"""
    if os.path.exists("/kaggle"):
        # Running on Kaggle
        ENV = "kaggle"
        PROJECT_ROOT = Path("/kaggle/working/project")
        DATA_PATH = Path("/kaggle/input/your-dataset-name")
        OUTPUT_DIR = Path("/kaggle/working/outputs")
    else:
        # Running locally
        ENV = "local"
        PROJECT_ROOT = Path.cwd()
        DATA_PATH = PROJECT_ROOT.parent / "data" / "opencv-segmentation-project"
        OUTPUT_DIR = PROJECT_ROOT / "outputs"

    OUTPUT_DIR.mkdir(exist_ok=True)
    return ENV, PROJECT_ROOT, DATA_PATH, OUTPUT_DIR

ENV, PROJECT_ROOT, DATA_PATH, OUTPUT_DIR = get_env_paths()
print(f"Environment: {ENV}")
print(f"Working Directory: {PROJECT_ROOT}")
print(f"Data Path: {DATA_PATH}")
print(f"Output Directory: {OUTPUT_DIR}")
```

---

## Training on Kaggle

Once setup is complete, training is identical to local:

```python
# Create dataset
dataset = SemanticSegmentationDataset(
    data_path=DATA_PATH,
    images_folder="imgs/imgs",
    masks_folder="masks/masks",
    image_ids=imageIDs,
    class_names=class_names
)

# Split data
train_ids, val_ids = SemanticSegmentationDataset.train_val_split(
    dataset.image_files,
    train_ratio=0.8,
    random_state=42
)

# Initialize trainer
trainer = SemanticSegmentationTrainer(
    config,
    DATA_PATH,
    train_ids,
    val_ids,
    OUTPUT_DIR
)

# Train
trainer.train()
```

---

## Debugging Workflow

1. **Develop and debug locally** (without GPU)
   - Test data loading, transforms, metrics
   - Verify notebook cells work correctly
   - Use small dataset samples for quick iterations

2. **Commit changes to GitHub**
   ```bash
   git add .
   git commit -m "Fix visualization, improve metrics"
   git push origin main
   ```

3. **Pull and run on Kaggle**
   - Kaggle notebook pulls latest code
   - Runs full training with GPU
   - Saves trained models and results

4. **Download results locally**
   - Pull submission CSV from Kaggle
   - Analyze metrics and plots
   - Refine and repeat

---

## Useful GPU Tips for Kaggle

- **Check GPU status**: `nvidia-smi`
- **Monitor during training**: Use epoch-wise logging
- **Save checkpoints**: Regularly save models
- **Avoid memory overflow**: Use smaller batch sizes initially
- **Use mixed precision**: Consider `torch.cuda.amp` for faster training

---

## Upload Dataset to Kaggle (One-time)

If your data isn't on Kaggle yet:

1. Create a Kaggle dataset
2. Upload your train images/masks
3. Reference in notebook as: `/kaggle/input/your-dataset-name`

---

## Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Add `sys.path.insert(0, str(Path.cwd()))` |
| Data not found | Use absolute paths with `/kaggle/input/` |
| GPU not detected | Kaggle detects automatically; verify with `torch.cuda.is_available()` |
| Out of memory | Reduce batch size in config |

---

**Next Steps:**
1. Push your code to GitHub *(if not already done)*
2. Create a Kaggle notebook
3. Add the setup cells above
4. Copy your notebook cells and adapt paths
5. Run training on Kaggle GPU!

