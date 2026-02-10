# Semantic Segmentation with Detectron2

A modular, production-ready implementation of semantic segmentation using Detectron2. Designed for local development in VS Code and training on Kaggle GPU resources.

## Project Structure

```
pjt4-semantic-segmentation/
├── Project4-Semantic-Segmentation.ipynb    # Main notebook (orchestrates training)
├── requirements.txt                        # Dependencies
├── README.md                              # This file
├── CONFIG_AND_TRACKING_GUIDE.md           # Configuration & tracking guide
├── configs/                               # YAML configuration files
│   ├── config_baseline.yaml               # Baseline Mask R-CNN config
│   ├── config_large_model.yaml            # Larger model (R-101)
│   └── config_fast_train.yaml             # Fast training config
├── src/                                   # Source code modules
│   ├── __init__.py                       # Package initialization
│   ├── dataset.py                        # Dataset class and transforms
│   ├── model.py                          # Model configuration (Detectron2)
│   ├── train.py                          # Training logic
│   ├── inference.py                      # Prediction and submission generation
│   ├── utils.py                          # Utility functions and metrics
│   └── config_loader.py                  # Configuration & performance tracking
├── data/                                 # Dataset directory (create locally)
│   ├── images/                           # Training/val images
│   └── masks/                            # Ground truth masks
└── outputs/                              # Model checkpoints and logs
    ├── {experiment_name}/
    │   ├── config.yaml                   # Saved configuration
    │   ├── metrics/                      # Performance metrics
    │   ├── model_final.pth
    │   └── log.txt
```

## Configuration Files

Pre-built YAML configurations for common use cases:

### `configs/config_baseline.yaml`
- Standard Mask R-CNN (ResNet-50 backbone)
- Good balance of accuracy and speed
- Recommended for: Most experiments

### `configs/config_large_model.yaml`
- Larger Mask R-CNN (ResNet-101 backbone)
- Better accuracy, slower training
- Recommended for: Fine-tuning, production models

### `configs/config_fast_train.yaml`
- Smaller image size, fewer iterations
- Quick training, good for debugging
- Recommended for: Prototyping, local testing

**To use a config:**
```python
from src.config_loader import ConfigLoader

config = ConfigLoader.load_experiment_config("configs/config_baseline.yaml")
```

## Performance Tracking

Automatically track and compare experiments:

```python
from src.config_loader import PerformanceTracker

# Initialize tracker
tracker = PerformanceTracker("./outputs", "baseline_maskrcnn")

# Log metrics during training
for epoch in range(num_epochs):
    val_metrics = validate(model, val_loader)
    tracker.log_iteration(epoch, val_metrics, stage='val')

# Save and analyze results
tracker.save_metrics_json()
tracker.save_metrics_csv()
tracker.save_summary()           # Generate text report
tracker.plot_metrics()           # Create visualization
```

**Output files:**
- `metrics_*.json` - All metrics as JSON
- `metrics_*.csv` - Metrics as CSV (easy for pandas)
- `summary_*.txt` - Human-readable summary
- `metrics_plot_*.png` - Training curves visualization

See [CONFIG_AND_TRACKING_GUIDE.md](CONFIG_AND_TRACKING_GUIDE.md) for detailed examples.

## Installation

### Local Setup (for development and debugging)

1. **Clone/Download the project**
   ```bash
   cd pjt4-semantic-segmentation
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: Detectron2 installation can be tricky. If you have issues:
   ```bash
   # For CPU (faster setup, good for development)
   pip install torch torchvision
   pip install "git+https://github.com/facebookresearch/detectron2.git"

   # For CUDA GPU (needed for Kaggle)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install "git+https://github.com/facebookresearch/detectron2.git"
   ```

4. **Prepare your data**
   - Create `data/` folder with structure:
     ```
     data/
     ├── images/    # Your image files (.jpg)
     └── masks/     # Your segmentation masks (.png)
     ```
   - Ensure image and mask filenames match (e.g., `image_001.jpg` → `image_001.png`)

## Usage

### Local Development Workflow

#### 1. Explore Data & Build Model Locally
```python
# In notebook or Python script
from src.dataset import SemanticSegmentationDataset, get_default_transforms
from src.utils import VisualizationUtils

# Load dataset
dataset = SemanticSegmentationDataset(
    data_path="./data",
    images_folder="images",
    masks_folder="masks"
)

# Split train/val
train_ids, val_ids = SemanticSegmentationDataset.train_val_split(
    dataset.image_files,
    train_ratio=0.8,
    random_state=42
)
```

#### 2. Load Configuration
```python
from src.config_loader import ConfigLoader, PerformanceTracker

# Load config (or create custom)
config = ConfigLoader.load_experiment_config("configs/config_baseline.yaml")

# Initialize performance tracker
tracker = PerformanceTracker(config.output_dir, config.experiment_name)
```

#### 3. Configure Model (using YAML config)
```python
# Config is already loaded, use it to set up model
model_config = config.model
training_config = config.training

# Create Detectron2 config from loaded configuration
from src.model import Detectron2Config

detectron2_cfg = Detectron2Config(
    model_name=model_config.model_name,
    num_classes=model_config.num_classes,
    learning_rate=training_config.learning_rate,
    max_iter=training_config.max_iter,
)
```

#### 4. Train with Performance Tracking
```python
# During training loop, log metrics
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, scheduler)
    val_metrics = validate_epoch(model, val_loader)

    # Log validation metrics
    tracker.log_iteration(epoch, val_metrics, stage='val')

    # Save checkpoint if it's the best
    if val_metrics['iou'] > best_iou:
        best_iou = val_metrics['iou']
        torch.save(model.state_dict(), f"{tracker.metrics_dir}/best_model.pth")

# Save all results
tracker.save_metrics_json()
tracker.save_metrics_csv()
tracker.save_summary()
tracker.plot_metrics()
```

#### 5. Compare Multiple Experiments
```python
# Train with different configs and compare
configs = [
    'config_baseline.yaml',
    'config_large_model.yaml',
    'config_fast_train.yaml'
]

for config_file in configs:
    config = ConfigLoader.load_experiment_config(f"configs/{config_file}")

    # Train model
    tracker = PerformanceTracker(config.output_dir, config.experiment_name)
    # ... training code ...

    # Results automatically saved to different directories
```

### Training on Kaggle
Once you're satisfied with your setup locally:

1. **Push to GitHub** (recommended):
   ```bash
   git add .
   git commit -m "Initial project setup"
   git push origin main
   ```

2. **On Kaggle**:
   - Clone the repo in a Kaggle notebook:
     ```python
     !git clone https://github.com/yourusername/your-repo.git
     %cd your-repo
     !pip install -r requirements.txt
     ```

   - Or upload the `src/` folder directly to Kaggle Datasets

   - Import and train:
     ```python
     import sys
     sys.path.append('../input/your-dataset/pjt4-semantic-segmentation')

     from src.train import train_detectron2_model
     from src.model import Detectron2Config
     # ... configure and train
     ```

### Making Predictions

```python
from src.inference import SemanticSegmentationPredictor

# Load trained model
predictor = SemanticSegmentationPredictor(
    cfg_or_model_path="./outputs/config.yaml",  # or cfg object
    model_weights="./outputs/model_final.pth",
    device="cuda"
)

# Single image prediction
import cv2
image = cv2.imread("path/to/image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pred_mask, metadata = predictor.predict(image)

# Batch prediction
predictions = predictor.predict_on_folder(
    folder_path="./data/test_images",
    output_folder="./predictions"
)

# Create submission
from src.inference import create_submission_csv
create_submission_csv(predictions, output_path="submission.csv")
```

## API Reference

### Configuration Management

```python
from src.config_loader import ConfigLoader, PerformanceTracker

# Load configuration from YAML
config = ConfigLoader.load_experiment_config("configs/config_baseline.yaml")

# Initialize performance tracker
tracker = PerformanceTracker("./outputs", config.experiment_name)

# Log metrics
tracker.log_iteration(epoch, {'loss': 0.5, 'iou': 0.65}, stage='val')

# Save results
tracker.save_metrics_json()
tracker.save_metrics_csv()
tracker.save_summary()
tracker.plot_metrics()

# Query best metrics
best_iou = tracker.get_best_metrics('iou', stage='val')
```

See [CONFIG_AND_TRACKING_GUIDE.md](CONFIG_AND_TRACKING_GUIDE.md) for detailed configuration documentation.

### Dataset Module (`src/dataset.py`)
- **`SemanticSegmentationDataset`**: Custom PyTorch Dataset
- **`get_default_transforms()`**: Returns train/val augmentation transforms

### Model Module (`src/model.py`)
- **`Detectron2Config`**: Wrapper for Detectron2 configuration
- **`SemanticSegmentationConfig`**: Simple config class

### Training Module (`src/train.py`)
- **`SemanticSegmentationTrainer`**: Trainer class
- **`train_detectron2_model()`**: Standalone training function

### Inference Module (`src/inference.py`)
- **`SemanticSegmentationPredictor`**: Make predictions
- **`create_submission_csv()`**: Generate submission file
- **`rle_encode()`/`rle_decode()`**: RLE encoding utilities

### Utils Module (`src/utils.py`)
- **`SegmentationMetrics`**: IoU, Dice, accuracy metrics
- **`VisualizationUtils`**: Plotting functions
- **`PathUtils`**: Path utilities

## Tips for Kaggle

1. **Use Detectron2 pre-trained models**: Faster convergence
2. **Data augmentation**: Albumentations for excellent augmentations
3. **Ensemble**: Train multiple seeds and ensemble predictions
4. **Batch size**: Use larger batches on Kaggle GPU (adjust in config)
5. **Early stopping**: Monitor validation metrics, save best model
6. **Mixed precision**: Use `torch.cuda.amp` for faster training

## Troubleshooting

**Detectron2 installation issues:**
- Try CPU version first, then upgrade to CUDA
- On Windows, you might need Visual Studio C++ build tools

**Memory issues:**
- Reduce image size in transforms
- Reduce batch size in config
- Use `torch.cuda.empty_cache()` in notebook

**Data not found:**
- Check file extensions match (`.jpg` vs `.JPG`)
- Verify folder structure matches config
- Use absolute paths if relative paths fail

## References

- [Detectron2 Documentation](https://detectron2.readthedocs.io/)
- [Albumentations Documentation](https://albumentations.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## License

Your License Here

## Author

Your Name / Organization
