# Configuration & Performance Tracking Guide

## Overview

The semantic segmentation project includes a professional configuration management system and automatic performance tracking for easy experiment management.

## Configuration System

### 1. YAML-Based Configurations

Configurations are stored in `configs/` folder as YAML files, making it easy to:
- Switch between different models
- Adjust hyperparameters
- Track different experimental setups

### 2. Available Configurations

#### `config_baseline.yaml`
- **Model**: Mask R-CNN with ResNet-50 backbone
- **FPN**: Yes
- **Image Size**: 512x512
- **Max Iterations**: 10,000
- **Learning Rate**: 0.02
- **Batch Size**: 4
- Use for: Standard baseline experiments

#### `config_large_model.yaml`
- **Model**: Mask R-CNN with ResNet-101 backbone (larger)
- **Image Size**: 768x768
- **Max Iterations**: 15,000
- **Learning Rate**: 0.01
- **Batch Size**: 2 (due to larger model)
- Use for: Better accuracy at cost of more GPU memory

#### `config_fast_train.yaml`
- **Model**: Mask R-CNN with ResNet-50 backbone
- **Image Size**: 384x384 (smaller)
- **Max Iterations**: 5,000
- **Batch Size**: 8
- **Learning Rate**: 0.02
- Use for: Quick experiments and debugging

### 3. Creating Custom Configurations

Create a new YAML file in `configs/`:

```yaml
experiment_name: "my_experiment"
model:
  name: "Mask R-CNN R50-FPN"
  model_name: "mask_rcnn_R_50_FPN_3x"
  num_classes: 13
  num_gpus: 1
  batch_size_per_image: 512
  pretrained: true

training:
  learning_rate: 0.02
  max_iter: 10000
  batch_size: 4
  momentum: 0.9
  weight_decay: 0.0001
  warmup_iters: 1000
  lr_decay_milestones: [5000, 7500]
  lr_decay_gamma: 0.1
  checkpoint_period: 500
  num_workers: 4
  patience: 10

data:
  data_path: "./data"
  images_folder: "images"
  masks_folder: "masks"
  train_ratio: 0.8
  image_size: 512
  num_classes: 13
  class_names:
    - "background"
    - "building"
    - ... etc

output_dir: "./outputs"
seed: 42
device: "cuda"
```

### 4. Loading Configurations in Code

**Option 1: Load from file**
```python
from src.config_loader import ConfigLoader

experiment_config = ConfigLoader.load_experiment_config("./configs/config_baseline.yaml")
```

**Option 2: Create programmatically**
```python
from src.config_loader import ExperimentConfig, ModelConfig, TrainingConfig, DataConfig

experiment_config = ExperimentConfig(
    experiment_name="my_exp",
    model=ModelConfig(...),
    training=TrainingConfig(...),
    data=DataConfig(...)
)
```

**Option 3: Save after modifications**
```python
ConfigLoader.save_config(experiment_config, "./configs/my_new_config.yaml")
```

## Performance Tracking

### 1. Initialize Tracker

```python
from src.config_loader import PerformanceTracker

tracker = PerformanceTracker(
    output_dir="./outputs",
    experiment_name="baseline_maskrcnn"
)
```

### 2. Log Metrics During Training

**Per Iteration**
```python
# After validation epoch
tracker.log_iteration(
    iteration=epoch,
    metrics={
        'loss': val_loss,
        'iou': val_iou,
        'dice': val_dice,
    },
    stage='val'
)
```

**Per Batch** (optional, for monitoring training)
```python
# During training
tracker.log_batch(
    iteration=epoch,
    batch_idx=batch_idx,
    metrics={'loss': batch_loss}
)
```

### 3. Save Results

```python
# Save to JSON
tracker.save_metrics_json()

# Save to CSV
tracker.save_metrics_csv()

# Generate text summary
tracker.save_summary()

# Plot metrics
tracker.plot_metrics(metrics_to_plot=['loss', 'iou', 'dice'])
```

### 4. Query Results

```python
# Get best metric value
best_iou = tracker.get_best_metrics('iou', stage='val')
print(f"Best IoU: {best_iou['iou']}")
print(f"Achieved at iteration: {best_iou['iteration']}")
```

## Output Structure

After training, your outputs directory will look like:

```
outputs/
├── baseline_maskrcnn/
│   ├── config.yaml                    # Saved configuration
│   ├── metrics/
│   │   └── baseline_maskrcnn/
│   │       ├── metrics_20260207_143022.json      # Raw metrics
│   │       ├── metrics_20260207_143022.csv       # Metrics as CSV
│   │       ├── summary_20260207_143022.txt       # Text summary
│       └── metrics_plot_20260207_143022.png      # Visualization
│   ├── model_final.pth                # Trained model
│   └── log.txt                        # Training logs
```

## Example Workflow

### 1. Run Baseline Experiment

```python
# Load baseline config
config = ConfigLoader.load_experiment_config("./configs/config_baseline.yaml")

# Initialize tracker
tracker = PerformanceTracker("./outputs", config.experiment_name)

# Train model (pseudo-code)
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_metrics = validate_epoch(model, val_loader)

    tracker.log_iteration(epoch, val_metrics, stage='val')

# Save results
tracker.save_metrics_json()
tracker.save_metrics_csv()
tracker.save_summary()
tracker.plot_metrics()
```

### 2. Run Experiment with Different Config

```python
# Just change the config file
config = ConfigLoader.load_experiment_config("./configs/config_large_model.yaml")

# Rest of code is identical
# Metrics are automatically saved to different directory
```

### 3. Compare Experiments

```python
import pandas as pd
import json
from pathlib import Path

results = {}

for exp_name in ['baseline_maskrcnn_r50', 'large_maskrcnn_r101', 'fast_training_r50']:
    metrics_file = Path(f"./outputs/{exp_name}/metrics/baseline_maskrcnn_r50/metrics_*.json")
    # Load and parse JSON
    # Extract best metrics

comparison_df = pd.DataFrame(results)
print(comparison_df)
```

## Tips & Best Practices

1. **Use version control for configs**
   ```bash
   git add configs/
   git commit -m "Add new baseline configuration"
   ```

2. **Track all experiments**
   - Always use performance tracker
   - Save config with results
   - Document any custom modifications

3. **Compare systematically**
   - Change one parameter at a time
   - Use clear experiment names
   - Track hyperparameter values

4. **Monitor during training**
   - Log per-batch loss for debugging
   - Check validation metrics frequently
   - Use early stopping based on best metric

5. **Document results**
   - Note important findings in summary
   - Keep analysis in notebook
   - Share comparisons with team

## Integration with Kaggle

### On Kaggle:

```python
from src.config_loader import ConfigLoader, PerformanceTracker

# Load config
config = ConfigLoader.load_experiment_config("configs/config_baseline.yaml")

# Update paths if needed
config.data.data_path = "../input/your-dataset"
config.output_dir = "./"

# Initialize tracker
tracker = PerformanceTracker("./", config.experiment_name)

# Train and track
# ... training code ...

# Save results
tracker.save_metrics_json()
tracker.save_summary()

# Download results from Kaggle Output folder
```

## Architecture Classes

### `ExperimentConfig`
Top-level configuration container

### `ModelConfig`
Model-specific settings (architecture, num_classes, etc.)

### `TrainingConfig`
Training hyperparameters (learning_rate, batch_size, etc.)

### `DataConfig`
Data-specific settings (paths, image_size, etc.)

### `ConfigLoader`
Static methods for loading/saving configs from YAML

### `PerformanceTracker`
Logs, saves, and analyzes training metrics

## Troubleshooting

**Q: YAML file not found**
```
A: Check that config file is in ./configs/ folder with correct name
```

**Q: Metrics not saving**
```
A: Check that output directory exists and is writable
   Path(output_dir).mkdir(parents=True, exist_ok=True)
```

**Q: Import errors**
```
A: Ensure all dependencies installed:
   pip install -r requirements.txt
```

## Next Steps

1. Create your own configurations for different models
2. Train and track multiple experiments
3. Use metrics comparison to find best configuration
4. Document results and findings
5. Push code and results to GitHub/Kaggle
