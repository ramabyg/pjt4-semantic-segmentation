# New Features: Configuration & Performance Tracking

## What's New

### 1. **YAML-Based Configuration System** ✨
- Store model and training configurations in YAML files
- Easy switching between different setups
- Version control friendly
- Pre-built configurations included

### 2. **Automatic Performance Tracking** 📊
- Automatically log metrics during training
- Export to JSON, CSV, or text format
- Generate training summaries and plots
- Track best metrics across iterations

### 3. **Experiment Comparison Tools** 📈
- Compare metrics across multiple experiments
- Generate comparison reports (CSV/HTML)
- Visualize results
- Find best performing configuration

### 4. **Helper Scripts** 🔧
- `train_with_config.py` - Train using YAML config
- `compare_experiments.py` - Compare multiple runs

## Files Added

### Configuration Files
```
configs/
├── config_baseline.yaml       # Mask R-CNN R50-FPN (recommended)
├── config_large_model.yaml    # Mask R-CNN R101-FPN (better accuracy)
└── config_fast_train.yaml     # Fast training (debugging)
```

### Source Code
- `src/config_loader.py` - Configuration system and performance tracking

### Documentation
- `CONFIG_AND_TRACKING_GUIDE.md` - Detailed guide

### Scripts
- `train_with_config.py` - Training script using YAML configs
- `compare_experiments.py` - Compare multiple experiments

### Updated Files
- `src/__init__.py` - Export new classes
- `requirements.txt` - Added PyYAML dependency
- `README.md` - Updated with new features
- `Project4-Semantic-Segmentation.ipynb` - Added example cells

## Quick Start

### 1. Load Configuration
```python
from src.config_loader import ConfigLoader

config = ConfigLoader.load_experiment_config("configs/config_baseline.yaml")
```

### 2. Initialize Tracker
```python
from src.config_loader import PerformanceTracker

tracker = PerformanceTracker("./outputs", config.experiment_name)
```

### 3. Log Metrics
```python
for epoch in range(num_epochs):
    val_metrics = validate(model, val_loader)
    tracker.log_iteration(epoch, val_metrics, stage='val')
```

### 4. Save Results
```python
tracker.save_metrics_json()
tracker.save_metrics_csv()
tracker.save_summary()
tracker.plot_metrics()
```

### 5. Compare Experiments
```python
from compare_experiments import compare_experiments, generate_comparison_report

comparison_df = compare_experiments(
    ['baseline_maskrcnn', 'large_maskrcnn'],
    output_dir='./outputs'
)
generate_comparison_report(comparison_df, 'comparison.csv')
```

## Configuration Classes

### `ExperimentConfig`
```python
@dataclass
class ExperimentConfig:
    experiment_name: str
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    output_dir: str = "./outputs"
    seed: int = 42
    device: str = "cuda"
```

### `ModelConfig`
```python
@dataclass
class ModelConfig:
    name: str
    model_name: str  # detectron2 model name
    num_classes: int
    num_gpus: int
    batch_size_per_image: int
    backbone: Optional[str] = None
    pretrained: bool = True
```

### `TrainingConfig`
```python
@dataclass
class TrainingConfig:
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
    patience: int = 10
```

### `DataConfig`
```python
@dataclass
class DataConfig:
    data_path: str
    images_folder: str = "images"
    masks_folder: str = "masks"
    train_ratio: float = 0.8
    image_size: int = 512
    num_classes: int = 13
    class_names: list = None
```

## Core Classes

### `ConfigLoader`
Static methods for loading/saving configurations:
- `load_yaml(yaml_path)` - Load YAML file
- `load_experiment_config(yaml_path)` - Load full experiment config
- `save_config(config, output_path)` - Save config to YAML

### `PerformanceTracker`
Track and save metrics:
- `log_iteration(iteration, metrics, stage)` - Log per-epoch metrics
- `log_batch(iteration, batch_idx, metrics)` - Log per-batch metrics
- `save_metrics_json()` - Export to JSON
- `save_metrics_csv()` - Export to CSV
- `save_summary()` - Generate text summary
- `plot_metrics(metrics_to_plot)` - Visualize metrics
- `get_best_metrics(metric_name, stage)` - Query best values

## Output Structure

After training with configuration system:

```
outputs/
└── baseline_maskrcnn_r50/
    ├── config.yaml                    # Saved configuration
    ├── metrics/
    │   ├── baseline_maskrcnn_r50/
    │   │   ├── metrics_20260207_143022.json
    │   │   ├── metrics_20260207_143022.csv
    │   │   ├── summary_20260207_143022.txt
    │   │   └── metrics_plot_20260207_143022.png
    │   └── experiment_config.yaml
    ├── model_final.pth
    └── log.txt
```

## Typical Workflow

### Step 1: Baseline Experiment
```bash
# Run baseline configuration
python train_with_config.py --config configs/config_baseline.yaml

# Check results
tensorboard --logdir outputs/baseline_maskrcnn_r50/metrics
```

### Step 2: Try Larger Model
```bash
# Run with larger model
python train_with_config.py --config configs/config_large_model.yaml
```

### Step 3: Compare Results
```bash
# Compare all experiments
python compare_experiments.py --output comparison_report.csv

# Or plot specific metric
python compare_experiments.py --plot-metric iou_best
```

### Step 4: Deploy Best Model
```python
# Load best model
checkpoint = torch.load("outputs/baseline_maskrcnn_r50/model_final.pth")
model.load_state_dict(checkpoint)
```

## Integration with Kaggle

### On Kaggle Notebook:

```python
# 1. Clone repo or upload files
!git clone https://github.com/yourname/semseg-project.git
cd semseg-project

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Use configuration system
from src.config_loader import ConfigLoader, PerformanceTracker

config = ConfigLoader.load_experiment_config("configs/config_baseline.yaml")
config.data.data_path = "../input/your-dataset/data"
config.output_dir = "./"

# 4. Train and track
tracker = PerformanceTracker("./", config.experiment_name)
# ... training code ...

# 5. Download results
# Results automatically saved to ./metrics/ folder
```

## Benefits

✅ **Easy Configuration Management**
- One YAML file per experiment
- Reproducible results
- Easy sharing and version control

✅ **Automatic Metric Tracking**
- No manual logging needed
- Consistent file format
- Ready-to-use summaries

✅ **Professional Experiment Management**
- Compare experiments easily
- Track performance over time
- Generate reports automatically

✅ **Kaggle Ready**
- Works locally and on Kaggle
- Portable configurations
- Result tracking for competition monitoring

## Example YAML Configuration

```yaml
experiment_name: "baseline_maskrcnn"
model:
  name: "Mask R-CNN R50-FPN"
  model_name: "mask_rcnn_R_50_FPN_3x"
  num_classes: 13
  num_gpus: 1
  batch_size_per_image: 512

training:
  learning_rate: 0.02
  max_iter: 10000
  batch_size: 4
  momentum: 0.9
  weight_decay: 0.0001
  lr_decay_milestones: [5000, 7500]
  checkpoint_period: 500

data:
  data_path: "./data"
  images_folder: "images"
  masks_folder: "masks"
  image_size: 512
  num_classes: 13

output_dir: "./outputs"
seed: 42
device: "cuda"
```

## Tips & Best Practices

1. **Create configs for each experiment variant**
   - Different batch sizes
   - Different learning rates
   - Different model architectures
   - Different epochs

2. **Use meaningful experiment names**
   ```yaml
   experiment_name: "maskrcnn_r50_lr001_bs4"  # ✓ Good
   experiment_name: "exp1"                     # ✗ Not clear
   ```

3. **Track everything**
   - Always initialize tracker
   - Always save metrics
   - Always generate summaries

4. **Compare systematically**
   - Change one parameter at a time
   - Use comparison tools
   - Document findings

5. **Version control your configs**
   ```bash
   git add configs/
   git commit -m "Add baseline and large model configs"
   ```

## Troubleshooting

**Q: YAML file not found**
```
A: Ensure file is in ./configs/ with correct path
   python train_with_config.py --config configs/config_baseline.yaml
```

**Q: Metrics not saving**
```
A: Check output directory exists and is writable
   Verify tracker is initialized properly
```

**Q: Import errors**
```
A: Install missing dependency
   pip install pyyaml
```

## Next Steps

1. ✅ Try the baseline configuration
2. ✅ Run training and check metrics
3. ✅ Create custom configuration
4. ✅ Compare multiple experiments
5. ✅ Analyze results and document findings
6. ✅ Submit best model to Kaggle
