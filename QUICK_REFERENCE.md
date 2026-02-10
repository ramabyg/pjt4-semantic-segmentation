# Quick Workflow Checklist: Local Development → Kaggle Training

## 📋 Pre-Kaggle Checklist (Local Machine)

### Step 1: Verify Local Setup Works
- [ ] Project runs without CUDA errors (uses CPU fallback)
- [ ] All imports work correctly
- [ ] Data loads successfully
- [ ] Dataset visualization works
- [ ] Metrics calculate without errors

**Test locally:**
```powershell
cd c:\rama\learn\pjt4-semantic-segmentation
python -c "import torch; from src.dataset import SemanticSegmentationDataset; print('✓ Imports OK')"
```

### Step 2: Ensure Git is Ready
```powershell
# Check status
git status

# Add all changes
git add .

# Commit
git commit -m "Pre-Kaggle: Ready for GPU training"

# Push to GitHub
git push origin main
```

**Verify on GitHub:** https://github.com/YOUR_USERNAME/pjt4-semantic-segmentation

---

## 🚀 Kaggle Setup (One-time)

### Step 1: Create Kaggle Account & Join Competition
1. Go to kaggle.com
2. Join the semantic segmentation competition
3. Create a new notebook in the competition

### Step 2: Add Your Dataset to Kaggle
Option A: Upload training data
1. Create new Kaggle dataset
2. Upload `train.csv`, `imgs/`, `masks/` folders
3. Make dataset public
4. Note dataset name: `your-dataset-name`

Option B: Use existing public dataset
1. Search Kaggle for "Aeroscapes"
2. Add to your notebook

### Step 3: Create Kaggle Notebook
1. Go to Competition → Code → Create New Notebook
2. Select "Upload notebook" or "New blank notebook"
3. Copy cells from `KAGGLE_NOTEBOOK_TEMPLATE.py`
4. Replace `YOUR_USERNAME` with your GitHub username
5. Replace dataset name if needed

### Step 4: Configure First Cell

```python
# Cell 1 - Update these:
REPO_URL = "https://github.com/YOUR_GITHUB_USERNAME/pjt4-semantic-segmentation.git"
```

---

## 💻 Development Workflow

### Local Debugging (No GPU Needed)

1. **Make code changes**
   ```powershell
   cd c:\rama\learn\pjt4-semantic-segmentation
   # Edit notebook or Python files
   ```

2. **Test locally**
   ```python
   # Notebook: Ctrl+Shift+Enter to run cell
   # Or terminal:
   python -c "from src.dataset import SemanticSegmentationDataset; print('✓ OK')"
   ```

3. **Commit to Git**
   ```powershell
   git add .
   git commit -m "Fix: [describe change]"
   git push origin main
   ```

### Kaggle Training (GPU)

1. **Pull latest code** (first cell automatically does this)
   - Kaggle notebook runs: `git clone` from GitHub
   - Gets your latest commits

2. **Run training**
   - Cell 6 trains model on GPU
   - Takes time (monitor progress)
   - Saves results to `/kaggle/working/outputs/`

3. **Download results** (optional)
   - Download `submission.csv` from notebook output
   - Download plots and metrics

4. **Submit predictions**
   - Copy `submission.csv` content
   - Paste into "Submit Predictions" tab
   - Check leaderboard score

---

## 🔄 Typical Iteration Cycle

```
Local Edit
   ↓
Test Locally (Python CPU)
   ↓
Commit & Push to GitHub
   ↓
Pull in Kaggle Notebook
   ↓
Run Training on GPU
   ↓
Download & Analyze Results
   ↓
Repeat
```

---

## 🛠️ Common Commands

### Local (PowerShell)

```powershell
# Run notebook locally
jupyter notebook

# Or test imports
python -c "from src.utils import SegmentationMetrics; print('OK')"

# Check git status
git status

# Commit changes
git add . && git commit -m "message" && git push
```

### Kaggle Notebook

```python
# Check GPU
import torch
print(torch.cuda.is_available())  # Should be True

# Monitor GPU usage
!nvidia-smi

# View training progress
!tail -20 outputs/training_log.txt
```

---

## 📊 Key Differences Summary

| Feature | Local Machine | Kaggle |
|---------|--------------|---------|
| **GPU Available** | ❌ No | ✅ Yes (V100) |
| **Python Path** | `c:\rama\...` | `/kaggle/working/project` |
| **Data Path** | `../data/...` | `/kaggle/input/aeroscapes-...` |
| **Training** | Slow (CPU) | Fast (GPU) |
| **Debugging** | Quick iterations | Takes time |
| **Best For** | Code development | Full training |

---

## ⚠️ Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'src'`
**Solution (Kaggle):**
```python
import sys
sys.path.insert(0, '/kaggle/working/project')
```

### Problem: `FileNotFoundError: Data not found`
**Solution:**
- Update `DATA_PATH` in cell
- Verify dataset is added to Kaggle notebook
- Check dataset name matches exactly

### Problem: `Out of memory` during training
**Solution:**
- Reduce batch size in config
- Reduce `max_iter` value
- Use smaller image size (512 → 256)

### Problem: GPU timeout on Kaggle
**Solution:**
- Kaggle notebooks auto-kill after 9 hours
- Check if training stayed under time limit
- Restart and resume from checkpoint

---

## 📈 What to Track

### For Debugging (Local)
- ✓ Dataset loads correctly
- ✓ Image shapes are correct
- ✓ Classes are properly labeled
- ✓ Transforms apply without errors
- ✓ Metrics calculate correctly

### For Training (Kaggle)
- 📊 Training loss curve
- 📊 Validation loss curve
- 📊 Mean IoU per epoch
- 📊 Per-class IoU scores
- 📊 Final submission IoU score

---

## 🎯 Success Checklist

- [ ] Local notebook runs without GPU (using CPU)
- [ ] Code committed to GitHub
- [ ] Kaggle notebook created
- [ ] Dataset added to Kaggle
- [ ] First Kaggle run completes
- [ ] Training metrics show improvement
- [ ] Submission CSV generated
- [ ] Predictions submitted to Kaggle
- [ ] Final score ≥ 0.60 IoU

---

## 📞 Quick Links

- **Your Project**: c:\rama\learn\pjt4-semantic-segmentation
- **GitHub**: https://github.com/YOUR_USERNAME/pjt4-semantic-segmentation
- **Kaggle Competition**: [Competition URL]
- **Kaggle Notebook**: [Your notebook URL after creation]

---

**Next Action:** Push your project to GitHub, then create the Kaggle notebook!

