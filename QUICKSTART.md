# Quick Start Guide

## Installation

### Step 1: Clone or Download
```bash
cd MeshRefine
```

### Step 2: Create Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python test_installation.py
```

If all checks pass, you're ready to go! ✓

---

## Running the Main Experiment

### Basic Run
```bash
python scripts/run_oracle_sphere.py
```

This will:
1. Generate a clean icosphere mesh
2. Add 10% radial noise
3. Optimize using curvature matching
4. Compare with Laplacian smoothing baseline
5. Generate 9 visualization images
6. Save metrics to CSV

**Total time**: ~2-5 minutes (depending on GPU)

### Output
Results are saved to `outputs/`:
- `figures/`: 9 PNG images showing results
- `logs/`: CSV files with optimization history
- `metrics.csv`: Comparison table

---

## Understanding the Results

### Key Figures

1. **00_clean_sphere.png**: Ground truth (perfect sphere)
2. **01_noisy_sphere.png**: Starting point (deformed sphere)
3. **04_loss_curves.png**: Optimization progress
4. **06_optimized_sphere.png**: Final result (should look like #1)
5. **09_comparison.png**: Side-by-side comparison of all methods

### Key Metrics (in `metrics.csv`)

- **mean_vertex_error**: Average distance to ground truth (lower is better)
- **curvature_mse**: Curvature matching error (lower is better)
- **mean_radius_error**: How much spheres shrinks/expands (lower is better)

Expected results for curvature matching:
```
Noisy:              ~0.100   (baseline)
Laplacian Smoothing: ~0.050   (2x improvement)
Curvature Matching:  ~0.030   (3-4x improvement ← best!)
```

---

## Customizing the Experiment

### Method 1: Edit the Script
Edit `scripts/run_oracle_sphere.py` around line 20-40:

```python
# Noise strength
epsilon_noise = 0.1  # Change this: 0.05 for less noise, 0.15 for more

# Optimization parameters
num_iterations = 500  # Change for speed/quality tradeoff
learning_rate = 0.01
lambda_edge = 0.1     # Edge regularization
lambda_pos = 0.01     # Position regularization
```

### Method 2: Use Configuration File
Edit `config.py` for global defaults

### Method 3: Run Advanced Example
```bash
python scripts/advanced_example.py
```

This shows custom configurations and sensitivity analysis.

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution**: 
```bash
pip install -r requirements.txt
```

### Issue: CUDA out of memory
**Solution**: 
```python
# In the script, change:
device = torch.device('cpu')  # Force CPU mode
```

### Issue: Slow execution
**Solution**:
- Reduce `num_iterations` (e.g., 100 instead of 500)
- Reduce mesh subdivision (e.g., 2 instead of 3)
- Use GPU (requires CUDA-capable NVIDIA GPU)

### Issue: Optimization doesn't converge
**Things to try**:
1. Reduce learning_rate (e.g., 0.001)
2. Increase num_iterations (e.g., 1000)
3. Adjust regularization weights
4. Check if noise level is too high (>0.15)

---

## Next Steps

After validating the basic experiment:

1. **Experiment with parameters**:
   - Try different noise levels
   - Adjust regularization weights
   - Compare different mesh resolutions

2. **Implement Stage 2** (Image-space):
   - Derive curvature from depth maps
   - Connect to camera models

3. **Implement Stage 3** (RGB):
   - Train neural network to predict curvature from RGB
   - Build end-to-end optimization pipeline

4. **Real-world data**:
   - Apply to 3D scans
   - Compare with other reconstruction methods

---

## File Structure Reference

```
MeshRefine/
├── README.md                 ← Detailed documentation
├── QUICKSTART.md             ← This file
├── requirements.txt          ← Dependencies
├── config.py                 ← Configuration
├── test_installation.py      ← Verify setup
│
├── src/                      ← Core modules
│   ├── create_mesh.py
│   ├── mesh_utils.py
│   ├── curvature.py
│   ├── losses.py
│   ├── optimise.py
│   ├── evaluate.py
│   └── visualise.py
│
├── scripts/
│   ├── run_oracle_sphere.py  ← Main experiment (run this!)
│   └── advanced_example.py   ← Custom configurations
│
├── outputs/                  ← Results (generated)
│   ├── figures/
│   ├── logs/
│   └── metrics.csv
│
└── notes/
    └── experiment_summary.md ← Technical details
```

---

## Questions?

Check `README.md` for more detailed documentation, or review the comments in the Python files for implementation details.

Happy experimenting! 🚀
