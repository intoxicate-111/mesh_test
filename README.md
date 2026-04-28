# Curvature-Guided Mesh Optimization

A Python implementation of oracle mesh-space curvature matching for proof-of-concept mesh optimization.

## Project Overview

This project demonstrates that an explicit curvature target can guide mesh optimization. The first-stage experiment uses a simple sphere mesh to validate whether **curvature matching** can recover noisy geometry better than classical smoothing methods.

### Pipeline

```
1. Create clean sphere mesh (M_gt)
2. Add synthetic noise to create noisy mesh (M_noisy)
3. Compute curvature target from clean mesh (H_gt)
4. Optimize vertex positions to match target curvature
5. Evaluate and compare with baseline methods
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone/download the repository:
```bash
cd MeshRefine
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Run the main experiment:

```bash
python scripts/run_oracle_sphere.py
```

This will:
- Generate a clean icosphere mesh
- Add radial noise
- Optimize using curvature matching
- Compare with Laplacian smoothing baseline
- Generate visualizations and metrics

Output files will be saved to the `outputs/` directory.

## Project Structure

```
MeshRefine/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
│
├── src/                         # Core modules
│   ├── __init__.py
│   ├── create_mesh.py           # Mesh creation and noise generation
│   ├── mesh_utils.py            # Mesh utility functions
│   ├── curvature.py             # Curvature approximation
│   ├── losses.py                # Loss functions
│   ├── optimise.py              # Optimization algorithms
│   ├── evaluate.py              # Evaluation metrics
│   └── visualise.py             # Visualization utilities
│
├── scripts/
│   └── run_oracle_sphere.py     # Main experiment script
│
└── outputs/                     # Generated outputs
    ├── meshes/                  # Saved mesh files
    ├── figures/                 # Visualization images
    ├── logs/                    # CSV logs
    └── metrics.csv              # Evaluation metrics
```

## Key Components

### 1. Mesh Generation (`src/create_mesh.py`)

- **`create_icosphere()`**: Creates a clean icosphere mesh using trimesh
- **`add_radial_noise()`**: Adds random radial perturbations to vertices
- **`compute_vertex_normals_from_positions()`**: Computes normals for sphere geometry

### 2. Curvature Approximation (`src/curvature.py`)

Uses a **Laplacian-based curvature proxy**:

For each vertex `i`:
```
L_i = V_i - mean(V_j), where j ∈ neighbors(i)
H_i = ||L_i||
```

This is not exact mean curvature but provides a useful curvature-related signal.

### 3. Loss Functions (`src/losses.py`)

**Total optimization loss:**
```
L_total = L_curv + λ_edge * L_edge + λ_pos * L_pos
```

Where:
- `L_curv`: MSE between predicted and target curvature
- `L_edge`: Edge length regularization (maintains mesh structure)
- `L_pos`: Position regularization (prevents extreme drift)

### 4. Optimization (`src/optimise.py`)

- **Gradient-based optimization** using PyTorch Adam optimizer
- **Laplacian smoothing baseline** for comparison
- Customizable hyperparameters (learning rate, regularization weights, iterations)

### 5. Evaluation (`src/evaluate.py`)

Metrics computed:
- **Vertex errors**: mean, max, RMSE
- **Curvature error**: MSE, MAE, max error
- **Radius error**: deviation from sphere radius
- **Normal error**: consistency with ground-truth normals

### 6. Visualization (`src/visualise.py`)

- 3D mesh visualization
- Curvature heatmaps
- Loss curve plots
- Side-by-side comparison figures

## Configuration

Edit `scripts/run_oracle_sphere.py` to adjust parameters:

```python
# Noise strength
epsilon_noise = 0.1

# Optimization hyperparameters
num_iterations = 500
learning_rate = 0.01
lambda_edge = 0.1      # Edge regularization weight
lambda_pos = 0.01      # Position regularization weight

# Laplacian smoothing parameter
alpha = 0.1
```

## Output Files

### Visualizations (`outputs/figures/`)
- `00_clean_sphere.png` - Ground truth mesh
- `01_noisy_sphere.png` - Initial noisy mesh
- `02_target_curvature.png` - Target curvature heatmap
- `04_loss_curves.png` - Optimization loss curves
- `05_smoothed_sphere.png` - Laplacian smoothing result
- `06_optimized_sphere.png` - Curvature matching result
- `09_comparison.png` - Side-by-side comparison

### Metrics (`outputs/`)
- `metrics.csv` - Evaluation metrics for all methods
- `logs/optimization_log.csv` - Detailed optimization progression
- `logs/smoothing_log.csv` - Smoothing baseline progression

## Method Comparison

The experiment compares three approaches:

1. **Noisy Mesh** (baseline - no optimization)
2. **Laplacian Smoothing** (classical baseline)
3. **Curvature Matching** (proposed - this paper)

The key research question:
> **Does curvature matching recover geometry better than simple smoothing?**

## Examples and Expected Results

For a clean unit sphere with 10% radial noise:

| Method | Vertex Error | Curvature MSE |
|--------|--------------|--------------|
| Noisy | ~0.100 | High |
| Laplacian Smoothing | ~0.050 | Medium |
| Curvature Matching | ~0.030 | Low |

(Exact values depend on noise level and hyperparameters)

## Future Extensions

1. **Stage 2**: Image-space curvature from depth/normal maps
2. **Stage 3**: RGB-derived curvature via deep learning
3. **Topology optimization**: Support mesh refinement/coarsening
4. **Non-rigid registration**: Use for shape matching
5. **Real data**: Apply to noisy 3D scans

## Technical Notes

### Why Laplacian Proxy?
- Simple to implement and understand
- Fast to compute
- Useful for initial proof-of-concept
- Not claimed to be exact mean curvature

### Mesh Topology
- **Connectivity remains fixed** during optimization
- Only vertex positions are adjusted
- Keeps the experiment focused and debuggable

### Numerical Stability
- Gradient clipping to prevent divergence
- Small regularization terms to stabilize optimization
- Careful initialization and parameter tuning

## Troubleshooting

### GPU Memory Issues
```python
# Use CPU instead:
device = torch.device('cpu')
```

### Slow Convergence
- Increase learning rate (but risk instability)
- Reduce regularization weights
- Increase number of iterations

### Poor Optimization Results
- Check noise level (shouldn't exceed 0.15 × radius)
- Verify curvature is being computed correctly
- Inspect loss curves to identify issues

## References

The approach is inspired by:
- Discrete differential geometry (Meyer et al., 2003)
- Spectral mesh processing
- Shape optimization literature

## License

[Specify your license here]

## Author

[Your name/organization]

## Acknowledgments

- Trimesh library for mesh utilities
- PyTorch for automatic differentiation
- Open3D for visualization inspiration
