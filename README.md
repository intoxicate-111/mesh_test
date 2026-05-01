# Mesh Optimization via Multi-Objective Matching

A comprehensive Python implementation of mesh optimization using gradient-based methods. Supports curvature, depth, and normal matching on both synthetic (sphere) and real (Stanford Bunny) meshes.

## Project Overview

This project implements and compares multiple mesh optimization objectives:
- **Curvature Matching**: Match target mean curvature via Laplacian proxy
- **Depth Matching**: Match vertex depth (z-axis projection)
- **Normal Matching**: Match vertex normals
- **Laplacian Smoothing**: Classical baseline for comparison

### Supported Features
- ✅ Multi-objective optimization (curvature/depth/normal)
- ✅ Vector and scalar curvature modes
- ✅ Realistic noise injection (radial/tangential/mixed modes)
- ✅ Dynamic learning rate and regularization scheduling
- ✅ Comprehensive mesh evaluation metrics
- ✅ Support for arbitrary mesh topologies (tested on sphere and Stanford Bunny)

### Pipeline

```
1. Load mesh (synthetic icosphere or Stanford Bunny)
2. Add mixed noise (50% radial + 50% tangential components)
3. Compute target measurements from clean mesh
4. Optimize using 4 different objectives (5000 iterations each):
   - Curvature matching (scalar/vector)
   - Depth matching
   - Normal matching
   - Laplacian smoothing baseline
5. Evaluate and compare all methods across 10+ metrics
```

## Installation

### Prerequisites
- Python 3.8+
- pip (or conda)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/intoxicate-111/mesh_test.git
cd mesh_test
```

2. Create a virtual environment (recommended):
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

### Run Sphere Experiment
```bash
python scripts/run_oracle_sphere.py
```
- Clean icosphere mesh with 642 vertices
- Noise level: epsilon=0.1
- Output: `outputs/` directory

### Run Stanford Bunny Experiment
```bash
python scripts/run_stanford_bunny.py
```
- Automatically downloads Stanford Bunny (2503 vertices)
- Automatic mesh normalization to unit scale
- Mixed noise: epsilon=0.05 (normalized scale)
- Output: `bunny_outputs/` directory

## Project Structure

```
mesh_test/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore configuration
│
├── src/                               # Core optimization modules
│   ├── __init__.py
│   ├── create_mesh.py                 # Mesh generation & noise modes
│   ├── mesh_utils.py                  # Mesh utility functions
│   ├── curvature.py                   # Curvature computation
│   ├── losses.py                      # Multi-objective loss functions
│   ├── optimise.py                    # Optimization algorithms
│   ├── evaluate.py                    # Evaluation metrics
│   └── visualise.py                   # Visualizations
│
├── scripts/                           # Experiment scripts
│   ├── run_oracle_sphere.py           # Sphere optimization experiment
│   ├── run_stanford_bunny.py          # Bunny optimization experiment
│   └── advanced_example.py            # Custom optimization template
│
├── outputs/                           # Sphere experiment results
│   ├── meshes/                        # Optimized mesh files (.obj)
│   ├── figures/                       # Visualizations (.png)
│   └── logs/                          # Optimization logs (.csv)
│
└── bunny_outputs/                     # Bunny experiment results
    ├── meshes/                        # Bunny mesh variants
    ├── figures/                       # Comparison visualizations
    └── logs/                          # Optimization metrics
```

## Key Components

### 1. Mesh Generation (`src/create_mesh.py`)

**Icosphere Creation:**
```python
vertices, faces, mesh = create_icosphere(subdivisions=3, radius=1.0)
```

**Noise Injection with Multiple Modes:**
```python
# Three noise modes available:
# - 'radial': pure normal-direction perturbation
# - 'tangential': surface-sliding disturbance
# - 'mixed': 50% radial + 50% tangential (DEFAULT)
vertices_noisy = add_radial_noise(
    vertices_gt, 
    normals, 
    epsilon=0.1, 
    seed=42,
    noise_mode='mixed'
)
```

**Stanford Bunny Loading (Auto-normalized):**
```python
# Automatically downloads from Stanford, centers to origin, scales to unit sphere
vertices, faces, mesh = load_stanford_bunny(device='cpu')
```

### 2. Multi-Objective Loss Functions (`src/losses.py`)

**Unified Loss Computation:**
```python
total_loss, loss_dict = objective_total_loss(
    vertices_current,
    vertices_init,
    faces,
    objective_target,
    edges,
    target_edge_lengths,
    objective_mode='curvature',  # or 'depth', 'normal'
    curvature_mode='vector',     # or 'scalar'
    lambda_edge=0.01,            # Edge regularization
    lambda_pos=0.001,            # Position regularization
    adjacency=adjacency,
)
```

**Available Objectives:**
- `curvature`: L = mean((H_pred - H_gt)²) with scalar or vector curvature
- `depth`: L = mean((z_pred - z_gt)²)
- `normal`: L = mean(||n_pred - n_gt||²)

### 3. Generic Optimization Loop (`src/optimise.py`)

**Dynamic Scheduling:**
- Learning rate decay: starts at iteration `lr_decay_start * num_iterations`
- Regularization decay: λ_edge and λ_pos linearly → 0 in final 30%
- Configurable decay curves

```python
vertices_opt, logs = optimize_mesh_objective(
    vertices_noisy,
    vertices_gt,
    faces,
    objective_target,
    objective_mode='curvature',
    num_iterations=5000,
    learning_rate=0.001,
    lambda_edge=0.01,
    lambda_pos=0.001,
    enable_dynamic_schedule=True,
    lambda_decay_start=0.7,
    lr_decay_start=0.7,
    min_lr_scale=0.1,
    device='cpu',
)
```

### 4. Comprehensive Evaluation (`src/evaluate.py`)

**Metrics Computed:**
- Vertex errors: mean, RMSE, max, std
- Curvature errors: MSE, MAE, max error
- Depth errors: MSE
- Normal errors: MSE
- Radius errors: mean deviation from target radius
- Normal consistency: smoothness of normal field

### 5. Visualization (`src/visualise.py`)

- 3D mesh rendering
- Curvature heatmaps (hot = high curvature)
- Loss curve analysis (total, objective, regularization)
- Multi-method comparison figures

## Configuration Examples

### Experiment Parameters (in scripts)

**Sphere Experiment:**
```python
epsilon_noise = 0.1              # 10% of radius
num_iterations = 5000
learning_rate = 0.001
lambda_edge = 0.01
lambda_pos = 0.001
curvature_mode = 'vector'        # Better than 'scalar' for curv matching
```

**Bunny Experiment:**
```python
epsilon_noise = 0.05             # 5% of normalized scale
noise_mode = 'mixed'             # Default: 50/50 radial + tangential
# Auto-normalized mesh → [-1, 1] bounding box
```

## Output Files

### Sphere Experiment (`outputs/`)
- **meshes/**: clean_sphere.obj, noisy_sphere.obj, curvature_optimized_sphere.obj, depth_optimized_sphere.obj, normal_optimized_sphere.obj, smoothed_sphere.obj
- **figures/**: input visualizations, loss curves, heatmaps, comparison
- **logs/**: CSV files with iteration-by-iteration metrics

### Bunny Experiment (`bunny_outputs/`)
- **meshes/**: clean_bunny.obj, noisy_bunny.obj, + 4 optimized variants
- **figures/**: curvature heatmaps, loss curves, comprehensive comparison
- **logs/**: curvature_optimization_log.csv, depth_optimization_log.csv, normal_optimization_log.csv, laplacian_smoothing_log.csv, metrics_summary.csv

## Method Comparison

The experiments compare 5 methods on each objective:

| Method | Description | Iterations | Best For |
|--------|-------------|-----------|----------|
| Noisy Baseline | No optimization | - | Reference |
| Curvature Matching | MSE(H_pred, H_gt) | 5000 | Surface curvature recovery |
| Depth Matching | MSE(depth_pred, depth_gt) | 5000 | Silhouette/depth preservation |
| Normal Matching | MSE(n_pred, n_gt) | 5000 | Surface orientation |
| Laplacian Smoothing | Classical diffusion-based smoothing | 5000 | General-purpose denoising |

### Expected Results for Unit Sphere + 10% Noise

| Metric | Noisy | Curvature | Depth | Normal | Laplacian |
|--------|-------|-----------|-------|--------|-----------|
| Mean Vertex Error | 0.100 | 0.025 | 0.045 | 0.035 | 0.055 |
| Curvature MSE | High | **Low** | Medium | Low | Medium |
| Normal Consistency | Low | Medium | Medium | **High** | Medium |

*(Actual values vary based on noise, hyperparameters, and mesh complexity)*

## Advanced Usage

### Custom Mesh Optimization

Modify `scripts/advanced_example.py` to test custom meshes, objectives, or hyperparameters:

```python
vertices, faces = load_your_mesh()
target_measurement = compute_target(vertices_gt)

vertices_opt, logs = optimize_mesh_objective(
    vertices_noisy,
    vertices_gt,
    faces,
    target_measurement,
    objective_mode='custom',  # Add custom objective
    num_iterations=10000,
    learning_rate=0.005,
)
```

### GPU Acceleration

Enable CUDA if available:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Technical Notes

### Discrete Curvature Approximation
Uses Laplacian-based proxy, NOT exact mean curvature:
```
L_i = V_i - mean(V_neighbors)  ≈ 2Hn where H = mean curvature, n = normal
```
Simple, fast, and effective for optimization signal.

### Regularization Terms
- **Edge Length**: Prevents mesh distortion/collapse
- **Position**: Prevents extreme drift from initial mesh

### Dynamic Scheduling Benefits
- Early iterations: strong regularization, larger learning rate
- Late iterations: focus on fine-tuning with smaller gradients

### Mixed Noise Mode
Realistic corruption model:
- 50% normal-direction (radial) component
- 50% tangential-direction component
- Preserves mesh connectivity while deforming surface

## Troubleshooting

### Import Errors
```bash
# Ensure path is correct
python scripts/run_oracle_sphere.py
# NOT from subdirectory
```

### CUDA Out of Memory
```python
# Use CPU mode
device = torch.device('cpu')
```

### Poor Convergence
- Reduce `lambda_pos` (less position constraint)
- Increase `learning_rate` (faster updates)
- Verify `epsilon_noise` isn't too large (>0.2)

### Visualization Issues
- Set matplotlib backend: `matplotlib.use('Agg')` for headless mode
- Check PNG files in `outputs/figures/` directory

## References

- Meyer et al. (2003): "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds"
- Discrete differential geometry and mesh processing literature
- PyTorch autodiff for gradient-based optimization

## License

MIT License

## Contributing

Contributions welcome! Areas for enhancement:
- [ ] Support for other mesh formats (PLY, STL, GLTF)
- [ ] GPU-accelerated adjacency list computation
- [ ] Anisotropic regularization
- [ ] Non-rigid registration utilities
- [ ] Deep learning-based mesh generation

## Authors

Developed as a research implementation for learning-based mesh optimization.

## Recent Changes

- Vectorized curvature computation: replaced per-vertex Python loops with a batched, edge-based aggregation using `scatter_add_`. This reduces Python-level autograd graph overhead and significantly speeds up optimization on large meshes. (See `src/curvature.py`.)
- Default dynamic schedule enabled: the CLI now enables adaptive scheduling by default for `run_custom_mesh.py` (use `--disable-schedule` to turn off). Decay start defaults were changed to be more aggressive where appropriate.
- Plateau-based adaptive decay: learning rate and regularizers now decay when the objective plateaus (default patience 50 iterations, configurable via CLI). This is implemented in `src/optimise.py` and replaces the previous epoch-ratio forced decay.
- Lambda zero-threshold: added an experimental aggressive option to zero `lambda_edge` and/or `lambda_pos` when they fall below a threshold after decay (`--lambda-zero-threshold`), intended for ablation testing.
- New CLI flags: `--enable-schedule`/`--disable-schedule`, `--plateau-patience`, `--plateau-min-delta`, `--decay-factor`, and `--lambda-zero-threshold` (see `scripts/run_custom_mesh.py`).

If you run into issues after these changes, please open an issue with the experiment command and log files from `outputs/` or `bunny_outputs/`.
