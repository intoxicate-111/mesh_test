# README: Curvature-Guided Mesh Optimisation from GT Geometry / RGB-Derived Signals

## 1. Project Goal

This project aims to build a minimal proof-of-concept experiment showing that an explicit curvature target can guide mesh optimisation.

The long-term research idea is:

> Given an RGB ground-truth observation, estimate or derive a curvature-related geometric signal, then use this signal as an explicit loss to optimise a generated or noisy mesh.

However, the first-stage experiment should **not** start from real RGB images or a learned RGB-to-curvature network. The first stage should be an **oracle controlled experiment** using a simple sphere/globe mesh, where the ground-truth curvature is known or can be computed from the clean mesh.

The purpose of this stage is to answer one basic question:

> If a reliable target curvature signal is available, can curvature matching optimise a noisy mesh back toward the correct geometry?

Only after this is validated should the project move toward image-space curvature and RGB-derived curvature estimation.

---

## 2. Core Hypothesis

A noisy or deformed mesh can be improved by explicitly matching its curvature distribution to a target curvature signal computed from a clean reference mesh.

This is different from generic smoothing.

Generic smoothing usually tries to reduce local surface variation, often by minimising Laplacian displacement or applying mean-curvature-flow-like updates. In contrast, this project aims to perform **curvature target matching**:

```text
current mesh curvature -> match target curvature -> optimise geometry
```

The target is not simply “make the surface smoother”. The target is:

```text
make the current mesh bend like the ground-truth mesh
```

For the first proof-of-concept, the ground-truth mesh can be a clean sphere or globe.

---

## 3. First-Stage Experiment: Oracle Mesh-Space Curvature Matching

### 3.1 Overview

The first experiment should use a simple clean sphere mesh.

Pipeline:

```text
1. Create clean sphere mesh M_gt
2. Add synthetic geometric noise to create M_noisy
3. Compute curvature target H_gt from M_gt
4. Compute current curvature H(M) from the optimised mesh
5. Optimise vertex positions so H(M) matches H_gt
6. Compare M_noisy and M_optimised against M_gt
```

This avoids the ambiguity of RGB-to-geometry estimation and directly tests whether curvature loss is useful.

---

## 4. Implementation Requirements

### 4.1 Recommended Environment

Use Python with PyTorch.

Recommended libraries:

```text
numpy
scipy
trimesh
pytorch3d
matplotlib
open3d optional
```

If PyTorch3D is difficult to install, the first version can be implemented using only PyTorch tensors and a fixed mesh adjacency list.

The implementation should prioritise clarity over speed.

---

## 5. Mesh Generation

### 5.1 Ground-Truth Mesh

Generate a clean sphere mesh.

Possible approaches:

- Use `trimesh.creation.icosphere`
- Use PyTorch3D ico-sphere utilities
- Load a pre-generated `.obj` sphere mesh

Recommended first setting:

```text
mesh type: icosphere
subdivision level: 3 or 4
radius: 1.0
```

The clean mesh is denoted as:

```text
M_gt = (V_gt, F)
```

where:

```text
V_gt: ground-truth vertex positions
F: mesh faces / connectivity
```

Connectivity should remain fixed during optimisation.

---

## 6. Noise Model

Create a noisy mesh by perturbing vertices.

Recommended noise:

```text
V_noisy = V_gt + epsilon * N_gt * random_noise
```

where:

```text
N_gt: vertex normals of the clean sphere
random_noise: random scalar per vertex, sampled from normal or uniform distribution
epsilon: noise strength, e.g. 0.05 to 0.15
```

For a sphere, vertex normals can be approximated by normalising the vertex positions:

```python
N_gt = normalize(V_gt)
```

This produces radial bumps and dents on the sphere surface.

Optional additional noise:

```text
small tangential noise
local bump noise
low-frequency deformation
```

But the first version should use simple radial noise only.

---

## 7. Curvature Approximation

### 7.1 Simple Laplacian Curvature Proxy

For the first version, use a simple umbrella Laplacian as a curvature-related signal.

For each vertex `i`:

```text
L_i = V_i - mean(V_j), where j belongs to neighbours(i)
```

Then define curvature magnitude proxy:

```text
H_i = ||L_i||
```

This is not a fully rigorous discrete mean curvature estimator, but it is acceptable for the first proof-of-concept.

Use careful wording:

```text
Laplacian-based curvature proxy
```

rather than claiming it is exact physical curvature.

### 7.2 Target Curvature

Compute the target curvature from the clean sphere:

```text
H_gt = curvature_proxy(V_gt, F)
```

Compute the current curvature during optimisation:

```text
H_pred = curvature_proxy(V_current, F)
```

---

## 8. Optimisation Objective

The main loss is curvature matching:

```text
L_curv = mean((H_pred - H_gt)^2)
```

However, curvature loss alone may be underconstrained. Add light regularisation.

Recommended total loss:

```text
L_total = L_curv + lambda_edge * L_edge + lambda_pos * L_pos
```

where:

```text
L_edge: edge length regularisation, keeps local mesh structure stable
L_pos: weak positional regularisation to prevent extreme drift
```

Suggested definitions:

```text
L_edge = mean((edge_length_current - edge_length_gt)^2)
L_pos = mean(||V_current - V_noisy||^2)
```

Use a small value for `lambda_pos`, because the goal is not to freeze the noisy mesh.

Example weights:

```text
lambda_edge = 0.1
lambda_pos = 0.01
```

These values should be treated as tunable.

---

## 9. Baselines

At minimum, compare the following:

### 9.1 Noisy Mesh

No optimisation. This is the starting point.

### 9.2 Laplacian Smoothing Baseline

Apply standard Laplacian smoothing:

```text
V_i <- V_i + alpha * (mean(V_j) - V_i)
```

This provides a classical smoothing baseline.

Expected issue:

```text
surface becomes smoother but may shrink
```

### 9.3 Curvature Matching Optimisation

Optimise vertex positions using the curvature target matching loss.

Main comparison:

```text
Does curvature matching recover geometry better than simple smoothing?
```

Optional later baseline:

```text
Taubin smoothing
```

---

## 10. Evaluation Metrics

Report both geometry accuracy and curvature consistency.

### 10.1 Vertex Distance to Clean Sphere

Since connectivity is fixed:

```text
mean_vertex_error = mean(||V_current - V_gt||)
```

Also report:

```text
max_vertex_error
RMSE_vertex_error
```

### 10.2 Curvature Error

```text
curvature_mse = mean((H_current - H_gt)^2)
```

### 10.3 Radius Error

For a sphere of radius 1:

```text
radius_i = ||V_i||
radius_error = mean(abs(radius_i - 1.0))
```

This is especially useful for detecting shrinkage.

### 10.4 Normal Consistency

Compute vertex normals and compare with GT normals:

```text
normal_error = mean(1 - dot(N_current, N_gt))
```

For a sphere, GT normal can be approximated as:

```text
N_gt = normalize(V_gt)
```

---

## 11. Expected Outputs

The agent should generate the following outputs.

### 11.1 Visualisations

Save images showing:

```text
1. clean sphere / globe
2. noisy sphere / globe
3. Laplacian-smoothed sphere
4. curvature-optimised sphere
```

Optional visualisations:

```text
curvature heatmap on clean mesh
curvature heatmap on noisy mesh
curvature heatmap after optimisation
loss curve during optimisation
```

### 11.2 Metrics Table

Save a table like:

```text
method                vertex_error    curvature_mse    radius_error    normal_error
noisy                 ...             ...              ...             ...
laplacian_smoothing   ...             ...              ...             ...
curvature_matching    ...             ...              ...             ...
```

### 11.3 Log File

Save optimisation logs:

```text
iteration
L_total
L_curv
L_edge
L_pos
vertex_error
curvature_mse
```

---

## 12. Suggested File Structure

```text
curvature_mesh_optimisation/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── create_mesh.py
│   ├── mesh_utils.py
│   ├── curvature.py
│   ├── losses.py
│   ├── optimise.py
│   ├── evaluate.py
│   └── visualise.py
│
├── scripts/
│   ├── run_oracle_sphere.py
│   └── run_laplacian_baseline.py
│
├── outputs/
│   ├── meshes/
│   ├── figures/
│   ├── logs/
│   └── metrics.csv
│
└── notes/
    └── experiment_summary.md
```

A single-file prototype is also acceptable for the first implementation, but the logic should still be separated into clear functions.

---

## 13. Minimal Single-Script Prototype

The first working version can be implemented as:

```text
scripts/run_oracle_sphere.py
```

This script should:

```text
1. create sphere
2. add noise
3. compute adjacency
4. compute curvature proxy
5. optimise vertices with PyTorch
6. evaluate metrics
7. save plots and metrics
```

This is the recommended first deliverable.

---

## 14. Important Design Decisions

### 14.1 Keep Connectivity Fixed

Do not change mesh topology in the first experiment.

Only optimise vertex positions:

```text
F remains fixed
V is optimised
```

This makes the experiment easier to debug and evaluate.

### 14.2 Do Not Start with RGB

Do not begin with RGB-to-curvature estimation.

Reason:

```text
RGB alone does not uniquely determine curvature.
```

Lighting, material, texture, shadow, and albedo can all change RGB appearance without changing geometry.

The project should therefore proceed in stages:

```text
Stage 1: mesh-space oracle curvature target
Stage 2: image-space curvature from GT depth / normal
Stage 3: curvature estimated from RGB via depth / normal prediction
```

### 14.3 Avoid Overclaiming

For the first experiment, describe the signal as:

```text
Laplacian-based curvature proxy
```

or:

```text
curvature-related geometric signal
```

Do not claim that the first version computes exact continuous curvature.

---

## 15. Acceptance Criteria

The experiment is considered successful if:

```text
1. The curvature-optimised mesh has lower curvature error than the noisy mesh.
2. The curvature-optimised mesh has lower vertex/radius/normal error than the noisy mesh.
3. The result can be visually shown as a noisy sphere becoming closer to the clean sphere.
4. The method is compared against simple Laplacian smoothing.
5. The output includes figures, metrics, and a short experiment summary.
```

The strongest result would be:

```text
Curvature matching improves curvature consistency while producing less shrinkage than naive Laplacian smoothing.
```

---

## 16. Possible Failure Cases

### 16.1 Curvature Loss Alone Does Not Recover Position

Curvature alone may not fully determine the surface position. Different surfaces can have similar local curvature distributions.

Solution:

```text
add weak positional, edge-length, or normal regularisation
```

### 16.2 Mesh Shrinkage

Laplacian smoothing may shrink the sphere.

This is expected and should be reported as a baseline limitation.

### 16.3 Noisy Curvature Estimates

The simple Laplacian curvature proxy may be noisy.

Solution:

```text
start with low mesh resolution
use mild noise
compare curvature distributions instead of only per-vertex values
```

### 16.4 Vertex Correspondence Assumption

The first experiment assumes that `M_gt` and `M_noisy` share the same topology and vertex ordering.

This is acceptable for the controlled oracle experiment.

Later stages should relax this assumption.

---

## 17. Next Stage After Success

After the mesh-space oracle experiment works, move to image-space curvature.

Stage 2 pipeline:

```text
1. Render clean sphere/globe to obtain GT depth and normal maps
2. Compute image-space curvature target from depth or normal maps
3. Render current mesh depth/normal maps
4. Compute predicted image-space curvature
5. Optimise mesh using image-space curvature loss
```

Possible image-space curvature proxies:

```text
normal gradient magnitude
Laplacian of depth
mean-curvature approximation from depth map
```

Then Stage 3:

```text
RGB image -> predicted depth/normal -> curvature target -> mesh optimisation
```

This separates the research problem into two parts:

```text
1. Is curvature supervision useful?
2. Can curvature supervision be reliably derived from RGB?
```

---

## 18. Suggested Short Experiment Summary Template

After running the experiment, write a short summary using this structure:

```text
Goal:
We tested whether an explicit curvature target can guide optimisation of a noisy mesh.

Setup:
A clean sphere was used as the ground-truth mesh. Radial noise was added to create a perturbed mesh. A Laplacian-based curvature proxy was computed on the clean mesh and used as the target signal.

Methods Compared:
1. Noisy mesh
2. Laplacian smoothing
3. Curvature matching optimisation

Metrics:
We measured vertex error, curvature MSE, radius error, and normal consistency against the clean sphere.

Result:
Curvature matching reduced curvature error and improved geometric consistency compared with the noisy mesh. Compared with naive Laplacian smoothing, it provided a more explicit target-driven optimisation objective.

Conclusion:
This supports the idea that curvature can be used as an explicit geometric supervision signal, motivating later experiments where the curvature target is derived from rendered depth/normal maps and eventually from RGB observations.
```

---

## 19. Immediate Task for the Agent

Implement the first-stage oracle sphere experiment.

Deliver:

```text
1. runnable Python script
2. generated figures
3. metrics.csv
4. short experiment summary
```

Do not implement RGB-to-curvature yet.

Focus on proving that curvature target matching can optimise a noisy mesh in a controlled setting.

