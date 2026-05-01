"""
Stanford Bunny Mesh-Space Optimization Experiment

This script demonstrates mesh optimization on the Stanford Bunny mesh,
comparing curvature, depth, and normal matching methods.
"""

import sys
import torch
import numpy as np
import os
import csv
import trimesh
from pathlib import Path

# Set matplotlib to non-interactive backend BEFORE importing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Fix path for imports when running from scripts directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Local imports
from src.create_mesh import add_radial_noise, compute_vertex_normals_from_positions
from src.curvature import compute_laplacian_curvature_proxy, compute_laplacian_curvature_vector
from src.optimise import optimize_mesh_objective, optimize_mesh_curvature, laplacian_smoothing_baseline
from src.evaluate import evaluate_mesh, print_metrics_table
from src.visualise import (
    visualize_mesh_3d,
    visualize_curvature_heatmap,
    plot_loss_curve,
    create_comparison_figure
)


def load_stanford_bunny(device='cpu'):
    """Load Stanford Bunny mesh from online repository.
    
    Returns:
        vertices: (N, 3) tensor
        faces: (M, 3) tensor
        mesh: trimesh object
    """
    print("Loading Stanford Bunny from online repository...")
    bunny_url = "https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj"
    
    try:
        # Try loading from URL using trimesh
        mesh = trimesh.load(bunny_url, process=False, allow_remote=True)
    except Exception as e:
        print(f"Note: Could not load from URL: {e}")
        print("Attempting to use local cached copy or creating alternative...")
        # Fallback: try common local paths or use a simpler mesh
        local_paths = [
            Path("data/bunny.obj"),
            Path("assets/bunny.obj"),
            Path(__file__).parent.parent / "data" / "bunny.obj"
        ]
        mesh = None
        for path in local_paths:
            if path.exists():
                print(f"Found local bunny at {path}")
                mesh = trimesh.load(str(path), process=False)
                break
        
        if mesh is None:
            raise FileNotFoundError(
                "Could not load Stanford Bunny. Please ensure internet connection "
                "or provide bunny.obj in data/ folder."
            )
    
    # Extract vertices and faces
    vertices = torch.from_numpy(mesh.vertices.astype(np.float32)).to(device)
    faces = torch.from_numpy(mesh.faces.astype(np.int64)).to(device)
    
    print(f"Loaded bunny: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    
    # ========== NORMALIZATION ==========
    print("\nNormalizing mesh...")
    
    # 1. Compute bounding box
    v_min = torch.min(vertices, dim=0)[0]
    v_max = torch.max(vertices, dim=0)[0]
    centroid = (v_min + v_max) / 2.0
    bb_size = torch.max(v_max - v_min)
    
    print(f"  Original bounding box: min={v_min.cpu().numpy()}, max={v_max.cpu().numpy()}")
    print(f"  Original centroid: {centroid.cpu().numpy()}")
    print(f"  Original size: {bb_size.item():.6f}")
    
    # 2. Center the mesh
    vertices = vertices - centroid
    print(f"  After translation: centroid={torch.mean(vertices, dim=0).cpu().numpy()}")
    
    # 3. Scale to unit sphere
    vertices = vertices / (bb_size / 2.0)
    print(f"  After scaling: bounding box size={torch.max(vertices).item():.6f}")
    
    # 4. Verify normalized mesh
    v_min_norm = torch.min(vertices, dim=0)[0]
    v_max_norm = torch.max(vertices, dim=0)[0]
    print(f"  Normalized range: [{v_min_norm.cpu().numpy()}, {v_max_norm.cpu().numpy()}]")
    
    return vertices, faces, mesh


def save_mesh(vertices, faces, save_path):
    """Save a mesh to disk using trimesh (supports .obj/.ply/.stl)."""
    if torch.is_tensor(vertices):
        vertices_np = vertices.detach().cpu().numpy()
    else:
        vertices_np = vertices

    if torch.is_tensor(faces):
        faces_np = faces.detach().cpu().numpy()
    else:
        faces_np = faces

    mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)
    mesh.export(str(save_path))
    print(f"Saved mesh to {save_path}")


def main():
    """Main experiment function for Stanford Bunny."""
    
    # ========== SETUP ==========
    print("=" * 80)
    print("Stanford Bunny Mesh-Space Optimization Experiment")
    print("=" * 80)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create output directories for bunny experiment
    output_dir = Path("bunny_outputs")
    meshes_dir = output_dir / "meshes"
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"
    
    for d in [meshes_dir, figures_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir.absolute()}\n")

    # Curvature loss mode for optimisation: 'scalar' or 'vector'
    curvature_mode = 'vector'
    view_direction = torch.tensor([0.0, 0.0, 1.0], device=device)
    lambda_zero_threshold = 1e-4
    
    # ========== MESH LOADING ==========
    print("-" * 80)
    print("STEP 1: Loading Stanford Bunny Mesh")
    print("-" * 80)
    
    # Load bunny mesh
    print("\n1.1 Loading clean bunny mesh...")
    vertices_gt, faces, mesh_obj = load_stanford_bunny(device=device)
    print(f"    Loaded mesh with {vertices_gt.shape[0]} vertices and {faces.shape[0]} faces")
    
    # Compute vertex normals for the clean bunny
    print("\n1.2 Computing vertex normals...")
    normals_gt = compute_vertex_normals_from_positions(vertices_gt, faces)
    print(f"    Computed normals for {normals_gt.shape[0]} vertices")
    
    # Add noise to create noisy mesh
    print("\n1.3 Adding mixed noise to create noisy mesh...")
    epsilon_noise = 0.05  # Noise strength relative to normalized mesh (scale ~2.0)
    vertices_noisy = add_radial_noise(vertices_gt, normals_gt, epsilon=epsilon_noise, 
                                      seed=42, noise_mode='mixed')
    print(f"    Noise strength (epsilon): {epsilon_noise}")
    print(f"    Noise mode: mixed (50% radial + 50% tangential)")
    
    # Compute target measurements from clean mesh
    print("\n1.4 Computing target measurements from clean mesh...")
    curvature_gt = compute_laplacian_curvature_proxy(vertices_gt, faces)
    curvature_vec_gt = compute_laplacian_curvature_vector(vertices_gt, faces)
    depth_gt = vertices_gt[:, 2]
    print(f"    Target curvature - Min: {curvature_gt.min():.6f}, Max: {curvature_gt.max():.6f}, "
          f"Mean: {curvature_gt.mean():.6f}")
    
    # Compute measurements of noisy mesh for reference
    curvature_noisy = compute_laplacian_curvature_proxy(vertices_noisy, faces)
    print(f"    Noisy curvature   - Min: {curvature_noisy.min():.6f}, Max: {curvature_noisy.max():.6f}, "
          f"Mean: {curvature_noisy.mean():.6f}")
    
    # ========== VISUALIZATION: INPUT MESHES ==========
    print("\n1.5 Saving input mesh visualizations...")
    visualize_mesh_3d(vertices_gt, faces, title="Clean Stanford Bunny",
                     save_path=figures_dir / "00_clean_bunny.png")
    visualize_mesh_3d(vertices_noisy, faces, title="Noisy Stanford Bunny",
                     save_path=figures_dir / "01_noisy_bunny.png")
    visualize_curvature_heatmap(vertices_gt, faces, curvature_gt,
                               title="Target Curvature (Clean Bunny)",
                               save_path=figures_dir / "02_target_curvature.png")
    visualize_curvature_heatmap(vertices_noisy, faces, curvature_noisy,
                               title="Initial Curvature (Noisy Bunny)",
                               save_path=figures_dir / "03_noisy_curvature.png")

    print("\n1.6 Saving input meshes...")
    save_mesh(vertices_gt, faces, meshes_dir / "clean_bunny.obj")
    save_mesh(vertices_noisy, faces, meshes_dir / "noisy_bunny.obj")
    
    # ========== OPTIMIZATION: CURVATURE MATCHING ==========
    print("\n" + "-" * 80)
    print("STEP 2: Curvature Matching Optimization")
    print("-" * 80)
    
    print("\n2.1 Optimizing mesh to match target curvature...")
    if curvature_mode == 'scalar':
        curvature_target = curvature_gt
    elif curvature_mode == 'vector':
        curvature_target = curvature_vec_gt
    else:
        raise ValueError(f"Unsupported curvature_mode: {curvature_mode}")

    vertices_optimized, logs_optimization = optimize_mesh_objective(
        vertices_noisy,
        vertices_gt,
        faces,
        curvature_target,
        objective_mode='curvature',
        curvature_mode=curvature_mode,
        num_iterations=5000,
        learning_rate=0.001,
        lambda_edge=0.01,
        lambda_pos=0.001,
        enable_dynamic_schedule=True,
        plateau_patience=50,
        plateau_min_delta=1e-5,
        decay_factor=0.5,
        lambda_zero_threshold=lambda_zero_threshold,
        min_lr_scale=0.1,
        device=device,
        verbose=True
    )
    
    print("\n2.2 Saving optimization log and loss curves...")
    log_path = logs_dir / "curvature_optimization_log.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=logs_optimization[0].keys())
        writer.writeheader()
        writer.writerows(logs_optimization)
    print(f"    Saved to {log_path}")
    
    plot_loss_curve(logs_optimization, save_path=figures_dir / "04_curvature_loss_curves.png")
    save_mesh(vertices_optimized, faces, meshes_dir / "curvature_optimized_bunny.obj")
    print("    Saved optimized mesh")

    # ========== OPTIMIZATION: DEPTH MATCHING ==========
    print("\n" + "-" * 80)
    print("STEP 3: Depth Matching Optimization")
    print("-" * 80)

    print("\n3.1 Optimizing mesh to match target depth...")
    vertices_depth, logs_depth = optimize_mesh_objective(
        vertices_noisy,
        vertices_gt,
        faces,
        depth_gt,
        objective_mode='depth',
        view_direction=view_direction,
        num_iterations=5000,
        learning_rate=0.001,
        lambda_edge=0.01,
        lambda_pos=0.001,
        enable_dynamic_schedule=True,
        plateau_patience=50,
        plateau_min_delta=1e-5,
        decay_factor=0.5,
        lambda_zero_threshold=lambda_zero_threshold,
        min_lr_scale=0.1,
        device=device,
        verbose=True
    )

    print("\n3.2 Saving depth log and loss curves...")
    log_path = logs_dir / "depth_optimization_log.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=logs_depth[0].keys())
        writer.writeheader()
        writer.writerows(logs_depth)
    print(f"    Saved to {log_path}")
    plot_loss_curve(logs_depth, save_path=figures_dir / "05_depth_loss_curves.png")
    save_mesh(vertices_depth, faces, meshes_dir / "depth_optimized_bunny.obj")
    print("    Saved optimized mesh")

    # ========== OPTIMIZATION: NORMAL MATCHING ==========
    print("\n" + "-" * 80)
    print("STEP 4: Normal Matching Optimization")
    print("-" * 80)

    print("\n4.1 Optimizing mesh to match target normals...")
    vertices_normal, logs_normal = optimize_mesh_objective(
        vertices_noisy,
        vertices_gt,
        faces,
        normals_gt,
        objective_mode='normal',
        num_iterations=5000,
        learning_rate=0.001,
        lambda_edge=0.01,
        lambda_pos=0.001,
        enable_dynamic_schedule=True,
        plateau_patience=50,
        plateau_min_delta=1e-5,
        decay_factor=0.5,
        lambda_zero_threshold=lambda_zero_threshold,
        min_lr_scale=0.1,
        device=device,
        verbose=True
    )

    print("\n4.2 Saving normal log and loss curves...")
    log_path = logs_dir / "normal_optimization_log.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=logs_normal[0].keys())
        writer.writeheader()
        writer.writerows(logs_normal)
    print(f"    Saved to {log_path}")
    plot_loss_curve(logs_normal, save_path=figures_dir / "06_normal_loss_curves.png")
    save_mesh(vertices_normal, faces, meshes_dir / "normal_optimized_bunny.obj")
    print("    Saved optimized mesh")
    
    # ========== BASELINE: LAPLACIAN SMOOTHING ==========
    print("\n" + "-" * 80)
    print("STEP 5: Laplacian Smoothing Baseline")
    print("-" * 80)
    
    print("\n5.1 Applying Laplacian smoothing...")
    vertices_smoothed, logs_smoothing = laplacian_smoothing_baseline(
        vertices_noisy,
        vertices_gt,
        faces,
        num_iterations=5000,
        alpha=0.1,
        device=device,
        verbose=True
    )
    
    print("\n5.2 Saving smoothing log...")
    log_path = logs_dir / "laplacian_smoothing_log.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=logs_smoothing[0].keys())
        writer.writeheader()
        writer.writerows(logs_smoothing)
    print(f"    Saved to {log_path}")
    save_mesh(vertices_smoothed, faces, meshes_dir / "smoothed_bunny.obj")
    print("    Saved smoothed mesh")
    
    # ========== EVALUATION ==========
    print("\n" + "-" * 80)
    print("STEP 6: Evaluation and Comparison")
    print("-" * 80)
    
    print("\n6.1 Computing evaluation metrics...")
    
    # Evaluate all methods
    metrics_noisy = evaluate_mesh(vertices_noisy, vertices_gt, faces)
    metrics_curvature = evaluate_mesh(vertices_optimized, vertices_gt, faces)
    metrics_depth = evaluate_mesh(vertices_depth, vertices_gt, faces)
    metrics_normal = evaluate_mesh(vertices_normal, vertices_gt, faces)
    metrics_smoothed = evaluate_mesh(vertices_smoothed, vertices_gt, faces)
    
    # Print metrics table
    print("\n6.2 Metrics comparison:")
    methods_dict = {
        'Noisy Mesh': metrics_noisy,
        'Curvature Matching': metrics_curvature,
        'Depth Matching': metrics_depth,
        'Normal Matching': metrics_normal,
        'Laplacian Smoothing': metrics_smoothed,
    }
    
    # Select key metrics to display
    key_metrics = [
        'mean_vertex_error',
        'rmse_vertex_error',
        'max_vertex_error',
        'curvature_mse',
        'depth_mse',
        'normal_mse',
        'mean_radius_error',
        'normal_consistency_error',
    ]
    
    print_metrics_table(methods_dict, target_names=key_metrics)
    
    # Save metrics to CSV
    print("\n6.3 Saving metrics to CSV...")
    metrics_csv = logs_dir / "metrics_summary.csv"
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header: Method | metric1 | metric2 | ...
        writer.writerow(['Method'] + key_metrics)
        
        # Rows: one per method
        for method, metrics in methods_dict.items():
            row = [method]
            for metric in key_metrics:
                row.append(f"{metrics.get(metric, 0):.6f}")
            writer.writerow(row)
    
    print(f"    Saved to {metrics_csv}")
    
    # Save visualization comparison
    print("\n6.4 Creating comparison visualization...")
    comparison_vertices = {
        'Noisy': vertices_noisy,
        'Curvature': vertices_optimized,
        'Depth': vertices_depth,
        'Normal': vertices_normal,
        'Laplacian': vertices_smoothed,
    }
    
    # Create comparison figures for each objective's optimized mesh
    fig_path = figures_dir / "07_comparison_all_methods.png"
    create_comparison_figure(
        comparison_vertices,
        faces,
        titles=[
            'Noisy Input',
            'Curvature-Matched',
            'Depth-Matched',
            'Normal-Matched',
            'Laplacian Smoothed',
        ],
        save_path=fig_path,
        figsize=(24, 5),
    )
    print(f"    Saved comparison figure to {fig_path}")
    
    print("\n" + "-" * 80)
    print("EXPERIMENT COMPLETE")
    print("-" * 80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print(f"  - Meshes: {meshes_dir}")
    print(f"  - Figures: {figures_dir}")
    print(f"  - Logs: {logs_dir}")


if __name__ == "__main__":
    main()
