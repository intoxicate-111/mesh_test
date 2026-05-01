"""
Main script: Oracle Mesh-Space Curvature Matching Experiment

This script demonstrates curvature-guided mesh optimization on a sphere.
The experiment compares three methods:
1. Noisy mesh (baseline, no optimization)
2. Laplacian smoothing (classical baseline)
3. Curvature matching optimization (proposed method)
"""

import sys
import torch
import numpy as np
import os
import csv
import trimesh
from pathlib import Path

# Fix path for imports when running from scripts directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Local imports
from src.create_mesh import create_icosphere, add_radial_noise, compute_vertex_normals_from_positions
from src.curvature import compute_laplacian_curvature_proxy, compute_laplacian_curvature_vector
from src.optimise import optimize_mesh_objective, optimize_mesh_curvature, laplacian_smoothing_baseline
from src.evaluate import evaluate_mesh, print_metrics_table
from src.visualise import (
    visualize_mesh_3d,
    visualize_curvature_heatmap,
    plot_loss_curve,
    create_comparison_figure
)


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
    """Main experiment function."""
    
    # ========== SETUP ==========
    print("=" * 80)
    print("Oracle Mesh-Space Curvature Matching Experiment")
    print("=" * 80)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create output directories
    output_dir = Path("outputs")
    meshes_dir = output_dir / "meshes"
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"
    
    for d in [meshes_dir, figures_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Curvature loss mode for optimisation: 'scalar' or 'vector'
    curvature_mode = 'vector'
    view_direction = torch.tensor([0.0, 0.0, 1.0], device=device)
    
    # ========== MESH GENERATION ==========
    print("-" * 80)
    print("STEP 1: Mesh Generation")
    print("-" * 80)
    
    # Create clean sphere
    print("\n1.1 Creating clean sphere mesh...")
    vertices_gt, faces, mesh_obj = create_icosphere(subdivisions=3, radius=1.0, device=device)
    print(f"    Created icosphere with {vertices_gt.shape[0]} vertices and {faces.shape[0]} faces")
    
    # Compute vertex normals for the clean sphere
    normals_gt = compute_vertex_normals_from_positions(vertices_gt, faces)
    
    # Add noise to create noisy mesh
    print("\n1.2 Adding radial noise to create noisy mesh...")
    epsilon_noise = 0.1
    vertices_noisy = add_radial_noise(vertices_gt, normals_gt, epsilon=epsilon_noise, seed=42)
    print(f"    Noise strength (epsilon): {epsilon_noise}")
    
    # Compute target curvature from clean mesh
    print("\n1.3 Computing target curvature from clean mesh...")
    curvature_gt = compute_laplacian_curvature_proxy(vertices_gt, faces)
    curvature_vec_gt = compute_laplacian_curvature_vector(vertices_gt, faces)
    depth_gt = vertices_gt[:, 2]
    print(f"    Target curvature - Min: {curvature_gt.min():.6f}, Max: {curvature_gt.max():.6f}, "
          f"Mean: {curvature_gt.mean():.6f}")
    
    # Compute curvature of noisy mesh for reference
    curvature_noisy = compute_laplacian_curvature_proxy(vertices_noisy, faces)
    print(f"    Noisy curvature   - Min: {curvature_noisy.min():.6f}, Max: {curvature_noisy.max():.6f}, "
          f"Mean: {curvature_noisy.mean():.6f}")
    
    # ========== VISUALIZATION: INPUT MESHES ==========
    print("\n1.4 Saving input mesh visualizations...")
    visualize_mesh_3d(vertices_gt, faces, title="Clean Sphere",
                     save_path=figures_dir / "00_clean_sphere.png")
    visualize_mesh_3d(vertices_noisy, faces, title="Noisy Sphere",
                     save_path=figures_dir / "01_noisy_sphere.png")
    visualize_curvature_heatmap(vertices_gt, faces, curvature_gt,
                               title="Target Curvature (Clean Mesh)",
                               save_path=figures_dir / "02_target_curvature.png")
    visualize_curvature_heatmap(vertices_noisy, faces, curvature_noisy,
                               title="Initial Curvature (Noisy Mesh)",
                               save_path=figures_dir / "03_noisy_curvature.png")

    print("\n1.5 Saving input meshes...")
    save_mesh(vertices_gt, faces, meshes_dir / "clean_sphere.obj")
    save_mesh(vertices_noisy, faces, meshes_dir / "noisy_sphere.obj")
    
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
        lambda_zero_threshold=1e-4,
        min_lr_scale=0.1,
        device=device,
        verbose=True
    )
    
    print("\n2.2 Saving optimization log and loss curves...")
    # Save optimization logs to CSV
    log_path = logs_dir / "optimization_log.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=logs_optimization[0].keys())
        writer.writeheader()
        writer.writerows(logs_optimization)
    print(f"    Saved to {log_path}")
    
    # Plot loss curves
    plot_loss_curve(logs_optimization, save_path=figures_dir / "04_loss_curves.png")

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
        lambda_zero_threshold=1e-4,
        min_lr_scale=0.1,
        device=device,
        verbose=True
    )

    print("\n3.2 Saving depth log and loss curves...")
    log_path = logs_dir / "depth_log.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=logs_depth[0].keys())
        writer.writeheader()
        writer.writerows(logs_depth)
    print(f"    Saved to {log_path}")
    plot_loss_curve(logs_depth, save_path=figures_dir / "05_depth_loss_curves.png")

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
        lambda_zero_threshold=1e-4,
        min_lr_scale=0.1,
        device=device,
        verbose=True
    )

    print("\n4.2 Saving normal log and loss curves...")
    log_path = logs_dir / "normal_log.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=logs_normal[0].keys())
        writer.writeheader()
        writer.writerows(logs_normal)
    print(f"    Saved to {log_path}")
    plot_loss_curve(logs_normal, save_path=figures_dir / "06_normal_loss_curves.png")
    
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
    log_path = logs_dir / "smoothing_log.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=logs_smoothing[0].keys())
        writer.writeheader()
        writer.writerows(logs_smoothing)
    print(f"    Saved to {log_path}")
    
    # ========== EVALUATION ==========
    print("\n" + "-" * 80)
    print("STEP 6: Evaluation and Comparison")
    print("-" * 80)
    
    print("\n6.1 Computing evaluation metrics...")
    
    # Evaluate noisy mesh
    metrics_noisy = evaluate_mesh(vertices_noisy, vertices_gt, faces)
    
    # Evaluate depth-optimized mesh
    metrics_depth = evaluate_mesh(vertices_depth, vertices_gt, faces)

    # Evaluate normal-optimized mesh
    metrics_normal = evaluate_mesh(vertices_normal, vertices_gt, faces)

    # Evaluate smoothed mesh
    metrics_smoothed = evaluate_mesh(vertices_smoothed, vertices_gt, faces)
    
    # Evaluate optimized mesh
    metrics_optimized = evaluate_mesh(vertices_optimized, vertices_gt, faces)
    
    # Print metrics table
    print("\n6.2 Metrics comparison:")
    methods_dict = {
        'Noisy Mesh': metrics_noisy,
        'Curvature Matching': metrics_optimized,
        'Depth Matching': metrics_depth,
        'Normal Matching': metrics_normal,
        'Laplacian Smoothing': metrics_smoothed,
    }
    
    # Select key metrics to display
    key_metrics = [
        'mean_vertex_error',
        'rmse_vertex_error',
        'curvature_mse',
        'depth_mse',
        'normal_mse',
        'mean_radius_error',
        'normal_error'
    ]
    
    selected_methods = {k: {m: v[m] for m in key_metrics if m in v}
                       for k, v in methods_dict.items()}
    
    print_metrics_table(selected_methods, target_names=key_metrics)
    
    # Save metrics to CSV
    metrics_path = output_dir / "metrics.csv"
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['Method'] + list(metrics_noisy.keys()))
        # Data
        for method_name, metrics in methods_dict.items():
            writer.writerow([method_name] + [metrics[k] for k in metrics_noisy.keys()])
    print(f"\n    Saved detailed metrics to {metrics_path}")
    
    # ========== VISUALIZATION: RESULTS ==========
    print("\n6.3 Saving result visualizations...")
    
    # Visualize results
    visualize_mesh_3d(vertices_smoothed, faces, title="Laplacian Smoothed Mesh",
                     save_path=figures_dir / "05_smoothed_sphere.png")
    visualize_mesh_3d(vertices_optimized, faces, title="Curvature-Optimized Mesh",
                     save_path=figures_dir / "07_optimized_sphere.png")
    visualize_mesh_3d(vertices_depth, faces, title="Depth-Optimized Mesh",
                     save_path=figures_dir / "08_depth_optimized_sphere.png")
    visualize_mesh_3d(vertices_normal, faces, title="Normal-Optimized Mesh",
                     save_path=figures_dir / "09_normal_optimized_sphere.png")
    
    # Curvature heatmaps for results
    curvature_smoothed = compute_laplacian_curvature_proxy(vertices_smoothed, faces)
    curvature_optimized = compute_laplacian_curvature_proxy(vertices_optimized, faces)
    curvature_depth = compute_laplacian_curvature_proxy(vertices_depth, faces)
    curvature_normal = compute_laplacian_curvature_proxy(vertices_normal, faces)
    
    visualize_curvature_heatmap(vertices_smoothed, faces, curvature_smoothed,
                               title="Curvature After Smoothing",
                               save_path=figures_dir / "10_smoothed_curvature.png")
    visualize_curvature_heatmap(vertices_optimized, faces, curvature_optimized,
                               title="Curvature After Optimization",
                               save_path=figures_dir / "11_optimized_curvature.png")
    visualize_curvature_heatmap(vertices_depth, faces, curvature_depth,
                               title="Curvature After Depth Optimization",
                               save_path=figures_dir / "12_depth_curvature.png")
    visualize_curvature_heatmap(vertices_normal, faces, curvature_normal,
                               title="Curvature After Normal Optimization",
                               save_path=figures_dir / "13_normal_curvature.png")
    
    # Side-by-side comparison
    comparison_vertices = {
        'Clean': vertices_gt,
        'Noisy': vertices_noisy,
        'Curvature': vertices_optimized,
        'Depth': vertices_depth,
        'Normal': vertices_normal,
        'Smoothed': vertices_smoothed,
    }
    create_comparison_figure(
        comparison_vertices,
        faces,
        titles=['Clean Sphere', 'Noisy Sphere', 'Curvature-Matched', 'Depth-Matched', 'Normal-Matched', 'Laplacian Smoothed'],
        save_path=figures_dir / "14_comparison.png",
        figsize=(24, 5)
    )

    print("\n6.4 Saving result meshes...")
    save_mesh(vertices_smoothed, faces, meshes_dir / "laplacian_smoothed.obj")
    save_mesh(vertices_optimized, faces, meshes_dir / "curvature_optimized.obj")
    save_mesh(vertices_depth, faces, meshes_dir / "depth_optimized.obj")
    save_mesh(vertices_normal, faces, meshes_dir / "normal_optimized.obj")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"  - Meshes saved to: {meshes_dir}")
    print(f"  - Figures saved to: {figures_dir}")
    print(f"  - Logs saved to: {logs_dir}")
    print(f"\nKey results:")
    print(f"  - Noisy mesh vertex error: {metrics_noisy['mean_vertex_error']:.6f}")
    print(f"  - Smoothed mesh vertex error: {metrics_smoothed['mean_vertex_error']:.6f}")
    print(f"  - Optimized mesh vertex error: {metrics_optimized['mean_vertex_error']:.6f}")
    print(f"  - Depth mesh vertex error: {metrics_depth['mean_vertex_error']:.6f}")
    print(f"  - Normal mesh vertex error: {metrics_normal['mean_vertex_error']:.6f}")
    print(f"\n  - Target curvature MSE (noisy): {metrics_noisy['curvature_mse']:.6f}")
    print(f"  - Target curvature MSE (smoothed): {metrics_smoothed['curvature_mse']:.6f}")
    print(f"  - Target curvature MSE (optimized): {metrics_optimized['curvature_mse']:.6f}")
    print(f"  - Target curvature MSE (depth): {metrics_depth['curvature_mse']:.6f}")
    print(f"  - Target curvature MSE (normal): {metrics_normal['curvature_mse']:.6f}")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
