"""
Advanced example: Custom mesh optimization with different parameters

This script shows how to use the curvature optimization modules
with custom parameters and configurations.
"""

import sys
import torch
from pathlib import Path

# Fix path for imports when running from scripts directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.create_mesh import create_icosphere, add_radial_noise, compute_vertex_normals_from_positions
from src.curvature import compute_laplacian_curvature_proxy
from src.optimise import optimize_mesh_curvature
from src.evaluate import evaluate_mesh, print_metrics_table
from src.visualise import create_comparison_figure, visualize_curvature_heatmap


def custom_optimization_example():
    """
    Example showing how to customize the optimization process.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create output directory
    output_dir = Path("outputs/custom_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== Custom Parameters =====
    # Mesh parameters
    mesh_subdivisions = 4  # Higher resolution
    mesh_radius = 1.0
    
    # Noise parameters
    noise_strength = 0.05  # Lower noise
    noise_seed = 123
    
    # Optimization parameters
    opt_iterations = 1000
    opt_lr = 0.005
    opt_lambda_edge = 0.2
    opt_lambda_pos = 0.005
    
    print("\n" + "="*60)
    print("CUSTOM OPTIMIZATION EXAMPLE")
    print("="*60)
    print(f"\nMesh: icosphere level {mesh_subdivisions}, radius {mesh_radius}")
    print(f"Noise: strength {noise_strength}, seed {noise_seed}")
    print(f"Optimization: {opt_iterations} iter, lr {opt_lr}")
    print(f"  λ_edge = {opt_lambda_edge}, λ_pos = {opt_lambda_pos}")
    
    # ===== Create Meshes =====
    print("\nGenerating meshes...")
    vertices_gt, faces, _ = create_icosphere(subdivisions=mesh_subdivisions, 
                                             radius=mesh_radius, 
                                             device=device)
    
    normals_gt = compute_vertex_normals_from_positions(vertices_gt, faces)
    vertices_noisy = add_radial_noise(vertices_gt, normals_gt, 
                                     epsilon=noise_strength, seed=noise_seed)
    
    curvature_gt = compute_laplacian_curvature_proxy(vertices_gt, faces)
    
    print(f"Mesh: {vertices_gt.shape[0]} vertices, {faces.shape[0]} faces")
    
    # ===== Optimize =====
    print("\nOptimizing...")
    vertices_optimized, logs = optimize_mesh_curvature(
        vertices_noisy,
        vertices_gt,
        faces,
        curvature_gt,
        num_iterations=opt_iterations,
        learning_rate=opt_lr,
        lambda_edge=opt_lambda_edge,
        lambda_pos=opt_lambda_pos,
        device=device,
        verbose=False  # Suppress iteration output
    )
    
    # Print optimization summary
    print(f"\nOptimization summary:")
    print(f"  Initial loss: {logs[0]['total']:.6f}")
    print(f"  Final loss:   {logs[-1]['total']:.6f}")
    print(f"  Loss reduction: {logs[0]['total']/logs[-1]['total']:.2f}x")
    
    # ===== Evaluate =====
    print("\nEvaluating...")
    metrics_noisy = evaluate_mesh(vertices_noisy, vertices_gt, faces)
    metrics_optimized = evaluate_mesh(vertices_optimized, vertices_gt, faces)
    
    print("\nResults comparison:")
    print(f"{'Metric':<25} {'Noisy':<15} {'Optimized':<15} {'Improvement':<15}")
    print("-" * 70)
    
    for key in ['mean_vertex_error', 'curvature_mse', 'mean_radius_error']:
        noisy_val = metrics_noisy[key]
        opt_val = metrics_optimized[key]
        improvement = (noisy_val - opt_val) / noisy_val * 100 if noisy_val != 0 else 0
        print(f"{key:<25} {noisy_val:<15.6f} {opt_val:<15.6f} {improvement:>13.1f}%")
    
    # ===== Visualize =====
    print("\nGenerating visualizations...")
    
    # Comparison figure
    comparison_dict = {
        'Ground Truth': vertices_gt,
        'Noisy': vertices_noisy,
        'Optimized': vertices_optimized
    }
    
    create_comparison_figure(
        comparison_dict,
        faces,
        titles=['Ground Truth', 'Noisy Input', 'Optimized Output'],
        save_path=output_dir / "comparison.png",
        figsize=(15, 5)
    )
    
    # Curvature comparison
    curvature_optimized = compute_laplacian_curvature_proxy(vertices_optimized, faces)
    
    visualize_curvature_heatmap(
        vertices_gt, faces, curvature_gt,
        title="Target Curvature",
        save_path=output_dir / "curvature_target.png"
    )
    
    visualize_curvature_heatmap(
        vertices_optimized, faces, curvature_optimized,
        title="Optimized Curvature",
        save_path=output_dir / "curvature_optimized.png"
    )
    
    print(f"\nOutputs saved to {output_dir}")
    print("="*60 + "\n")


def sensitivity_analysis():
    """
    Analyze sensitivity to noise level and regularization weights.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create base mesh
    vertices_gt, faces, _ = create_icosphere(subdivisions=3, radius=1.0, device=device)
    normals_gt = compute_vertex_normals_from_positions(vertices_gt, faces)
    curvature_gt = compute_laplacian_curvature_proxy(vertices_gt, faces)
    
    # Test different noise levels
    noise_levels = [0.03, 0.05, 0.10, 0.15]
    regularization_weights = [0.05, 0.1, 0.2, 0.5]
    
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS")
    print("="*60)
    
    results = {}
    
    for noise in noise_levels:
        print(f"\n--- Noise level: {noise} ---")
        vertices_noisy = add_radial_noise(vertices_gt, normals_gt, epsilon=noise, seed=42)
        
        for lambda_edge in regularization_weights:
            # Quick optimization
            vertices_opt, _ = optimize_mesh_curvature(
                vertices_noisy, vertices_gt, faces, curvature_gt,
                num_iterations=200,
                learning_rate=0.01,
                lambda_edge=lambda_edge,
                lambda_pos=0.01,
                device=device,
                verbose=False
            )
            
            metrics = evaluate_mesh(vertices_opt, vertices_gt, faces)
            vertex_error = metrics['mean_vertex_error']
            
            key = (noise, lambda_edge)
            results[key] = vertex_error
            print(f"  λ_edge={lambda_edge:<4} → vertex_error: {vertex_error:.6f}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Run custom optimization example
    custom_optimization_example()
    
    # Run sensitivity analysis (optional, can be slow)
    # sensitivity_analysis()
