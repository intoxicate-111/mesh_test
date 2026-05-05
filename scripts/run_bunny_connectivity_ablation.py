"""
Connectivity ablation on Stanford Bunny.

Compares curvature optimization with mesh, KNN, and minimum-ball graphs.
"""
import sys
import csv
from pathlib import Path

import numpy as np
import torch
import trimesh

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.create_mesh import add_radial_noise, compute_vertex_normals_from_positions
from src.curvature import (
    compute_laplacian_curvature_proxy,
    compute_laplacian_curvature_vector,
    compute_weighted_laplacian_curvature_proxy,
    compute_weighted_laplacian_vector,
)
from src.minimum_ball import build_minimum_ball_graph
from src.optimise import optimize_mesh_objective
from src.evaluate import evaluate_mesh, print_metrics_table
from src.visualise import visualize_mesh_3d, visualize_curvature_heatmap, plot_loss_curve


def load_stanford_bunny(device='cpu'):
    bunny_url = "https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj"
    mesh = trimesh.load(bunny_url, process=False, allow_remote=True)

    vertices = torch.from_numpy(mesh.vertices.astype(np.float32)).to(device)
    faces = torch.from_numpy(mesh.faces.astype(np.int64)).to(device)

    v_min = torch.min(vertices, dim=0)[0]
    v_max = torch.max(vertices, dim=0)[0]
    centroid = (v_min + v_max) / 2.0
    bb_size = torch.max(v_max - v_min)
    vertices = (vertices - centroid) / (bb_size / 2.0)

    return vertices, faces


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path("bunny_outputs_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Stanford Bunny...")
    vertices_gt, faces = load_stanford_bunny(device=device)
    normals_gt = compute_vertex_normals_from_positions(vertices_gt, faces)

    print("Adding noise...")
    vertices_noisy = add_radial_noise(vertices_gt, normals_gt, epsilon=0.05, seed=42, noise_mode='mixed')

    # Target curvature uses mesh connectivity by default.
    curvature_gt = compute_laplacian_curvature_proxy(vertices_gt, faces)
    curvature_vec_gt = compute_laplacian_curvature_vector(vertices_gt, faces)

    visualize_mesh_3d(vertices_gt, faces, title="Clean Bunny", save_path=output_dir / "00_clean.png")
    visualize_mesh_3d(vertices_noisy, faces, title="Noisy Bunny", save_path=output_dir / "01_noisy.png")
    visualize_curvature_heatmap(vertices_gt, faces, curvature_gt,
                                title="Target Curvature", save_path=output_dir / "02_target_curvature.png")

    configs = [
        {
            "name": "mesh",
            "curvature_graph_mode": "mesh",
            "connectivity_mode": "faces",
            "graph_mode": "knn",
        },
        {
            "name": "knn",
            "curvature_graph_mode": "knn",
            "connectivity_mode": "dynamic",
            "graph_mode": "knn",
        },
        {
            "name": "minimum_ball",
            "curvature_graph_mode": "minimum_ball",
            "connectivity_mode": "dynamic",
            "graph_mode": "knn",
        },
    ]

    metrics = {}

    for cfg in configs:
        print(f"\n=== Running {cfg['name']} connectivity ===")
        mode_dir = output_dir / cfg["name"]
        mode_dir.mkdir(parents=True, exist_ok=True)

        vertices_opt, logs = optimize_mesh_objective(
            vertices_noisy,
            vertices_gt,
            faces,
            curvature_vec_gt,
            objective_mode='curvature',
            curvature_mode='vector',
            num_iterations=5000,
            learning_rate=1e-4,
            lambda_edge=0.0,
            lambda_pos=0.0,
            lambda_edge_min=1e-4,
            lambda_edge_var=1e-4,
            lambda_repulsion=0.0,
            lambda_center_scale=1e-4,
            min_edge_length=0.0,
            repulsion_min_dist=0.0,
            curvature_graph_mode=cfg["curvature_graph_mode"],
            min_ball_k=16,
            min_ball_alpha=10.0,
            min_ball_max_triangles=32,
            min_ball_tri_chunk_size=2048,
            connectivity_mode=cfg["connectivity_mode"],
            graph_mode=cfg["graph_mode"],
            graph_k=16,
            graph_radius=0.01,
            graph_chunk_size=2048,
            graph_update_interval=20,
            edge_target_loss_in_dynamic=False,
            enable_dynamic_schedule=True,
            plateau_patience=50,
            plateau_min_delta=1e-6,
            decay_factor=0.5,
            lambda_zero_threshold=1e-4,
            schedule_delay_iters=2000,
            min_lr_scale=0.01,
            device=device,
            verbose=True,
        )

        log_path = mode_dir / "optimization_log.csv"
        fieldnames = sorted({key for row in logs for key in row.keys()})
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(logs)

        plot_loss_curve(logs, save_path=mode_dir / "loss_curves.png")
        visualize_mesh_3d(vertices_opt, faces, title=f"{cfg['name']} optimized",
                          save_path=mode_dir / "optimized.png")

        metrics[cfg["name"]] = evaluate_mesh(vertices_opt, vertices_gt, faces)

    print("\nConnectivity ablation metrics:")
    print_metrics_table(metrics)


if __name__ == "__main__":
    main()
