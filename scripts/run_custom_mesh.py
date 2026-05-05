"""
Run mesh optimization on an arbitrary input mesh.

Usage example:
  python scripts/run_custom_mesh.py --mesh path/to/model.obj --objective curvature --curvature-mode vector
"""

import sys
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import trimesh

# Use non-interactive backend for headless runs
import matplotlib
matplotlib.use("Agg")

# Fix path for imports when running from scripts directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.create_mesh import add_radial_noise, compute_vertex_normals_from_positions
from src.curvature import (
    compute_laplacian_curvature_proxy,
    compute_laplacian_curvature_vector,
    compute_weighted_laplacian_curvature_proxy,
    compute_weighted_laplacian_vector,
)
from src.minimum_ball import build_minimum_ball_graph, build_minimum_ball_faces, build_knn_faces
from src.optimise import optimize_mesh_objective, laplacian_smoothing_baseline
from src.evaluate import evaluate_mesh, print_metrics_table
from src.visualise import visualize_mesh_3d, visualize_curvature_heatmap, plot_loss_curve, create_comparison_figure
from src.mesh_utils import build_dynamic_graph


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize an arbitrary mesh with multiple objectives.")
    parser.add_argument("--mesh", type=str, required=True, help="Path to input mesh file (obj/ply/stl).")
    parser.add_argument("--output", type=str, default="outputs/custom_mesh", help="Output directory.")

    parser.add_argument("--objective", type=str, default="curvature",
                        choices=["curvature", "depth", "normal"], help="Optimization objective.")
    parser.add_argument("--curvature-mode", type=str, default="vector",
                        choices=["scalar", "vector"], help="Curvature mode for curvature objective.")
    parser.add_argument("--curvature-graph-mode", type=str, default="mesh",
                        choices=["mesh", "knn", "minimum_ball"],
                        help="Graph used for curvature computation.")

    parser.add_argument("--noise-mode", type=str, default="mixed",
                        choices=["radial", "tangential", "mixed"], help="Noise mode for corruption.")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Noise strength.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for noise.")

    parser.add_argument("--no-normalize", action="store_true",
                        help="Disable normalization (default is normalized to unit scale).")
    parser.add_argument("--num-iterations", type=int, default=50000, help="Optimization iterations.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--lambda-edge", type=float, default=0.0, help="Edge regularization weight.")
    parser.add_argument("--lambda-pos", type=float, default=0.0, help="Position regularization weight.")
    parser.add_argument("--lambda-edge-min", type=float, default=0.0,
                        help="Min edge length penalty weight (dynamic graph).")
    parser.add_argument("--lambda-edge-var", type=float, default=0.0,
                        help="Edge length uniformity penalty weight (dynamic graph).")
    parser.add_argument("--lambda-repulsion", type=float, default=0.0,
                        help="Neighbor repulsion penalty weight (dynamic graph).")
    parser.add_argument("--lambda-center-scale", type=float, default=0.0,
                        help="Global center/scale regularization weight.")
    parser.add_argument("--min-edge-length", type=float, default=0.0,
                        help="Minimum edge length for collapse prevention.")
    parser.add_argument("--repulsion-min-dist", type=float, default=0.0,
                        help="Repulsion hinge distance (defaults to min-edge-length when 0).")

    parser.add_argument("--connectivity-mode", type=str, default="faces",
                        choices=["faces", "dynamic"],
                        help="Connectivity source: mesh faces or dynamic point graph.")
    parser.add_argument("--graph-mode", type=str, default="knn",
                        choices=["knn", "radius", "min_ball"],
                        help="Dynamic graph construction method.")
    parser.add_argument("--graph-k", type=int, default=16, help="k for kNN/min-ball graphs.")
    parser.add_argument("--graph-radius", type=float, default=0.1, help="Radius for radius graph.")
    parser.add_argument("--graph-chunk-size", type=int, default=2048,
                        help="Chunk size for dynamic graph distance computation.")
    parser.add_argument("--graph-update-interval", type=int, default=20,
                        help="Rebuild dynamic graph every N iterations.")
    parser.add_argument("--edge-target-loss-in-dynamic", action="store_true",
                        help="Enable target edge-length matching for dynamic graphs (not recommended).")

    parser.add_argument("--min-ball-k", type=int, default=16,
                        help="KNN size for minimum-ball triangle proposals.")
    parser.add_argument("--min-ball-alpha", type=float, default=10.0,
                        help="Sigmoid sharpness for minimum-ball face weights.")
    parser.add_argument("--min-ball-max-triangles", type=int, default=32,
                        help="Max triangles per vertex from KNN proposals.")
    parser.add_argument("--min-ball-tri-chunk-size", type=int, default=2048,
                        help="Chunk size for minimum-ball face evaluation.")
    parser.add_argument("--min-ball-face-weight-threshold", type=float, default=0.5,
                        help="Face weight threshold when exporting minimum-ball faces.")

    parser.add_argument("--enable-schedule", dest="enable_schedule", action="store_true",
                        help="Enable dynamic lr/lambda decay.")
    parser.add_argument("--disable-schedule", dest="enable_schedule", action="store_false",
                        help="Disable dynamic lr/lambda decay.")
    parser.set_defaults(enable_schedule=True)
    parser.add_argument("--plateau-patience", type=int, default=50,
                        help="Epochs without significant loss improvement before decay.")
    parser.add_argument("--plateau-min-delta", type=float, default=1e-5,
                        help="Minimum loss improvement treated as significant.")
    parser.add_argument("--decay-factor", type=float, default=0.5,
                        help="Multiplier applied when plateau is detected.")
    parser.add_argument("--lambda-zero-threshold", type=float, default=1e-4,
                        help="Threshold below which lambda_edge/pos are set to zero when decayed.")
    parser.add_argument("--min-lr-scale", type=float, default=0.1, help="Min lr scale at end.")

    parser.add_argument("--run-laplacian", action="store_true", help="Also run Laplacian smoothing baseline.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device.")

    return parser.parse_args()


def load_mesh(path):
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded asset is not a mesh.")
    return mesh


def normalize_mesh(vertices):
    v_min = torch.min(vertices, dim=0)[0]
    v_max = torch.max(vertices, dim=0)[0]
    centroid = (v_min + v_max) / 2.0
    scale = torch.max(v_max - v_min)
    if scale <= 0:
        return vertices
    vertices = vertices - centroid
    vertices = vertices / (scale / 2.0)
    return vertices


def save_mesh(vertices, faces, save_path):
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


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output)
    meshes_dir = output_dir / "meshes"
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"
    for d in [meshes_dir, figures_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    print(f"Loading mesh: {mesh_path}")
    mesh = load_mesh(str(mesh_path))

    vertices_gt = torch.from_numpy(mesh.vertices.astype(np.float32)).to(device)
    faces = torch.from_numpy(mesh.faces.astype(np.int64)).to(device)

    if not args.no_normalize:
        print("Normalizing mesh to unit scale...")
        vertices_gt = normalize_mesh(vertices_gt)

    print(f"Mesh: {vertices_gt.shape[0]} vertices, {faces.shape[0]} faces")

    normals_gt = compute_vertex_normals_from_positions(vertices_gt, faces)

    print("Adding noise...")
    vertices_noisy = add_radial_noise(
        vertices_gt,
        normals_gt,
        epsilon=args.epsilon,
        seed=args.seed,
        noise_mode=args.noise_mode,
    )

    if args.curvature_graph_mode == "minimum_ball":
        edges_gt, edge_w_gt, _ = build_minimum_ball_graph(
            vertices_gt,
            k=args.min_ball_k,
            alpha_min=args.min_ball_alpha,
            max_triangles_per_vertex=args.min_ball_max_triangles,
            knn_chunk_size=args.graph_chunk_size,
            tri_chunk_size=args.min_ball_tri_chunk_size,
        )
        edges_noisy, edge_w_noisy, _ = build_minimum_ball_graph(
            vertices_noisy,
            k=args.min_ball_k,
            alpha_min=args.min_ball_alpha,
            max_triangles_per_vertex=args.min_ball_max_triangles,
            knn_chunk_size=args.graph_chunk_size,
            tri_chunk_size=args.min_ball_tri_chunk_size,
        )
        curvature_gt = compute_weighted_laplacian_curvature_proxy(vertices_gt, edges_gt, edge_w_gt)
        curvature_vec_gt = compute_weighted_laplacian_vector(vertices_gt, edges_gt, edge_w_gt)
        curvature_noisy = compute_weighted_laplacian_curvature_proxy(vertices_noisy, edges_noisy, edge_w_noisy)
    elif args.curvature_graph_mode == "knn":
        edges_gt = build_dynamic_graph(
            vertices_gt,
            mode="knn",
            k=args.graph_k,
            radius=args.graph_radius,
            chunk_size=args.graph_chunk_size,
        )
        edges_noisy = build_dynamic_graph(
            vertices_noisy,
            mode="knn",
            k=args.graph_k,
            radius=args.graph_radius,
            chunk_size=args.graph_chunk_size,
        )
        curvature_gt = compute_laplacian_curvature_proxy(vertices_gt, faces, edges=edges_gt)
        curvature_vec_gt = compute_laplacian_curvature_vector(vertices_gt, faces, edges=edges_gt)
        curvature_noisy = compute_laplacian_curvature_proxy(vertices_noisy, faces, edges=edges_noisy)
    else:
        curvature_gt = compute_laplacian_curvature_proxy(vertices_gt, faces)
        curvature_vec_gt = compute_laplacian_curvature_vector(vertices_gt, faces)
        curvature_noisy = compute_laplacian_curvature_proxy(vertices_noisy, faces)

    depth_gt = vertices_gt[:, 2]

    print("Saving input visualizations...")
    visualize_mesh_3d(vertices_gt, faces, title="Clean Mesh", save_path=figures_dir / "00_clean.png")
    visualize_mesh_3d(vertices_noisy, faces, title="Noisy Mesh", save_path=figures_dir / "01_noisy.png")
    visualize_curvature_heatmap(vertices_gt, faces, curvature_gt,
                                title="Target Curvature", save_path=figures_dir / "02_target_curvature.png")
    visualize_curvature_heatmap(vertices_noisy, faces, curvature_noisy,
                                title="Noisy Curvature", save_path=figures_dir / "03_noisy_curvature.png")

    save_mesh(vertices_gt, faces, meshes_dir / "clean_mesh.obj")
    save_mesh(vertices_noisy, faces, meshes_dir / "noisy_mesh.obj")

    print(f"Optimizing objective: {args.objective}")
    if args.objective == "curvature":
        objective_target = curvature_vec_gt if args.curvature_mode == "vector" else curvature_gt
    elif args.objective == "depth":
        objective_target = depth_gt
    else:
        objective_target = normals_gt

    vertices_opt, logs = optimize_mesh_objective(
        vertices_noisy,
        vertices_gt,
        faces,
        objective_target,
        objective_mode=args.objective,
        curvature_mode=args.curvature_mode,
        num_iterations=args.num_iterations,
        learning_rate=args.learning_rate,
        lambda_edge=args.lambda_edge,
        lambda_pos=args.lambda_pos,
        lambda_edge_min=args.lambda_edge_min,
        lambda_edge_var=args.lambda_edge_var,
        lambda_repulsion=args.lambda_repulsion,
        lambda_center_scale=args.lambda_center_scale,
        min_edge_length=args.min_edge_length,
        repulsion_min_dist=args.repulsion_min_dist,
        curvature_graph_mode=args.curvature_graph_mode,
        min_ball_k=args.min_ball_k,
        min_ball_alpha=args.min_ball_alpha,
        min_ball_max_triangles=args.min_ball_max_triangles,
        min_ball_tri_chunk_size=args.min_ball_tri_chunk_size,
        connectivity_mode=args.connectivity_mode,
        graph_mode=args.graph_mode,
        graph_k=args.graph_k,
        graph_radius=args.graph_radius,
        graph_chunk_size=args.graph_chunk_size,
        graph_update_interval=args.graph_update_interval,
        edge_target_loss_in_dynamic=args.edge_target_loss_in_dynamic,
        enable_dynamic_schedule=args.enable_schedule,
        plateau_patience=args.plateau_patience,
        plateau_min_delta=args.plateau_min_delta,
        decay_factor=args.decay_factor,
        lambda_zero_threshold=args.lambda_zero_threshold,
        min_lr_scale=args.min_lr_scale,
        device=device,
        verbose=True,
    )

    log_path = logs_dir / "optimization_log.csv"
    fieldnames = sorted({key for row in logs for key in row.keys()})
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(logs)

    plot_loss_curve(logs, save_path=figures_dir / "04_loss_curves.png")

    if args.curvature_graph_mode == "minimum_ball":
        faces_mb, _face_w_mb, _ = build_minimum_ball_faces(
            vertices_opt,
            k=args.min_ball_k,
            alpha_min=args.min_ball_alpha,
            max_triangles_per_vertex=args.min_ball_max_triangles,
            knn_chunk_size=args.graph_chunk_size,
            tri_chunk_size=args.min_ball_tri_chunk_size,
            face_weight_threshold=args.min_ball_face_weight_threshold,
        )
        if faces_mb.numel() > 0:
            save_mesh(vertices_opt, faces_mb, meshes_dir / f"optimized_{args.objective}.obj")
        else:
            save_mesh(vertices_opt, faces, meshes_dir / f"optimized_{args.objective}.obj")
    elif args.curvature_graph_mode == "knn":
        faces_knn, _knn_diag = build_knn_faces(
            vertices_opt,
            k=args.graph_k,
            max_triangles_per_vertex=args.min_ball_max_triangles,
            knn_chunk_size=args.graph_chunk_size,
        )
        if faces_knn.numel() > 0:
            save_mesh(vertices_opt, faces_knn, meshes_dir / f"optimized_{args.objective}.obj")
        else:
            save_mesh(vertices_opt, faces, meshes_dir / f"optimized_{args.objective}.obj")
    else:
        save_mesh(vertices_opt, faces, meshes_dir / f"optimized_{args.objective}.obj")

    metrics_noisy = evaluate_mesh(vertices_noisy, vertices_gt, faces)
    metrics_opt = evaluate_mesh(vertices_opt, vertices_gt, faces)

    methods = {
        "Noisy Mesh": metrics_noisy,
        f"Optimized ({args.objective})": metrics_opt,
    }
    print_metrics_table(methods)

    if args.run_laplacian:
        print("Running Laplacian smoothing baseline...")
        vertices_smooth, logs_smooth = laplacian_smoothing_baseline(
            vertices_noisy,
            vertices_gt,
            faces,
            num_iterations=min(args.num_iterations, 1000),
            alpha=0.1,
            device=device,
            verbose=True,
        )
        save_mesh(vertices_smooth, faces, meshes_dir / "smoothed.obj")
        methods["Laplacian"] = evaluate_mesh(vertices_smooth, vertices_gt, faces)

        print_metrics_table(methods)

        comparison = {
            "Noisy": vertices_noisy,
            "Optimized": vertices_opt,
            "Laplacian": vertices_smooth,
        }
        create_comparison_figure(
            comparison,
            faces,
            titles=["Noisy", "Optimized", "Laplacian"],
            save_path=figures_dir / "05_comparison.png",
            figsize=(18, 5),
        )
    else:
        comparison = {
            "Noisy": vertices_noisy,
            "Optimized": vertices_opt,
        }
        create_comparison_figure(
            comparison,
            faces,
            titles=["Noisy", "Optimized"],
            save_path=figures_dir / "05_comparison.png",
            figsize=(12, 5),
        )

    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
