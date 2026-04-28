"""
Evaluation metrics for mesh optimization.
"""
import torch
import numpy as np
from src.curvature import compute_laplacian_curvature_proxy


def compute_vertex_errors(vertices_pred, vertices_gt):
    """
    Compute various vertex position errors.
    
    Args:
        vertices_pred: torch tensor of shape (V, 3)
        vertices_gt: torch tensor of shape (V, 3)
    
    Returns:
        dict with mean, max, and RMSE errors
    """
    errors = torch.norm(vertices_pred - vertices_gt, dim=1)
    
    return {
        'mean_vertex_error': torch.mean(errors).item(),
        'max_vertex_error': torch.max(errors).item(),
        'rmse_vertex_error': torch.sqrt(torch.mean(errors ** 2)).item(),
        'std_vertex_error': torch.std(errors).item()
    }


def compute_curvature_error(curvature_pred, curvature_gt):
    """
    Compute curvature matching error.
    
    Args:
        curvature_pred: torch tensor of shape (V,)
        curvature_gt: torch tensor of shape (V,)
    
    Returns:
        dict with curvature errors
    """
    curv_error = torch.abs(curvature_pred - curvature_gt)
    
    return {
        'curvature_mse': torch.mean((curvature_pred - curvature_gt) ** 2).item(),
        'curvature_mae': torch.mean(curv_error).item(),
        'curvature_max_error': torch.max(curv_error).item()
    }


def compute_radius_error(vertices_pred, target_radius=1.0):
    """
    Compute radius error for a sphere.
    
    For a sphere of radius R, measure how much vertices deviate from this radius.
    
    Args:
        vertices_pred: torch tensor of shape (V, 3)
        target_radius: expected radius of the sphere
    
    Returns:
        dict with radius errors
    """
    radii = torch.norm(vertices_pred, dim=1)
    radius_errors = torch.abs(radii - target_radius)
    
    return {
        'mean_radius_error': torch.mean(radius_errors).item(),
        'max_radius_error': torch.max(radius_errors).item(),
        'std_radius_error': torch.std(radius_errors).item()
    }


def compute_normal_error(vertices_pred, normals_gt):
    """
    Compute normal consistency error.
    
    Args:
        vertices_pred: torch tensor of shape (V, 3)
        normals_gt: torch tensor of shape (V, 3), ground truth normals
    
    Returns:
        dict with normal errors
    """
    # Compute normals from predicted vertices
    # For a sphere centered at origin, normals are normalized vertex positions
    normals_pred = vertices_pred / (torch.norm(vertices_pred, dim=1, keepdim=True) + 1e-8)
    
    # Compute dot product (cosine similarity)
    dot_product = torch.sum(normals_pred * normals_gt, dim=1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Compute angle in radians
    angles = torch.acos(dot_product)
    angles_deg = angles * 180.0 / np.pi
    
    normal_error = 1.0 - dot_product
    
    return {
        'normal_error': torch.mean(normal_error).item(),
        'normal_error_max': torch.max(normal_error).item(),
        'angle_error_deg': torch.mean(angles_deg).item(),
        'angle_error_max_deg': torch.max(angles_deg).item()
    }


def evaluate_mesh(vertices_pred, vertices_gt, faces, target_radius=1.0):
    """
    Comprehensively evaluate mesh quality compared to ground truth.
    
    Args:
        vertices_pred: torch tensor of shape (V, 3)
        vertices_gt: torch tensor of shape (V, 3)
        faces: torch tensor of shape (F, 3)
        target_radius: expected radius for a sphere
    
    Returns:
        dict with all evaluation metrics
    """
    # Compute curvature
    curvature_pred = compute_laplacian_curvature_proxy(vertices_pred, faces)
    curvature_gt = compute_laplacian_curvature_proxy(vertices_gt, faces)
    
    # Compute normals
    normals_gt = vertices_gt / (torch.norm(vertices_gt, dim=1, keepdim=True) + 1e-8)
    normals_pred = vertices_pred / (torch.norm(vertices_pred, dim=1, keepdim=True) + 1e-8)

    # Depth along z-axis for this sphere experiment
    depth_pred = vertices_pred[:, 2]
    depth_gt = vertices_gt[:, 2]
    
    # Collect all metrics
    metrics = {}
    metrics.update(compute_vertex_errors(vertices_pred, vertices_gt))
    metrics.update(compute_curvature_error(curvature_pred, curvature_gt))
    metrics.update(compute_radius_error(vertices_pred, target_radius))
    metrics.update(compute_normal_error(vertices_pred, normals_gt))
    metrics['depth_mse'] = torch.mean((depth_pred - depth_gt) ** 2).item()
    metrics['normal_mse'] = torch.mean(torch.sum((normals_pred - normals_gt) ** 2, dim=1)).item()
    
    return metrics


def print_metrics_table(methods_dict, target_names=None):
    """
    Print evaluation metrics in a nice table format.
    
    Args:
        methods_dict: dict mapping method name to metrics dict
        target_names: list of metric names to include (if None, shows all)
    """
    # Get all available metric keys
    if target_names is None:
        if len(methods_dict) > 0:
            first_metrics = next(iter(methods_dict.values()))
            target_names = sorted(first_metrics.keys())
        else:
            return
    
    # Print header
    print("\n" + "=" * 120)
    print(f"{'Method':<30}", end="")
    for name in target_names:
        print(f"{name:<15}", end="")
    print()
    print("=" * 120)
    
    # Print rows
    for method_name, metrics in methods_dict.items():
        print(f"{method_name:<30}", end="")
        for name in target_names:
            value = metrics.get(name, np.nan)
            print(f"{value:<15.6f}", end="")
        print()
    
    print("=" * 120 + "\n")
