"""
Curvature approximation using Laplacian-based proxy.
"""
import torch
from src.mesh_utils import compute_adjacency_list


def compute_laplacian_curvature_proxy(vertices, faces, adjacency=None):
    """
    Compute Laplacian-based curvature proxy for each vertex.
    
    For each vertex i:
        L_i = V_i - mean(V_j), where j belongs to neighbors(i)
    Then curvature magnitude proxy:
        H_i = ||L_i||
    
    Args:
        vertices: torch tensor of shape (V, 3)
        faces: torch tensor of shape (F, 3)
        adjacency: pre-computed adjacency list (if None, will be computed)
    
    Returns:
        curvature: torch tensor of shape (V,), curvature magnitude for each vertex
    """
    num_verts = vertices.shape[0]
    device = vertices.device
    
    if adjacency is None:
        adjacency = compute_adjacency_list(faces, num_verts)
    
    curvature = torch.zeros(num_verts, device=device)
    
    for i in range(num_verts):
        neighbors = adjacency[i]
        if len(neighbors) > 0:
            # Compute mean position of neighbors
            neighbor_positions = vertices[neighbors]
            mean_neighbor_pos = torch.mean(neighbor_positions, dim=0)
            
            # Compute Laplacian vector
            laplacian = vertices[i] - mean_neighbor_pos
            
            # Compute magnitude
            curvature[i] = torch.norm(laplacian)
        else:
            curvature[i] = 0.0
    
    return curvature


def compute_laplacian_curvature_vector(vertices, faces, adjacency=None):
    """
    Compute Laplacian curvature vectors for each vertex.

    This vector is a discrete mean-curvature-normal proxy:
        h_i = V_i - mean(V_j), j in neighbors(i)
    and can be interpreted as an approximation of 2Hn.
    
    Args:
        vertices: torch tensor of shape (V, 3)
        faces: torch tensor of shape (F, 3)
        adjacency: pre-computed adjacency list
    
    Returns:
        laplacian: torch tensor of shape (V, 3)
    """
    num_verts = vertices.shape[0]
    device = vertices.device
    
    if adjacency is None:
        adjacency = compute_adjacency_list(faces, num_verts)
    
    laplacian = torch.zeros(num_verts, 3, device=device)
    
    for i in range(num_verts):
        neighbors = adjacency[i]
        if len(neighbors) > 0:
            neighbor_positions = vertices[neighbors]
            mean_neighbor_pos = torch.mean(neighbor_positions, dim=0)
            laplacian[i] = vertices[i] - mean_neighbor_pos
    
    return laplacian


def compute_laplacian_vector(vertices, faces, adjacency=None):
    """
    Backward-compatible alias for compute_laplacian_curvature_vector.
    """
    return compute_laplacian_curvature_vector(vertices, faces, adjacency)


def compute_mean_curvature_normal(vertices, faces, adjacency=None):
    """
    Compute mean curvature normal vector for each vertex.
    
    This is the Laplacian vector scaled by local mesh properties.
    
    Args:
        vertices: torch tensor of shape (V, 3)
        faces: torch tensor of shape (F, 3)
        adjacency: pre-computed adjacency list
    
    Returns:
        mean_curvature_normal: torch tensor of shape (V, 3)
    """
    laplacian = compute_laplacian_curvature_vector(vertices, faces, adjacency)
    return laplacian


def curvature_mse_loss(curvature_pred, curvature_target):
    """
    Mean squared error loss between predicted and target curvature.
    
    Args:
        curvature_pred: torch tensor of shape (V,)
        curvature_target: torch tensor of shape (V,)
    
    Returns:
        loss: scalar tensor
    """
    return torch.mean((curvature_pred - curvature_target) ** 2)
