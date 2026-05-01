"""
Curvature approximation using Laplacian-based proxy.
"""
import torch
from src.mesh_utils import compute_adjacency_list


def _prepare_edge_index(faces, adjacency=None, edges=None, device=None):
    """Build an undirected edge list once for vectorized neighborhood aggregation."""
    if edges is not None:
        edge_index = edges
        if not torch.is_tensor(edge_index):
            edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=device)
        else:
            edge_index = edge_index.to(device=device, dtype=torch.long)

        if edge_index.numel() == 0:
            return edge_index.reshape(0, 2)
        if edge_index.dim() != 2 or edge_index.size(1) != 2:
            raise ValueError("edges must have shape (E, 2).")
        return edge_index

    if adjacency is not None:
        edge_list = []
        for src, neighbors in enumerate(adjacency):
            for dst in neighbors:
                if src < dst:
                    edge_list.append((src, dst))

        if not edge_list:
            return torch.empty((0, 2), dtype=torch.long, device=device)
        return torch.tensor(edge_list, dtype=torch.long, device=device)

    face_edges = torch.cat(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        dim=0,
    )
    face_edges = torch.sort(face_edges, dim=1).values
    return torch.unique(face_edges, dim=0)


def _neighbor_statistics(vertices, faces, adjacency=None, edges=None):
    edge_index = _prepare_edge_index(faces, adjacency=adjacency, edges=edges, device=vertices.device)

    num_verts = vertices.shape[0]
    neighbor_sum = torch.zeros_like(vertices)
    neighbor_count = torch.zeros(num_verts, dtype=vertices.dtype, device=vertices.device)

    if edge_index.numel() == 0:
        return neighbor_sum, neighbor_count

    src = edge_index[:, 0]
    dst = edge_index[:, 1]

    neighbor_sum.scatter_add_(0, src.unsqueeze(1).expand(-1, 3), vertices[dst])
    neighbor_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, 3), vertices[src])

    ones = torch.ones(src.shape[0], dtype=vertices.dtype, device=vertices.device)
    neighbor_count.scatter_add_(0, src, ones)
    neighbor_count.scatter_add_(0, dst, ones)

    return neighbor_sum, neighbor_count


def compute_laplacian_curvature_proxy(vertices, faces, adjacency=None, edges=None):
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
    neighbor_sum, neighbor_count = _neighbor_statistics(vertices, faces, adjacency=adjacency, edges=edges)
    valid = neighbor_count > 0
    mean_neighbor_pos = neighbor_sum / neighbor_count.clamp_min(1.0).unsqueeze(1)
    laplacian = vertices - mean_neighbor_pos
    laplacian = laplacian * valid.unsqueeze(1).to(vertices.dtype)
    return torch.norm(laplacian, dim=1)


def compute_laplacian_curvature_vector(vertices, faces, adjacency=None, edges=None):
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
    neighbor_sum, neighbor_count = _neighbor_statistics(vertices, faces, adjacency=adjacency, edges=edges)
    valid = neighbor_count > 0
    mean_neighbor_pos = neighbor_sum / neighbor_count.clamp_min(1.0).unsqueeze(1)
    laplacian = vertices - mean_neighbor_pos
    return laplacian * valid.unsqueeze(1).to(vertices.dtype)


def compute_laplacian_vector(vertices, faces, adjacency=None, edges=None):
    """
    Backward-compatible alias for compute_laplacian_curvature_vector.
    """
    return compute_laplacian_curvature_vector(vertices, faces, adjacency=adjacency, edges=edges)


def compute_mean_curvature_normal(vertices, faces, adjacency=None, edges=None):
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
    laplacian = compute_laplacian_curvature_vector(vertices, faces, adjacency=adjacency, edges=edges)
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
