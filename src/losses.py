"""
Loss functions for mesh optimization.
"""
import torch
from src.curvature import (
    compute_laplacian_curvature_proxy,
    compute_laplacian_curvature_vector,
    compute_weighted_laplacian_curvature_proxy,
    compute_weighted_laplacian_vector,
)
from src.create_mesh import compute_vertex_normals_from_positions
from src.mesh_utils import get_edge_lengths_tensor


def curvature_scalar_loss(vertices, faces, curvature_target, adjacency=None, edges=None, edge_weights=None):
    """
    Scalar curvature matching loss.
    
    L_curv = mean((H_pred - H_gt)^2)
    
    Args:
        vertices: torch tensor of shape (V, 3)
        faces: torch tensor of shape (F, 3)
        curvature_target: torch tensor of shape (V,)
        adjacency: pre-computed adjacency list
    
    Returns:
        loss: scalar tensor
    """
    if edge_weights is None:
        curvature_pred = compute_laplacian_curvature_proxy(vertices, faces, adjacency=adjacency, edges=edges)
    else:
        curvature_pred = compute_weighted_laplacian_curvature_proxy(vertices, edges, edge_weights)
    return torch.mean((curvature_pred - curvature_target) ** 2)


def curvature_vector_loss(vertices, faces, h_target, adjacency=None, edges=None, edge_weights=None):
    """
    Vector curvature matching loss.

    Match mean-curvature-normal proxy vector h ~= Delta V ~= 2Hn:
        L_curv_vec = mean(||h_pred - h_target||^2)

    Args:
        vertices: torch tensor of shape (V, 3)
        faces: torch tensor of shape (F, 3)
        h_target: torch tensor of shape (V, 3)
        adjacency: pre-computed adjacency list

    Returns:
        loss: scalar tensor
    """
    if edge_weights is None:
        h_pred = compute_laplacian_curvature_vector(vertices, faces, adjacency=adjacency, edges=edges)
    else:
        h_pred = compute_weighted_laplacian_vector(vertices, edges, edge_weights)
    return torch.mean(torch.sum((h_pred - h_target) ** 2, dim=1))


def curvature_loss(vertices,
                   faces,
                   curvature_target,
                   adjacency=None,
                   curvature_mode='scalar',
                   edges=None,
                   edge_weights=None,
                   curvature_graph_mode='mesh'):
    """
    Unified curvature loss wrapper.

    Args:
        curvature_mode: 'scalar' to match |Delta V|, 'vector' to match Delta V.
    """
    if curvature_graph_mode == 'minimum_ball':
        if curvature_mode == 'scalar':
            return curvature_scalar_loss(
                vertices,
                faces,
                curvature_target,
                adjacency=adjacency,
                edges=edges,
                edge_weights=edge_weights,
            )
        if curvature_mode == 'vector':
            return curvature_vector_loss(
                vertices,
                faces,
                curvature_target,
                adjacency=adjacency,
                edges=edges,
                edge_weights=edge_weights,
            )
    if curvature_mode == 'scalar':
        return curvature_scalar_loss(vertices, faces, curvature_target, adjacency=adjacency, edges=edges)
    if curvature_mode == 'vector':
        return curvature_vector_loss(vertices, faces, curvature_target, adjacency=adjacency, edges=edges)
    raise ValueError(f"Unsupported curvature_mode: {curvature_mode}. Use 'scalar' or 'vector'.")


def depth_loss(vertices_current, depth_target, view_direction=None):
    """
    Depth matching loss.

    For this sphere experiment, depth is the projection of each vertex onto
    a fixed view direction. By default we use the z-axis.

    Args:
        vertices_current: torch tensor of shape (V, 3)
        depth_target: torch tensor of shape (V,)
        view_direction: optional tensor-like of shape (3,)

    Returns:
        loss: scalar tensor
    """
    if view_direction is None:
        current_depth = vertices_current[:, 2]
    else:
        view_direction = torch.as_tensor(view_direction, dtype=vertices_current.dtype, device=vertices_current.device)
        view_direction = view_direction / (torch.norm(view_direction) + 1e-8)
        current_depth = torch.sum(vertices_current * view_direction, dim=1)

    return torch.mean((current_depth - depth_target) ** 2)


def normal_loss(vertices_current, faces, normal_target, adjacency=None):
    """
    Normal matching loss.

    Args:
        vertices_current: torch tensor of shape (V, 3)
        faces: torch tensor of shape (F, 3)
        normal_target: torch tensor of shape (V, 3)
        adjacency: unused, kept for signature compatibility

    Returns:
        loss: scalar tensor
    """
    normal_pred = compute_vertex_normals_from_positions(vertices_current, faces)
    return torch.mean(torch.sum((normal_pred - normal_target) ** 2, dim=1))


def edge_length_regularization(vertices_current, edges, target_lengths):
    """
    Edge length regularization loss.
    
    L_edge = mean((edge_length_current - edge_length_target)^2)
    
    Args:
        vertices_current: torch tensor of shape (V, 3)
        edges: torch tensor of shape (E, 2)
        target_lengths: torch tensor of shape (E,)
    
    Returns:
        loss: scalar tensor
    """
    if len(edges) == 0:
        return torch.tensor(0.0, device=vertices_current.device)
    
    current_lengths = torch.norm(
        vertices_current[edges[:, 0]] - vertices_current[edges[:, 1]], 
        dim=1
    )
    
    loss = torch.mean((current_lengths - target_lengths) ** 2)
    return loss


def position_regularization(vertices_current, vertices_init):
    """
    Positional regularization to prevent extreme drift.
    
    L_pos = mean(||V_current - V_init||^2)
    
    Args:
        vertices_current: torch tensor of shape (V, 3)
        vertices_init: torch tensor of shape (V, 3)
    
    Returns:
        loss: scalar tensor
    """
    loss = torch.mean(torch.norm(vertices_current - vertices_init, dim=1) ** 2)
    return loss


def center_scale_regularization(vertices_current, vertices_ref, eps=1e-8):
    """
    Global center/scale regularization to avoid drift or collapse.

    L = ||c_cur - c_ref||^2 + (s_cur - s_ref)^2
    """
    c_cur = torch.mean(vertices_current, dim=0)
    c_ref = torch.mean(vertices_ref, dim=0)

    centered_cur = vertices_current - c_cur
    centered_ref = vertices_ref - c_ref

    s_cur = torch.sqrt(torch.mean(torch.sum(centered_cur ** 2, dim=1)) + eps)
    s_ref = torch.sqrt(torch.mean(torch.sum(centered_ref ** 2, dim=1)) + eps)

    l_center = torch.sum((c_cur - c_ref) ** 2)
    l_scale = (s_cur - s_ref) ** 2
    return l_center + l_scale


def edge_length_min_loss(vertices_current, edges, min_edge_length=0.0):
    """
    Soft lower-bound penalty to prevent edges collapsing.

    L_min = mean(max(min_edge_length - L, 0)^2)
    """
    if edges.numel() == 0 or min_edge_length <= 0.0:
        return torch.tensor(0.0, device=vertices_current.device)

    lengths = torch.norm(vertices_current[edges[:, 0]] - vertices_current[edges[:, 1]], dim=1)
    shortfall = torch.clamp(min_edge_length - lengths, min=0.0)
    return torch.mean(shortfall ** 2)


def edge_length_uniformity_loss(vertices_current, edges, eps=1e-8):
    """
    Penalize highly uneven edge length distributions.

    L_var = var(L) / (mean(L)^2 + eps)
    """
    if edges.numel() == 0:
        return torch.tensor(0.0, device=vertices_current.device)

    lengths = torch.norm(vertices_current[edges[:, 0]] - vertices_current[edges[:, 1]], dim=1)
    mean_len = torch.mean(lengths)
    if mean_len <= 0.0:
        return torch.tensor(0.0, device=vertices_current.device)

    var_len = torch.mean((lengths - mean_len) ** 2)
    return var_len / (mean_len * mean_len + eps)


def edge_repulsion_hinge_loss(vertices_current, edges, min_dist=0.0, eps=1e-8):
    """
    Hinge repulsion to avoid near-collisions among neighbors.

    L_rep = mean(max(min_dist - L, 0)^2 / (min_dist^2 + eps))
    """
    if edges.numel() == 0 or min_dist <= 0.0:
        return torch.tensor(0.0, device=vertices_current.device)

    diff = vertices_current[edges[:, 0]] - vertices_current[edges[:, 1]]
    lengths = torch.sqrt(torch.sum(diff * diff, dim=1) + eps)
    shortfall = torch.clamp(min_dist - lengths, min=0.0)
    return torch.mean((shortfall / (min_dist + eps)) ** 2)


def total_loss(vertices_current, 
               vertices_init,
               faces, 
               curvature_target, 
               edges,
               target_edge_lengths,
               lambda_edge=0.1, 
               lambda_pos=0.01,
               lambda_edge_min=0.0,
               lambda_edge_var=0.0,
               lambda_repulsion=0.0,
               lambda_center_scale=0.0,
               min_edge_length=0.0,
               repulsion_min_dist=0.0,
               curvature_mode='scalar',
               curvature_graph_mode='mesh',
               curvature_edges=None,
               curvature_edge_weights=None,
               adjacency=None):
    """
    Total optimization loss.
    
    L_total = L_curv + lambda_edge * L_edge + lambda_pos * L_pos
    
    Args:
        vertices_current: torch tensor of shape (V, 3)
        vertices_init: torch tensor of shape (V, 3), initial/noisy vertices
        faces: torch tensor of shape (F, 3)
        curvature_target: torch tensor of shape (V,) for scalar mode or (V, 3) for vector mode
        edges: torch tensor of shape (E, 2)
        target_edge_lengths: torch tensor of shape (E,)
        lambda_edge: edge regularization weight
        lambda_pos: position regularization weight
        curvature_mode: 'scalar' or 'vector'
        adjacency: pre-computed adjacency list
    
    Returns:
        total_loss, loss_dict: scalar tensor and dict with individual losses
    """
    curv_edges = curvature_edges if curvature_edges is not None else edges
    l_curv = curvature_loss(
        vertices_current,
        faces,
        curvature_target,
        adjacency=adjacency,
        curvature_mode=curvature_mode,
        edges=curv_edges,
        edge_weights=curvature_edge_weights,
        curvature_graph_mode=curvature_graph_mode,
    )
    if lambda_edge == 0.0:
        l_edge = torch.tensor(0.0, device=vertices_current.device)
    else:
        l_edge = edge_length_regularization(vertices_current, edges, target_edge_lengths)

    if lambda_pos == 0.0:
        l_pos = torch.tensor(0.0, device=vertices_current.device)
    else:
        l_pos = position_regularization(vertices_current, vertices_init)

    if lambda_edge_min == 0.0 or min_edge_length <= 0.0:
        l_edge_min = torch.tensor(0.0, device=vertices_current.device)
    else:
        l_edge_min = edge_length_min_loss(vertices_current, edges, min_edge_length=min_edge_length)

    if lambda_edge_var == 0.0:
        l_edge_var = torch.tensor(0.0, device=vertices_current.device)
    else:
        l_edge_var = edge_length_uniformity_loss(vertices_current, edges)

    repulsion_dist = repulsion_min_dist if repulsion_min_dist > 0.0 else min_edge_length
    if lambda_repulsion == 0.0 or repulsion_dist <= 0.0:
        l_repulsion = torch.tensor(0.0, device=vertices_current.device)
    else:
        l_repulsion = edge_repulsion_hinge_loss(vertices_current, edges, min_dist=repulsion_dist)

    if lambda_center_scale == 0.0:
        l_center_scale = torch.tensor(0.0, device=vertices_current.device)
    else:
        l_center_scale = center_scale_regularization(vertices_current, vertices_init)
    
    total = (
        l_curv
        + lambda_edge * l_edge
        + lambda_pos * l_pos
        + lambda_edge_min * l_edge_min
        + lambda_edge_var * l_edge_var
        + lambda_repulsion * l_repulsion
        + lambda_center_scale * l_center_scale
    )
    
    loss_dict = {
        'l_curv': l_curv.item(),
        'l_edge': l_edge.item(),
        'l_pos': l_pos.item(),
        'l_edge_min': l_edge_min.item(),
        'l_edge_var': l_edge_var.item(),
        'l_repulsion': l_repulsion.item(),
        'l_center_scale': l_center_scale.item(),
        'total': total.item(),
    }
    
    return total, loss_dict


def objective_total_loss(vertices_current,
                         vertices_init,
                         faces,
                         objective_target,
                         edges,
                         target_edge_lengths,
                         objective_mode='curvature',
                         curvature_mode='scalar',
                         view_direction=None,
                         lambda_edge=0.1,
                         lambda_pos=0.01,
                         lambda_edge_min=0.0,
                         lambda_edge_var=0.0,
                         lambda_repulsion=0.0,
                         lambda_center_scale=0.0,
                         min_edge_length=0.0,
                         repulsion_min_dist=0.0,
                         curvature_graph_mode='mesh',
                         curvature_edges=None,
                         curvature_edge_weights=None,
                         adjacency=None):
    """
    Generic objective loss wrapper for curvature / depth / normal matching.
    """
    if objective_mode == 'curvature':
        curv_edges = curvature_edges if curvature_edges is not None else edges
        l_obj = curvature_loss(
            vertices_current,
            faces,
            objective_target,
            adjacency=adjacency,
            curvature_mode=curvature_mode,
            edges=curv_edges,
            edge_weights=curvature_edge_weights,
            curvature_graph_mode=curvature_graph_mode,
        )
    elif objective_mode == 'depth':
        l_obj = depth_loss(vertices_current, objective_target, view_direction=view_direction)
    elif objective_mode == 'normal':
        l_obj = normal_loss(vertices_current, faces, objective_target, adjacency=adjacency)
    else:
        raise ValueError(f"Unsupported objective_mode: {objective_mode}. Use 'curvature', 'depth', or 'normal'.")

    if lambda_edge == 0.0:
        l_edge = torch.tensor(0.0, device=vertices_current.device)
    else:
        l_edge = edge_length_regularization(vertices_current, edges, target_edge_lengths)

    if lambda_pos == 0.0:
        l_pos = torch.tensor(0.0, device=vertices_current.device)
    else:
        l_pos = position_regularization(vertices_current, vertices_init)

    if lambda_edge_min == 0.0 or min_edge_length <= 0.0:
        l_edge_min = torch.tensor(0.0, device=vertices_current.device)
    else:
        l_edge_min = edge_length_min_loss(vertices_current, edges, min_edge_length=min_edge_length)

    if lambda_edge_var == 0.0:
        l_edge_var = torch.tensor(0.0, device=vertices_current.device)
    else:
        l_edge_var = edge_length_uniformity_loss(vertices_current, edges)

    repulsion_dist = repulsion_min_dist if repulsion_min_dist > 0.0 else min_edge_length
    if lambda_repulsion == 0.0 or repulsion_dist <= 0.0:
        l_repulsion = torch.tensor(0.0, device=vertices_current.device)
    else:
        l_repulsion = edge_repulsion_hinge_loss(vertices_current, edges, min_dist=repulsion_dist)

    if lambda_center_scale == 0.0:
        l_center_scale = torch.tensor(0.0, device=vertices_current.device)
    else:
        l_center_scale = center_scale_regularization(vertices_current, vertices_init)

    total = (
        l_obj
        + lambda_edge * l_edge
        + lambda_pos * l_pos
        + lambda_edge_min * l_edge_min
        + lambda_edge_var * l_edge_var
        + lambda_repulsion * l_repulsion
        + lambda_center_scale * l_center_scale
    )

    loss_dict = {
        'l_obj': l_obj.item(),
        'l_edge': l_edge.item(),
        'l_pos': l_pos.item(),
        'l_edge_min': l_edge_min.item(),
        'l_edge_var': l_edge_var.item(),
        'l_repulsion': l_repulsion.item(),
        'l_center_scale': l_center_scale.item(),
        'total': total.item(),
    }

    return total, loss_dict


def laplacian_smoothing_step(vertices, faces, alpha=0.1, adjacency=None):
    """
    Perform one step of Laplacian smoothing.
    
    V_i <- V_i + alpha * (mean(V_j) - V_i)
    
    Args:
        vertices: torch tensor of shape (V, 3)
        faces: torch tensor of shape (F, 3)
        alpha: smoothing parameter
        adjacency: pre-computed adjacency list
    
    Returns:
        vertices_smoothed: torch tensor of shape (V, 3)
    """
    num_verts = vertices.shape[0]
    device = vertices.device
    
    if adjacency is None:
        from src.mesh_utils import compute_adjacency_list
        adjacency = compute_adjacency_list(faces, num_verts)
    
    updated_positions = []

    for i in range(num_verts):
        neighbors = adjacency[i]
        if len(neighbors) > 0:
            neighbor_positions = vertices[neighbors]
            mean_neighbor_pos = torch.mean(neighbor_positions, dim=0)
            updated_positions.append(vertices[i] + alpha * (mean_neighbor_pos - vertices[i]))
        else:
            updated_positions.append(vertices[i])
    
    return torch.stack(updated_positions, dim=0)
