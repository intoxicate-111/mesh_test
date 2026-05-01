"""
Loss functions for mesh optimization.
"""
import torch
from src.curvature import compute_laplacian_curvature_proxy, compute_laplacian_curvature_vector
from src.create_mesh import compute_vertex_normals_from_positions
from src.mesh_utils import get_edge_lengths_tensor


def curvature_scalar_loss(vertices, faces, curvature_target, adjacency=None, edges=None):
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
    curvature_pred = compute_laplacian_curvature_proxy(vertices, faces, adjacency=adjacency, edges=edges)
    return torch.mean((curvature_pred - curvature_target) ** 2)


def curvature_vector_loss(vertices, faces, h_target, adjacency=None, edges=None):
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
    h_pred = compute_laplacian_curvature_vector(vertices, faces, adjacency=adjacency, edges=edges)
    return torch.mean(torch.sum((h_pred - h_target) ** 2, dim=1))


def curvature_loss(vertices, faces, curvature_target, adjacency=None, curvature_mode='scalar', edges=None):
    """
    Unified curvature loss wrapper.

    Args:
        curvature_mode: 'scalar' to match |Delta V|, 'vector' to match Delta V.
    """
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


def total_loss(vertices_current, 
               vertices_init,
               faces, 
               curvature_target, 
               edges,
               target_edge_lengths,
               lambda_edge=0.1, 
               lambda_pos=0.01,
               curvature_mode='scalar',
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
    l_curv = curvature_loss(
        vertices_current,
        faces,
        curvature_target,
        adjacency=adjacency,
        curvature_mode=curvature_mode,
        edges=edges,
    )
    l_edge = edge_length_regularization(vertices_current, edges, target_edge_lengths)
    l_pos = position_regularization(vertices_current, vertices_init)
    
    total = l_curv + lambda_edge * l_edge + lambda_pos * l_pos
    
    loss_dict = {
        'l_curv': l_curv.item(),
        'l_edge': l_edge.item(),
        'l_pos': l_pos.item(),
        'total': total.item()
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
                         adjacency=None):
    """
    Generic objective loss wrapper for curvature / depth / normal matching.
    """
    if objective_mode == 'curvature':
        l_obj = curvature_loss(
            vertices_current,
            faces,
            objective_target,
            adjacency=adjacency,
            curvature_mode=curvature_mode,
            edges=edges,
        )
    elif objective_mode == 'depth':
        l_obj = depth_loss(vertices_current, objective_target, view_direction=view_direction)
    elif objective_mode == 'normal':
        l_obj = normal_loss(vertices_current, faces, objective_target, adjacency=adjacency)
    else:
        raise ValueError(f"Unsupported objective_mode: {objective_mode}. Use 'curvature', 'depth', or 'normal'.")

    l_edge = edge_length_regularization(vertices_current, edges, target_edge_lengths)
    l_pos = position_regularization(vertices_current, vertices_init)

    total = l_obj + lambda_edge * l_edge + lambda_pos * l_pos

    loss_dict = {
        'l_obj': l_obj.item(),
        'l_edge': l_edge.item(),
        'l_pos': l_pos.item(),
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
