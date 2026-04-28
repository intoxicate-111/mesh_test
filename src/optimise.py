"""
Mesh optimization using gradient-based methods.
"""
import torch
import torch.optim as optim
from src.losses import total_loss, objective_total_loss, laplacian_smoothing_step
from src.mesh_utils import compute_adjacency_list, get_edge_lengths_tensor
from src.curvature import compute_laplacian_curvature_proxy, compute_laplacian_curvature_vector
from src.create_mesh import compute_vertex_normals_from_positions


def optimize_mesh_objective(
    vertices_noisy,
    vertices_gt,
    faces,
    objective_target,
    objective_mode='curvature',
    curvature_mode='scalar',
    view_direction=None,
    num_iterations=500,
    learning_rate=0.01,
    lambda_edge=0.1,
    lambda_pos=0.01,
    enable_dynamic_schedule=True,
    lambda_decay_start=0.7,
    lr_decay_start=0.7,
    min_lr_scale=0.1,
    device='cpu',
    verbose=True,
):
    """
    Generic mesh optimization loop for curvature/depth/normal objectives.
    """
    vertices_opt = vertices_noisy.clone().to(device)
    vertices_opt.requires_grad = True

    num_verts = vertices_noisy.shape[0]
    adjacency = compute_adjacency_list(faces, num_verts)
    edges, target_edge_lengths = get_edge_lengths_tensor(vertices_noisy, faces)

    optimizer = optim.Adam([vertices_opt], lr=learning_rate)
    logs = []

    for iteration in range(num_iterations):
        progress = iteration / max(1, num_iterations - 1)

        current_lambda_edge = lambda_edge
        current_lambda_pos = lambda_pos
        current_lr = learning_rate

        if enable_dynamic_schedule:
            if progress > lambda_decay_start:
                t_lambda = (progress - lambda_decay_start) / max(1e-8, 1.0 - lambda_decay_start)
                lambda_scale = max(0.0, 1.0 - t_lambda)
                current_lambda_edge = lambda_edge * lambda_scale
                current_lambda_pos = lambda_pos * lambda_scale

            if progress > lr_decay_start:
                t_lr = (progress - lr_decay_start) / max(1e-8, 1.0 - lr_decay_start)
                lr_scale = 1.0 - (1.0 - min_lr_scale) * t_lr
                current_lr = learning_rate * max(min_lr_scale, lr_scale)

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        optimizer.zero_grad()

        loss, loss_dict = objective_total_loss(
            vertices_opt,
            vertices_noisy,
            faces,
            objective_target,
            edges,
            target_edge_lengths,
            objective_mode=objective_mode,
            curvature_mode=curvature_mode,
            view_direction=view_direction,
            lambda_edge=current_lambda_edge,
            lambda_pos=current_lambda_pos,
            adjacency=adjacency,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_([vertices_opt], max_norm=0.1)
        optimizer.step()

        vertex_error = torch.mean(torch.norm(vertices_opt - vertices_gt, dim=1)).item()

        if objective_mode == 'curvature':
            if curvature_mode == 'scalar':
                objective_current = compute_laplacian_curvature_proxy(vertices_opt, faces, adjacency)
                objective_mse = torch.mean((objective_current - objective_target) ** 2).item()
            else:
                objective_current = compute_laplacian_curvature_vector(vertices_opt, faces, adjacency)
                objective_mse = torch.mean(torch.sum((objective_current - objective_target) ** 2, dim=1)).item()
        elif objective_mode == 'depth':
            if view_direction is None:
                objective_current = vertices_opt[:, 2]
            else:
                view_direction_tensor = torch.as_tensor(view_direction, dtype=vertices_opt.dtype, device=vertices_opt.device)
                view_direction_tensor = view_direction_tensor / (torch.norm(view_direction_tensor) + 1e-8)
                objective_current = torch.sum(vertices_opt * view_direction_tensor, dim=1)
            objective_mse = torch.mean((objective_current - objective_target) ** 2).item()
        elif objective_mode == 'normal':
            objective_current = compute_vertex_normals_from_positions(vertices_opt, faces)
            objective_mse = torch.mean(torch.sum((objective_current - objective_target) ** 2, dim=1)).item()
        else:
            raise ValueError(f"Unsupported objective_mode: {objective_mode}")

        log_dict = {
            'iteration': iteration,
            **loss_dict,
            'lr': current_lr,
            'lambda_edge_current': current_lambda_edge,
            'lambda_pos_current': current_lambda_pos,
            'vertex_error': vertex_error,
            'objective_mse': objective_mse,
        }
        logs.append(log_dict)

        if verbose and (iteration + 1) % 50 == 0:
            print(
                f"Iter {iteration+1}/{num_iterations} | Loss: {loss.item():.6f} | "
                f"Obj: {loss_dict['l_obj']:.6f} | VError: {vertex_error:.6f} | "
                f"ObjMSE({objective_mode}): {objective_mse:.6f} | lr: {current_lr:.6e} | "
                f"lam_e: {current_lambda_edge:.4e} | lam_p: {current_lambda_pos:.4e}"
            )

    return vertices_opt.detach().clone(), logs


def optimize_mesh_curvature(
    vertices_noisy,
    vertices_gt,
    faces,
    curvature_target,
    curvature_mode='scalar',
    num_iterations=500,
    learning_rate=0.01,
    lambda_edge=0.1,
    lambda_pos=0.01,
    enable_dynamic_schedule=True,
    lambda_decay_start=0.7,
    lr_decay_start=0.7,
    min_lr_scale=0.1,
    device='cpu',
    verbose=True
):
    """
    Optimize mesh vertices to match target curvature using gradient descent.
    
    Args:
        vertices_noisy: torch tensor of shape (V, 3), initial noisy vertices
        vertices_gt: torch tensor of shape (V, 3), ground truth vertices (for reference)
        faces: torch tensor of shape (F, 3)
        curvature_target: torch tensor of shape (V,) for scalar mode or (V, 3) for vector mode
        curvature_mode: 'scalar' or 'vector'
        num_iterations: number of optimization iterations
        learning_rate: learning rate for optimizer
        lambda_edge: edge regularization weight
        lambda_pos: position regularization weight
        enable_dynamic_schedule: whether to decay lambda/lr in late iterations
        lambda_decay_start: progress ratio to start lambda decay (0-1)
        lr_decay_start: progress ratio to start lr decay (0-1)
        min_lr_scale: minimum lr scale at final iteration
        device: torch device
        verbose: whether to print progress
    
    Returns:
        vertices_optimized: torch tensor of shape (V, 3)
        logs: list of dicts with loss history
    """
    return optimize_mesh_objective(
        vertices_noisy,
        vertices_gt,
        faces,
        curvature_target,
        objective_mode='curvature',
        curvature_mode=curvature_mode,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        lambda_edge=lambda_edge,
        lambda_pos=lambda_pos,
        enable_dynamic_schedule=enable_dynamic_schedule,
        lambda_decay_start=lambda_decay_start,
        lr_decay_start=lr_decay_start,
        min_lr_scale=min_lr_scale,
        device=device,
        verbose=verbose,
    )


def laplacian_smoothing_baseline(
    vertices_noisy,
    vertices_gt,
    faces,
    num_iterations=500,
    alpha=0.1,
    device='cpu',
    verbose=True
):
    """
    Laplacian smoothing baseline for comparison.
    
    Args:
        vertices_noisy: torch tensor of shape (V, 3)
        vertices_gt: torch tensor of shape (V, 3), ground truth vertices
        faces: torch tensor of shape (F, 3)
        num_iterations: number of smoothing iterations
        alpha: smoothing parameter
        device: torch device
        verbose: whether to print progress
    
    Returns:
        vertices_smoothed: torch tensor of shape (V, 3)
        logs: list of dicts with metrics history
    """
    vertices_current = vertices_noisy.clone().to(device)
    
    num_verts = vertices_noisy.shape[0]
    adjacency = compute_adjacency_list(faces, num_verts)
    
    logs = []
    
    for iteration in range(num_iterations):
        # Apply Laplacian smoothing step
        vertices_current = laplacian_smoothing_step(vertices_current, faces, alpha, adjacency)
        
        # Compute metrics
        vertex_error = torch.mean(torch.norm(vertices_current - vertices_gt, dim=1)).item()
        curvature_current = compute_laplacian_curvature_proxy(vertices_current, faces, adjacency)
        curvature_gt = compute_laplacian_curvature_proxy(vertices_gt, faces, adjacency)
        curvature_mse = torch.mean((curvature_current - curvature_gt) ** 2).item()
        
        log_dict = {
            'iteration': iteration,
            'vertex_error': vertex_error,
            'curvature_mse': curvature_mse
        }
        logs.append(log_dict)
        
        if verbose and (iteration + 1) % 50 == 0:
            print(f"Iter {iteration+1}/{num_iterations} | "
                  f"VError: {vertex_error:.6f} | CurvMSE: {curvature_mse:.6f}")
    
    return vertices_current, logs
