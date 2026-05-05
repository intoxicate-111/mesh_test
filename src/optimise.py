"""
Mesh optimization using gradient-based methods.
"""
from collections import deque
import torch
import torch.optim as optim
from src.losses import total_loss, objective_total_loss, laplacian_smoothing_step
from src.mesh_utils import compute_adjacency_list, get_edge_lengths_tensor, build_dynamic_graph
from src.curvature import (
    compute_laplacian_curvature_proxy,
    compute_laplacian_curvature_vector,
    compute_weighted_laplacian_curvature_proxy,
    compute_weighted_laplacian_vector,
)
from src.minimum_ball import build_minimum_ball_graph
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
    lambda_edge_min=0.0,
    lambda_edge_var=0.0,
    lambda_repulsion=0.0,
    lambda_center_scale=0.0,
    min_edge_length=0.0,
    repulsion_min_dist=0.0,
    curvature_graph_mode='mesh',
    min_ball_k=16,
    min_ball_alpha=10.0,
    min_ball_max_triangles=32,
    min_ball_tri_chunk_size=2048,
    connectivity_mode='faces',
    graph_mode='knn',
    graph_k=16,
    graph_radius=0.1,
    graph_chunk_size=2048,
    graph_update_interval=20,
    edge_target_loss_in_dynamic=False,
    enable_dynamic_schedule=True,
    plateau_patience=1000,
    plateau_min_delta=1e-5,
    decay_factor=0.5,
    lambda_zero_threshold=1e-4,
    schedule_delay_iters=0.4,
    osc_window=500,
    osc_threshold_scale=10.0,
    lr_cooldown=2000,
    min_lr_scale=0.01,
    device='cpu',
    verbose=True,
):
    """
    Generic mesh optimization loop for curvature/depth/normal objectives.
    """
    vertices_opt = vertices_noisy.clone().to(device)
    vertices_opt.requires_grad = True

    num_verts = vertices_noisy.shape[0]
    adjacency = None
    edges = torch.empty((0, 2), dtype=torch.long, device=vertices_opt.device)
    target_edge_lengths = torch.empty((0,), dtype=vertices_opt.dtype, device=vertices_opt.device)
    curv_edges = None
    curv_edge_weights = None
    curv_diag = {}

    use_edge_target_loss = (lambda_edge != 0.0) and (
        connectivity_mode == 'faces' or edge_target_loss_in_dynamic
    )

    if connectivity_mode == 'faces':
        adjacency = compute_adjacency_list(faces, num_verts)
        edges, target_edge_lengths = get_edge_lengths_tensor(vertices_noisy, faces)
    elif connectivity_mode == 'dynamic':
        graph_update_interval = max(int(graph_update_interval), 1)
        with torch.no_grad():
            edges = build_dynamic_graph(
                vertices_noisy,
                mode=graph_mode,
                k=graph_k,
                radius=graph_radius,
                chunk_size=graph_chunk_size,
            )
        if use_edge_target_loss and edges.numel() > 0:
            target_edge_lengths = torch.norm(vertices_noisy[edges[:, 0]] - vertices_noisy[edges[:, 1]], dim=1)
    else:
        raise ValueError(f"Unsupported connectivity_mode: {connectivity_mode}. Use 'faces' or 'dynamic'.")

    if curvature_graph_mode == 'mesh':
        curv_edges = None
    elif curvature_graph_mode == 'knn':
        curv_edges = edges
        curv_edge_weights = None
    elif curvature_graph_mode == 'minimum_ball':
        with torch.no_grad():
            curv_edges, curv_edge_weights, curv_diag = build_minimum_ball_graph(
                vertices_noisy,
                k=min_ball_k,
                alpha_min=min_ball_alpha,
                max_triangles_per_vertex=min_ball_max_triangles,
                knn_chunk_size=graph_chunk_size,
                tri_chunk_size=min_ball_tri_chunk_size,
                collapse_threshold=min_edge_length,
            )
    else:
        raise ValueError(
            f"Unsupported curvature_graph_mode: {curvature_graph_mode}. "
            "Use 'mesh', 'knn', or 'minimum_ball'."
        )

    optimizer = optim.Adam([vertices_opt], lr=learning_rate)
    logs = []
    best_loss = None
    epochs_since_improvement = 0
    epochs_since_decrease = 0
    prev_loss = None
    min_learning_rate = learning_rate * min_lr_scale
    current_lambda_edge = lambda_edge
    current_lambda_pos = lambda_pos

    if 0.0 < schedule_delay_iters < 1.0:
        schedule_delay_iters = int(num_iterations * schedule_delay_iters)
    schedule_delay_iters = max(int(schedule_delay_iters), 0)
    osc_window = max(int(osc_window), 1)
    lr_cooldown = max(int(lr_cooldown), 0)
    loss_window = deque(maxlen=osc_window)
    last_lr_update_iter = -lr_cooldown

    for iteration in range(num_iterations):
        current_lr = optimizer.param_groups[0]['lr']

        optimizer.zero_grad()

        if connectivity_mode == 'dynamic' and (iteration % graph_update_interval == 0):
            with torch.no_grad():
                edges = build_dynamic_graph(
                    vertices_opt,
                    mode=graph_mode,
                    k=graph_k,
                    radius=graph_radius,
                    chunk_size=graph_chunk_size,
                )
            if use_edge_target_loss and edges.numel() > 0:
                target_edge_lengths = torch.norm(vertices_noisy[edges[:, 0]] - vertices_noisy[edges[:, 1]], dim=1)

        if curvature_graph_mode == 'knn':
            curv_edges = edges
            curv_edge_weights = None
        elif curvature_graph_mode == 'minimum_ball' and (iteration % graph_update_interval == 0):
            with torch.no_grad():
                curv_edges, curv_edge_weights, curv_diag = build_minimum_ball_graph(
                    vertices_opt,
                    k=min_ball_k,
                    alpha_min=min_ball_alpha,
                    max_triangles_per_vertex=min_ball_max_triangles,
                    knn_chunk_size=graph_chunk_size,
                    tri_chunk_size=min_ball_tri_chunk_size,
                    collapse_threshold=min_edge_length,
                )

        effective_lambda_edge = current_lambda_edge if use_edge_target_loss else 0.0

        reg_edges = curv_edges if (curvature_graph_mode == 'minimum_ball' and not use_edge_target_loss) else edges

        loss, loss_dict = objective_total_loss(
            vertices_opt,
            vertices_noisy,
            faces,
            objective_target,
            reg_edges,
            target_edge_lengths,
            objective_mode=objective_mode,
            curvature_mode=curvature_mode,
            view_direction=view_direction,
            lambda_edge=effective_lambda_edge,
            lambda_pos=current_lambda_pos,
            lambda_edge_min=lambda_edge_min,
            lambda_edge_var=lambda_edge_var,
            lambda_repulsion=lambda_repulsion,
            lambda_center_scale=lambda_center_scale,
            min_edge_length=min_edge_length,
            repulsion_min_dist=repulsion_min_dist,
            curvature_graph_mode=curvature_graph_mode,
            curvature_edges=curv_edges,
            curvature_edge_weights=curv_edge_weights,
            adjacency=adjacency,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_([vertices_opt], max_norm=0.1)
        optimizer.step()

        loss_value = loss.item()
        loss_window.append(loss_value)

        if prev_loss is None:
            prev_loss = loss_value
        else:
            if loss_value < prev_loss:
                epochs_since_decrease = 0
            else:
                epochs_since_decrease += 1
            prev_loss = loss_value

        if best_loss is None or loss_value < (best_loss - plateau_min_delta):
            best_loss = loss_value
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        loss_window_amp = None
        if len(loss_window) == osc_window:
            loss_window_amp = max(loss_window) - min(loss_window)

        in_cooldown = (iteration - last_lr_update_iter) < lr_cooldown
        should_decay_plateau = (
            epochs_since_improvement >= plateau_patience
            and epochs_since_decrease >= plateau_patience
        )
        should_decay_osc = (
            loss_window_amp is not None
            and loss_window_amp <= (osc_threshold_scale * current_lr)
        )

        if (
            enable_dynamic_schedule
            and iteration >= schedule_delay_iters
            and not in_cooldown
            and (should_decay_plateau or should_decay_osc)
        ):
            current_lr = max(current_lr * decay_factor, min_learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            current_lambda_edge = max(current_lambda_edge * decay_factor, 0.0)
            current_lambda_pos = max(current_lambda_pos * decay_factor, 0.0)

            # Aggressive test: zero out small regularizers
            if current_lambda_edge <= lambda_zero_threshold:
                current_lambda_edge = 0.0
            if current_lambda_pos <= lambda_zero_threshold:
                current_lambda_pos = 0.0

            epochs_since_improvement = 0
            epochs_since_decrease = 0
            last_lr_update_iter = iteration

        vertex_error = torch.mean(torch.norm(vertices_opt - vertices_gt, dim=1)).item()

        if objective_mode == 'curvature':
            if curvature_mode == 'scalar':
                if curvature_graph_mode == 'minimum_ball':
                    objective_current = compute_weighted_laplacian_curvature_proxy(
                        vertices_opt,
                        curv_edges,
                        curv_edge_weights,
                    )
                else:
                    objective_current = compute_laplacian_curvature_proxy(
                        vertices_opt,
                        faces,
                        adjacency=adjacency,
                        edges=curv_edges,
                    )
                objective_mse = torch.mean((objective_current - objective_target) ** 2).item()
            else:
                if curvature_graph_mode == 'minimum_ball':
                    objective_current = compute_weighted_laplacian_vector(
                        vertices_opt,
                        curv_edges,
                        curv_edge_weights,
                    )
                else:
                    objective_current = compute_laplacian_curvature_vector(
                        vertices_opt,
                        faces,
                        adjacency=adjacency,
                        edges=curv_edges,
                    )
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
            'lambda_edge_current': effective_lambda_edge,
            'lambda_pos_current': current_lambda_pos,
            'best_loss': best_loss,
            'epochs_since_improvement': epochs_since_improvement,
            'epochs_since_decrease': epochs_since_decrease,
            'vertex_error': vertex_error,
            'objective_mse': objective_mse,
        }
        if loss_window_amp is not None:
            log_dict['loss_window_amp'] = loss_window_amp
        if curv_diag:
            log_dict.update(curv_diag)
        logs.append(log_dict)

        if verbose and (iteration + 1) % 50 == 0:
            print(
                f"Iter {iteration+1}/{num_iterations} | Loss: {loss.item():.6f} | "
                f"Obj: {loss_dict['l_obj']:.6f} | VError: {vertex_error:.6f} | "
                f"ObjMSE({objective_mode}): {objective_mse:.6f} | lr: {current_lr:.6e} | "
                f"lam_e: {effective_lambda_edge:.4e} | lam_p: {current_lambda_pos:.4e}"
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
    lambda_edge_min=0.0,
    lambda_edge_var=0.0,
    lambda_repulsion=0.0,
    lambda_center_scale=0.0,
    min_edge_length=0.0,
    repulsion_min_dist=0.0,
    curvature_graph_mode='mesh',
    min_ball_k=16,
    min_ball_alpha=10.0,
    min_ball_max_triangles=32,
    min_ball_tri_chunk_size=2048,
    connectivity_mode='faces',
    graph_mode='knn',
    graph_k=16,
    graph_radius=0.1,
    graph_chunk_size=2048,
    graph_update_interval=20,
    edge_target_loss_in_dynamic=False,
    enable_dynamic_schedule=True,
    plateau_patience=50,
    plateau_min_delta=1e-5,
    decay_factor=0.5,
    lambda_zero_threshold=1e-4,
    schedule_delay_iters=2000,
    osc_window=500,
    osc_threshold_scale=10.0,
    lr_cooldown=100,
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
        lambda_edge_min=lambda_edge_min,
        lambda_edge_var=lambda_edge_var,
        lambda_repulsion=lambda_repulsion,
        lambda_center_scale=lambda_center_scale,
        min_edge_length=min_edge_length,
        repulsion_min_dist=repulsion_min_dist,
        curvature_graph_mode=curvature_graph_mode,
        min_ball_k=min_ball_k,
        min_ball_alpha=min_ball_alpha,
        min_ball_max_triangles=min_ball_max_triangles,
        min_ball_tri_chunk_size=min_ball_tri_chunk_size,
        connectivity_mode=connectivity_mode,
        graph_mode=graph_mode,
        graph_k=graph_k,
        graph_radius=graph_radius,
        graph_chunk_size=graph_chunk_size,
        graph_update_interval=graph_update_interval,
        edge_target_loss_in_dynamic=edge_target_loss_in_dynamic,
        enable_dynamic_schedule=enable_dynamic_schedule,
        plateau_patience=plateau_patience,
        plateau_min_delta=plateau_min_delta,
        decay_factor=decay_factor,
        lambda_zero_threshold=lambda_zero_threshold,
        schedule_delay_iters=schedule_delay_iters,
        osc_window=osc_window,
        osc_threshold_scale=osc_threshold_scale,
        lr_cooldown=lr_cooldown,
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
        curvature_gt = compute_laplacian_curvature_proxy(vertices_gt, faces, adjacency=adjacency)
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
