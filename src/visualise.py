"""
Visualization utilities for meshes and results.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def visualize_mesh_3d(vertices, faces, title="Mesh", save_path=None, figsize=(8, 8)):
    """
    Visualize a 3D mesh using matplotlib.
    
    Args:
        vertices: torch tensor or numpy array of shape (V, 3)
        faces: torch tensor or numpy array of shape (F, 3)
        title: title for the plot
        save_path: if provided, save to this path
        figsize: figure size
    """
    # Convert to numpy if needed
    if torch.is_tensor(vertices):
        vertices = vertices.cpu().detach().numpy()
    if torch.is_tensor(faces):
        faces = faces.cpu().detach().numpy()
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot using plot_trisurf with all vertices and faces
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                   triangles=faces, color='cyan', alpha=0.3, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                          vertices[:, 1].max() - vertices[:, 1].min(),
                          vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig


def visualize_curvature_heatmap(vertices, faces, curvature, title="Curvature Heatmap", 
                               save_path=None, figsize=(8, 8)):
    """
    Visualize curvature as a heatmap on the mesh.
    
    Args:
        vertices: torch tensor or numpy array of shape (V, 3)
        faces: torch tensor or numpy array of shape (F, 3)
        curvature: torch tensor or numpy array of shape (V,)
        title: title for the plot
        save_path: if provided, save to this path
        figsize: figure size
    """
    # Convert to numpy if needed
    if torch.is_tensor(vertices):
        vertices = vertices.cpu().detach().numpy()
    if torch.is_tensor(faces):
        faces = faces.cpu().detach().numpy()
    if torch.is_tensor(curvature):
        curvature = curvature.cpu().detach().numpy()
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize curvature for colormap
    curv_min = curvature.min()
    curv_max = curvature.max()
    if curv_max > curv_min:
        curv_normalized = (curvature - curv_min) / (curv_max - curv_min)
    else:
        curv_normalized = np.zeros_like(curvature)
    
    # Plot mesh with vertex colors
    # Create a color for each vertex based on curvature
    cmap = plt.cm.viridis
    face_colors = curv_normalized[faces]  # Shape: (F, 3) - one color per vertex per face
    
    # Use plot_trisurf with facecolors
    surface = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                              triangles=faces, alpha=0.8, edgecolor='black', 
                              linewidth=0.3, shade=False)
    
    # Manually set face colors
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=curv_min, vmax=curv_max)
    face_colorvals = curv_normalized[faces].mean(axis=1)  # Average color per face
    
    # Update colors
    face_colors_rgb = cmap(norm(curv_normalized[faces].mean(axis=1)))
    surface.set_facecolor(face_colors_rgb)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                          vertices[:, 1].max() - vertices[:, 1].min(),
                          vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=curv_min, vmax=curv_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Curvature')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved curvature heatmap to {save_path}")
    
    return fig


def plot_loss_curve(logs, save_path=None, figsize=(12, 4)):
    """
    Plot optimization loss curves.
    
    Args:
        logs: list of dicts with loss information
        save_path: if provided, save to this path
        figsize: figure size
    """
    if len(logs) == 0:
        return None
    
    iterations = [log['iteration'] for log in logs]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Total loss
    if 'total' in logs[0]:
        total_losses = [log['total'] for log in logs]
        axes[0].plot(iterations, total_losses, label='Total Loss')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Total Loss')
        axes[0].grid(True)
        axes[0].legend()
    
    # Individual losses
    if 'l_obj' in logs[0]:
        l_obj = [log['l_obj'] for log in logs]
        axes[1].plot(iterations, l_obj, label='Objective Loss')
    elif 'l_curv' in logs[0]:
        l_curv = [log['l_curv'] for log in logs]
        axes[1].plot(iterations, l_curv, label='Curvature Loss')

    if 'l_edge' in logs[0]:
        l_edge = [log['l_edge'] for log in logs]
        axes[1].plot(iterations, l_edge, label='Edge Loss')

    if 'l_pos' in logs[0]:
        l_pos = [log['l_pos'] for log in logs]
        axes[1].plot(iterations, l_pos, label='Position Loss')

    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Component Losses')
    axes[1].grid(True)
    axes[1].legend()
    axes[1].set_yscale('log')
    
    # Vertex and curvature error
    if 'vertex_error' in logs[0]:
        vertex_errors = [log['vertex_error'] for log in logs]
        axes[2].plot(iterations, vertex_errors, label='Vertex Error', linewidth=2)
        
        if 'objective_mse' in logs[0]:
            curv_mse = [log['objective_mse'] for log in logs]
            ax2 = axes[2].twinx()
            ax2.plot(iterations, curv_mse, label='Objective MSE', color='orange', linewidth=2)
            ax2.set_ylabel('Objective MSE', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            ax2.legend(loc='upper right')
        elif 'curvature_mse' in logs[0]:
            curv_mse = [log['curvature_mse'] for log in logs]
            ax2 = axes[2].twinx()
            ax2.plot(iterations, curv_mse, label='Curvature MSE', color='orange', linewidth=2)
            ax2.set_ylabel('Curvature MSE', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            ax2.legend(loc='upper right')
        
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Vertex Error', color='blue')
        axes[2].set_title('Error Metrics')
        axes[2].grid(True)
        axes[2].legend(loc='upper left')
        axes[2].tick_params(axis='y', labelcolor='blue')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss curve to {save_path}")
    
    return fig


def create_comparison_figure(vertices_dict, faces, titles, save_path=None, figsize=(16, 4)):
    """
    Create a side-by-side comparison of multiple meshes.
    
    Args:
        vertices_dict: dict mapping name to vertices
        faces: mesh faces
        titles: list of titles for each mesh
        save_path: if provided, save to this path
        figsize: figure size
    """
    num_meshes = len(vertices_dict)
    fig = plt.figure(figsize=figsize)
    
    for idx, (name, vertices) in enumerate(vertices_dict.items(), 1):
        ax = fig.add_subplot(1, num_meshes, idx, projection='3d')
        
        # Convert to numpy if needed
        if torch.is_tensor(vertices):
            vertices_np = vertices.cpu().detach().numpy()
        else:
            vertices_np = vertices
        
        if torch.is_tensor(faces):
            faces_np = faces.cpu().detach().numpy()
        else:
            faces_np = faces
        
        # Plot triangles
        ax.plot_trisurf(vertices_np[:, 0], vertices_np[:, 1], vertices_np[:, 2],
                       triangles=faces_np, color='cyan', alpha=0.3, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(titles[idx-1] if idx-1 < len(titles) else name)
        
        # Set equal aspect ratio
        max_range = np.array([vertices_np[:, 0].max() - vertices_np[:, 0].min(),
                              vertices_np[:, 1].max() - vertices_np[:, 1].min(),
                              vertices_np[:, 2].max() - vertices_np[:, 2].min()]).max() / 2.0
        mid_x = (vertices_np[:, 0].max() + vertices_np[:, 0].min()) * 0.5
        mid_y = (vertices_np[:, 1].max() + vertices_np[:, 1].min()) * 0.5
        mid_z = (vertices_np[:, 2].max() + vertices_np[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison figure to {save_path}")
    
    return fig
