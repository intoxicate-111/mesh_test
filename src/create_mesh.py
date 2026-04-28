"""
Mesh creation utilities for generating sphere meshes.
"""
import numpy as np
import trimesh
import torch


def create_icosphere(subdivisions=3, radius=1.0, device='cpu'):
    """
    Create an icosphere mesh using trimesh.
    
    Args:
        subdivisions: Number of subdivision levels (3-4 recommended)
        radius: Radius of the sphere
        device: torch device ('cpu' or 'cuda')
    
    Returns:
        vertices: torch tensor of shape (V, 3)
        faces: torch tensor of shape (F, 3)
        trimesh_obj: trimesh object for reference
    """
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    
    vertices = torch.from_numpy(mesh.vertices).float().to(device)
    faces = torch.from_numpy(mesh.faces).long().to(device)
    
    return vertices, faces, mesh


def add_radial_noise(vertices_clean, normals, epsilon=0.05, seed=None, noise_mode='mixed'):
    """
    Add synthetic geometric noise to mesh vertices.
    
    Args:
        vertices_clean: torch tensor of shape (V, 3)
        normals: torch tensor of shape (V, 3), vertex normals
        epsilon: noise strength
        seed: random seed for reproducibility
        noise_mode: 'radial', 'tangential', or 'mixed'
    
    Returns:
        vertices_noisy: torch tensor of shape (V, 3)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    num_verts = vertices_clean.shape[0]
    device = vertices_clean.device
    
    normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-8)

    # Random noise per vertex
    radial_noise = torch.randn(num_verts, 1, device=device)

    # Tangential noise: sample a random vector, then project out the normal component.
    random_vec = torch.randn(num_verts, 3, device=device)
    tangential_vec = random_vec - torch.sum(random_vec * normals, dim=1, keepdim=True) * normals
    tangential_norm = torch.norm(tangential_vec, dim=1, keepdim=True)
    tangential_vec = tangential_vec / (tangential_norm + 1e-8)
    tangential_noise = torch.randn(num_verts, 1, device=device)

    if noise_mode == 'radial':
        displacement = normals * radial_noise
    elif noise_mode == 'tangential':
        displacement = tangential_vec * tangential_noise
    elif noise_mode == 'mixed':
        displacement = 0.5 * normals * radial_noise + 0.5 * tangential_vec * tangential_noise
    else:
        raise ValueError(f"Unsupported noise_mode: {noise_mode}. Use 'radial', 'tangential', or 'mixed'.")

    vertices_noisy = vertices_clean + epsilon * displacement
    
    return vertices_noisy


def compute_vertex_normals_from_positions(vertices, faces, normalize=True):
    """
    Compute vertex normals for a sphere by normalizing vertex positions.
    
    For a perfect sphere centered at origin, vertex normals are simply
    the normalized vertex positions.
    
    Args:
        vertices: torch tensor of shape (V, 3)
        faces: torch tensor of shape (F, 3) (used for proper normal calculation)
        normalize: whether to normalize the normals
    
    Returns:
        normals: torch tensor of shape (V, 3)
    """
    # Compute per-face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    
    # Average face normals to get vertex normals
    num_verts = vertices.shape[0]
    vertex_normals = torch.zeros_like(vertices)
    
    for i in range(3):
        vertex_normals.scatter_add_(0, faces[:, i].unsqueeze(1).expand(-1, 3), face_normals)
    
    if normalize:
        vertex_normals = vertex_normals / (torch.norm(vertex_normals, dim=1, keepdim=True) + 1e-8)
    
    return vertex_normals
