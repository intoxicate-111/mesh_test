"""
Mesh utility functions for adjacency, edge operations, etc.
"""
import torch
import numpy as np


def compute_adjacency_list(faces, num_verts):
    """
    Compute vertex adjacency list from face connectivity.
    
    Args:
        faces: torch tensor of shape (F, 3)
        num_verts: number of vertices
    
    Returns:
        adjacency: list of lists, adjacency[i] contains indices of neighbors of vertex i
    """
    adjacency = [[] for _ in range(num_verts)]
    
    faces_np = faces.cpu().numpy() if torch.is_tensor(faces) else faces
    
    for face in faces_np:
        v0, v1, v2 = face
        
        # Add edges
        if v1 not in adjacency[v0]:
            adjacency[v0].append(v1)
        if v0 not in adjacency[v1]:
            adjacency[v1].append(v0)
        
        if v2 not in adjacency[v1]:
            adjacency[v1].append(v2)
        if v1 not in adjacency[v2]:
            adjacency[v2].append(v1)
        
        if v0 not in adjacency[v2]:
            adjacency[v2].append(v0)
        if v2 not in adjacency[v0]:
            adjacency[v0].append(v2)
    
    return adjacency


def get_edge_lengths(vertices, faces):
    """
    Compute edge lengths for each edge in the mesh.
    
    Args:
        vertices: torch tensor of shape (V, 3)
        faces: torch tensor of shape (F, 3)
    
    Returns:
        edge_lengths: dict mapping edge tuple to length
    """
    edge_lengths = {}
    
    for face in faces:
        edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        for v1, v2 in edges:
            edge_key = tuple(sorted([v1.item(), v2.item()]))
            if edge_key not in edge_lengths:
                edge_length = torch.norm(vertices[v1] - vertices[v2])
                edge_lengths[edge_key] = edge_length
    
    return edge_lengths


def get_edge_lengths_tensor(vertices, faces):
    """
    Compute edge lengths for optimization (vectorized).
    
    Args:
        vertices: torch tensor of shape (V, 3)
        faces: torch tensor of shape (F, 3)
    
    Returns:
        edges: torch tensor of shape (E, 2)
        edge_lengths: torch tensor of shape (E,)
    """
    # Get unique edges
    edges_list = []
    edge_set = set()
    
    for face in faces:
        for i in range(3):
            v1 = face[i]
            v2 = face[(i + 1) % 3]
            edge_key = tuple(sorted([v1.item(), v2.item()]))
            
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                edges_list.append([edge_key[0], edge_key[1]])
    
    edges = torch.tensor(edges_list, dtype=torch.long, device=vertices.device)
    
    if len(edges) > 0:
        edge_lengths = torch.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], dim=1)
    else:
        edge_lengths = torch.tensor([], dtype=vertices.dtype, device=vertices.device)
    
    return edges, edge_lengths


def compute_surface_area(vertices, faces):
    """
    Compute total surface area of the mesh.
    
    Args:
        vertices: torch tensor of shape (V, 3)
        faces: torch tensor of shape (F, 3)
    
    Returns:
        area: total surface area
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Compute area of each triangle
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = torch.cross(edge1, edge2)
    face_areas = 0.5 * torch.norm(cross, dim=1)
    
    total_area = torch.sum(face_areas)
    return total_area
