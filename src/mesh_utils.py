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


def build_knn_graph(vertices, k=16, chunk_size=2048):
    """
    Build an undirected k-NN graph from point positions.

    Args:
        vertices: torch tensor of shape (V, 3)
        k: number of neighbors per vertex
        chunk_size: chunk size for distance computation

    Returns:
        edges: torch tensor of shape (E, 2)
    """
    if not torch.is_tensor(vertices):
        vertices = torch.as_tensor(vertices, dtype=torch.float32)

    num_verts = vertices.shape[0]
    device = vertices.device

    if num_verts <= 1 or k <= 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    k = min(k, num_verts - 1)
    chunk_size = max(int(chunk_size), 1)

    edges_chunks = []
    for start in range(0, num_verts, chunk_size):
        end = min(start + chunk_size, num_verts)
        chunk = vertices[start:end]

        dist = torch.cdist(chunk, vertices)
        row_indices = torch.arange(start, end, device=device)
        dist[torch.arange(end - start, device=device), row_indices] = float('inf')

        knn_idx = torch.topk(dist, k=k, largest=False).indices
        src = row_indices.unsqueeze(1).expand(-1, k)
        edges_chunks.append(torch.stack([src.reshape(-1), knn_idx.reshape(-1)], dim=1))

    edges = torch.cat(edges_chunks, dim=0) if edges_chunks else torch.empty((0, 2), dtype=torch.long, device=device)
    if edges.numel() == 0:
        return edges

    edges = torch.sort(edges, dim=1).values
    return torch.unique(edges, dim=0)


def build_radius_graph(vertices, radius=0.1, chunk_size=2048):
    """
    Build an undirected radius graph from point positions.

    Args:
        vertices: torch tensor of shape (V, 3)
        radius: distance threshold
        chunk_size: chunk size for distance computation

    Returns:
        edges: torch tensor of shape (E, 2)
    """
    if not torch.is_tensor(vertices):
        vertices = torch.as_tensor(vertices, dtype=torch.float32)

    num_verts = vertices.shape[0]
    device = vertices.device

    if num_verts <= 1 or radius <= 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    chunk_size = max(int(chunk_size), 1)
    edges_chunks = []

    for start in range(0, num_verts, chunk_size):
        end = min(start + chunk_size, num_verts)
        chunk = vertices[start:end]

        dist = torch.cdist(chunk, vertices)
        row_indices = torch.arange(start, end, device=device)
        dist[torch.arange(end - start, device=device), row_indices] = float('inf')

        mask = dist <= radius
        rows, cols = torch.where(mask)
        if rows.numel() > 0:
            src = rows + start
            edges_chunks.append(torch.stack([src, cols], dim=1))

    edges = torch.cat(edges_chunks, dim=0) if edges_chunks else torch.empty((0, 2), dtype=torch.long, device=device)
    if edges.numel() == 0:
        return edges

    edges = torch.sort(edges, dim=1).values
    return torch.unique(edges, dim=0)


def build_min_ball_graph(vertices, k=16, chunk_size=2048):
    """
    Build an undirected graph using per-vertex k-NN radius (minimum ball).

    For each vertex i, use the distance to its k-th nearest neighbor as a radius
    and connect to all vertices within that radius.
    """
    if not torch.is_tensor(vertices):
        vertices = torch.as_tensor(vertices, dtype=torch.float32)

    num_verts = vertices.shape[0]
    device = vertices.device

    if num_verts <= 1 or k <= 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    k = min(k, num_verts - 1)
    chunk_size = max(int(chunk_size), 1)
    edges_chunks = []

    for start in range(0, num_verts, chunk_size):
        end = min(start + chunk_size, num_verts)
        chunk = vertices[start:end]

        dist = torch.cdist(chunk, vertices)
        row_indices = torch.arange(start, end, device=device)
        dist[torch.arange(end - start, device=device), row_indices] = float('inf')

        knn_dist = torch.topk(dist, k=k, largest=False).values
        radius = knn_dist[:, -1].unsqueeze(1)

        mask = dist <= radius
        rows, cols = torch.where(mask)
        if rows.numel() > 0:
            src = rows + start
            edges_chunks.append(torch.stack([src, cols], dim=1))

    edges = torch.cat(edges_chunks, dim=0) if edges_chunks else torch.empty((0, 2), dtype=torch.long, device=device)
    if edges.numel() == 0:
        return edges

    edges = torch.sort(edges, dim=1).values
    return torch.unique(edges, dim=0)


def build_dynamic_graph(vertices, mode='knn', k=16, radius=0.1, chunk_size=2048):
    """
    Build an undirected graph from point positions.

    Args:
        mode: 'knn', 'radius', or 'min_ball'
        k: k for k-NN/min_ball
        radius: distance threshold for radius graph
        chunk_size: chunk size for distance computation

    Returns:
        edges: torch tensor of shape (E, 2)
    """
    if mode == 'knn':
        return build_knn_graph(vertices, k=k, chunk_size=chunk_size)
    if mode == 'radius':
        return build_radius_graph(vertices, radius=radius, chunk_size=chunk_size)
    if mode == 'min_ball':
        return build_min_ball_graph(vertices, k=k, chunk_size=chunk_size)
    raise ValueError(f"Unsupported graph mode: {mode}. Use 'knn', 'radius', or 'min_ball'.")


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
