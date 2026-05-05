"""
Minimum-ball connectivity prototype inspired by DMesh++.

This module builds candidate triangles from KNN proposals, scores faces using a
minimum enclosing ball test, and converts weighted faces into weighted edges.
"""
import torch


def build_knn_indices(vertices, k=16, chunk_size=2048):
    """
    Build KNN indices for each vertex.

    Returns:
        knn_idx: (V, k) long tensor
        knn_dist: (V, k) float tensor
    """
    if not torch.is_tensor(vertices):
        vertices = torch.as_tensor(vertices, dtype=torch.float32)

    num_verts = vertices.shape[0]
    device = vertices.device

    if num_verts <= 1 or k <= 0:
        return (
            torch.empty((num_verts, 0), dtype=torch.long, device=device),
            torch.empty((num_verts, 0), dtype=vertices.dtype, device=device),
        )

    k = min(k, num_verts - 1)
    chunk_size = max(int(chunk_size), 1)

    knn_idx_chunks = []
    knn_dist_chunks = []

    for start in range(0, num_verts, chunk_size):
        end = min(start + chunk_size, num_verts)
        chunk = vertices[start:end]

        dist = torch.cdist(chunk, vertices)
        row_indices = torch.arange(start, end, device=device)
        dist[torch.arange(end - start, device=device), row_indices] = float('inf')

        knn = torch.topk(dist, k=k, largest=False)
        knn_idx_chunks.append(knn.indices)
        knn_dist_chunks.append(knn.values)

    knn_idx = torch.cat(knn_idx_chunks, dim=0)
    knn_dist = torch.cat(knn_dist_chunks, dim=0)
    return knn_idx, knn_dist


def propose_triangles(knn_idx, max_triangles_per_vertex=32):
    """
    Propose candidate triangles from KNN neighborhoods.

    Args:
        knn_idx: (V, k) long tensor of neighbors per vertex
        max_triangles_per_vertex: cap on triangles per vertex (None for all)

    Returns:
        triangles: (T, 3) long tensor, sorted per triangle and unique
    """
    if knn_idx.numel() == 0:
        return torch.empty((0, 3), dtype=torch.long, device=knn_idx.device)

    num_verts, k = knn_idx.shape
    max_triangles_per_vertex = None if max_triangles_per_vertex is None else int(max_triangles_per_vertex)

    triangles = []
    for i in range(num_verts):
        neighbors = knn_idx[i].tolist()
        count = 0
        for a in range(k):
            for b in range(a + 1, k):
                triangles.append([i, neighbors[a], neighbors[b]])
                count += 1
                if max_triangles_per_vertex is not None and count >= max_triangles_per_vertex:
                    break
            if max_triangles_per_vertex is not None and count >= max_triangles_per_vertex:
                break

    if not triangles:
        return torch.empty((0, 3), dtype=torch.long, device=knn_idx.device)

    tri = torch.tensor(triangles, dtype=torch.long, device=knn_idx.device)
    tri = torch.sort(tri, dim=1).values
    return torch.unique(tri, dim=0)


def build_knn_faces(vertices, k=16, max_triangles_per_vertex=32, knn_chunk_size=2048):
    """
    Build triangle faces from KNN proposals.

    Returns:
        triangles: (T, 3)
        diagnostics: dict of summary stats
    """
    knn_idx, knn_dist = build_knn_indices(vertices, k=k, chunk_size=knn_chunk_size)
    triangles = propose_triangles(knn_idx, max_triangles_per_vertex=max_triangles_per_vertex)

    diagnostics = {
        'knn_num_faces': int(triangles.shape[0]),
    }
    if knn_dist.numel() > 0:
        nn_min = knn_dist[:, 0]
        diagnostics.update({
            'knn_nn_min': float(nn_min.min().item()),
            'knn_nn_mean': float(nn_min.mean().item()),
            'knn_nn_max': float(nn_min.max().item()),
        })

    return triangles, diagnostics


def minimum_enclosing_ball(triangles, vertices, eps=1e-12):
    """
    Compute the minimum enclosing ball (center, radius) for each triangle.

    For obtuse or degenerate triangles, returns the ball of the longest edge.
    """
    if triangles.numel() == 0:
        return (
            torch.empty((0, 3), dtype=vertices.dtype, device=vertices.device),
            torch.empty((0,), dtype=vertices.dtype, device=vertices.device),
        )

    a = vertices[triangles[:, 0]]
    b = vertices[triangles[:, 1]]
    c = vertices[triangles[:, 2]]

    ab = b - a
    ac = c - a
    bc = c - b

    lab2 = torch.sum(ab * ab, dim=1)
    lac2 = torch.sum(ac * ac, dim=1)
    lbc2 = torch.sum(bc * bc, dim=1)

    dot_ab_ac = torch.sum(ab * ac, dim=1)
    dot_ba_bc = torch.sum((-ab) * bc, dim=1)
    dot_ca_cb = torch.sum((-ac) * (-bc), dim=1)

    obtuse = (dot_ab_ac < 0) | (dot_ba_bc < 0) | (dot_ca_cb < 0)

    cross_ab_ac = torch.cross(ab, ac, dim=1)
    denom = 2.0 * torch.sum(cross_ab_ac * cross_ab_ac, dim=1)
    degenerate = denom <= eps

    use_longest_edge = obtuse | degenerate

    centers = torch.zeros_like(a)
    radii = torch.zeros(a.shape[0], dtype=vertices.dtype, device=vertices.device)

    if torch.any(~use_longest_edge):
        mask = ~use_longest_edge
        ab_m = ab[mask]
        ac_m = ac[mask]
        a_m = a[mask]
        cross_m = cross_ab_ac[mask]
        denom_m = denom[mask]
        lab2_m = lab2[mask]
        lac2_m = lac2[mask]

        term1 = torch.cross(cross_m, ab_m, dim=1) * lac2_m.unsqueeze(1)
        term2 = torch.cross(ac_m, cross_m, dim=1) * lab2_m.unsqueeze(1)
        center_m = a_m + (term1 + term2) / denom_m.unsqueeze(1)
        centers[mask] = center_m
        radii[mask] = torch.norm(center_m - a_m, dim=1)

    if torch.any(use_longest_edge):
        mask = use_longest_edge
        max_idx = torch.stack([lab2, lac2, lbc2], dim=1).argmax(dim=1)
        centers_obtuse = centers[mask]
        radii_obtuse = radii[mask]

        idx_mask = mask & (max_idx == 0)
        centers[idx_mask] = 0.5 * (a[idx_mask] + b[idx_mask])
        radii[idx_mask] = 0.5 * torch.sqrt(lab2[idx_mask])

        idx_mask = mask & (max_idx == 1)
        centers[idx_mask] = 0.5 * (a[idx_mask] + c[idx_mask])
        radii[idx_mask] = 0.5 * torch.sqrt(lac2[idx_mask])

        idx_mask = mask & (max_idx == 2)
        centers[idx_mask] = 0.5 * (b[idx_mask] + c[idx_mask])
        radii[idx_mask] = 0.5 * torch.sqrt(lbc2[idx_mask])

    return centers, radii


def compute_face_weights(vertices,
                         triangles,
                         knn_idx,
                         alpha_min=10.0,
                         tri_chunk_size=2048):
    """
    Compute minimum-ball face weights using a local candidate set.

    The signed distance d is approximated using the union of KNN neighbors of
    the triangle vertices.
    """
    device = vertices.device
    if triangles.numel() == 0:
        empty = torch.empty((0,), dtype=vertices.dtype, device=device)
        return empty, empty, empty

    centers, radii = minimum_enclosing_ball(triangles, vertices)
    num_tris = triangles.shape[0]
    k = knn_idx.shape[1]

    tri_chunk_size = max(int(tri_chunk_size), 1)
    d_vals = torch.empty((num_tris,), dtype=vertices.dtype, device=device)

    for start in range(0, num_tris, tri_chunk_size):
        end = min(start + tri_chunk_size, num_tris)
        tri = triangles[start:end]
        centers_chunk = centers[start:end]
        radii_chunk = radii[start:end]

        cand_idx = torch.cat(
            [
                knn_idx[tri[:, 0]],
                knn_idx[tri[:, 1]],
                knn_idx[tri[:, 2]],
                tri,
            ],
            dim=1,
        )

        cand_pts = vertices[cand_idx]
        diff = cand_pts - centers_chunk.unsqueeze(1)
        dists = torch.norm(diff, dim=2)

        mask = (
            cand_idx == tri[:, 0].unsqueeze(1)
        ) | (
            cand_idx == tri[:, 1].unsqueeze(1)
        ) | (
            cand_idx == tri[:, 2].unsqueeze(1)
        )
        dists = dists.masked_fill(mask, float('inf'))

        min_dist = torch.min(dists, dim=1).values
        d_vals[start:end] = min_dist - radii_chunk

    weights = torch.sigmoid(alpha_min * d_vals)
    return weights, d_vals, radii


def faces_to_weighted_edges(triangles, face_weights):
    """
    Convert weighted triangles into weighted undirected edges.
    """
    if triangles.numel() == 0:
        return (
            torch.empty((0, 2), dtype=torch.long, device=triangles.device),
            torch.empty((0,), dtype=face_weights.dtype, device=triangles.device),
        )

    edges = torch.cat(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ],
        dim=0,
    )
    weights = torch.cat([face_weights, face_weights, face_weights], dim=0)

    edges = torch.sort(edges, dim=1).values
    unique_edges, inverse = torch.unique(edges, dim=0, return_inverse=True)
    edge_weights = torch.zeros(unique_edges.shape[0], dtype=weights.dtype, device=weights.device)
    edge_weights.scatter_add_(0, inverse, weights)

    return unique_edges, edge_weights


def build_minimum_ball_faces(vertices,
                             k=16,
                             alpha_min=10.0,
                             max_triangles_per_vertex=32,
                             knn_chunk_size=2048,
                             tri_chunk_size=2048,
                             face_weight_threshold=0.0):
    """
    Build weighted triangle faces using minimum-ball selection.

    Returns:
        triangles: (T, 3)
        face_weights: (T,)
        diagnostics: dict of summary stats
    """
    knn_idx, knn_dist = build_knn_indices(vertices, k=k, chunk_size=knn_chunk_size)
    triangles = propose_triangles(knn_idx, max_triangles_per_vertex=max_triangles_per_vertex)

    face_weights, face_d, _face_r = compute_face_weights(
        vertices,
        triangles,
        knn_idx,
        alpha_min=alpha_min,
        tri_chunk_size=tri_chunk_size,
    )

    diagnostics = {
        'mb_num_faces_raw': int(triangles.shape[0]),
    }

    if face_weights.numel() > 0:
        diagnostics.update({
            'mb_face_w_min': float(face_weights.min().item()),
            'mb_face_w_mean': float(face_weights.mean().item()),
            'mb_face_w_max': float(face_weights.max().item()),
            'mb_face_d_min': float(face_d.min().item()),
            'mb_face_d_mean': float(face_d.mean().item()),
            'mb_face_d_max': float(face_d.max().item()),
        })

    if face_weight_threshold > 0.0 and face_weights.numel() > 0:
        mask = face_weights >= face_weight_threshold
        triangles = triangles[mask]
        face_weights = face_weights[mask]
        face_d = face_d[mask]

    diagnostics['mb_num_faces'] = int(triangles.shape[0])

    if knn_dist.numel() > 0:
        nn_min = knn_dist[:, 0]
        diagnostics.update({
            'mb_nn_min': float(nn_min.min().item()),
            'mb_nn_mean': float(nn_min.mean().item()),
            'mb_nn_max': float(nn_min.max().item()),
        })

    return triangles, face_weights, diagnostics


def build_minimum_ball_graph(vertices,
                             k=16,
                             alpha_min=10.0,
                             max_triangles_per_vertex=32,
                             knn_chunk_size=2048,
                             tri_chunk_size=2048,
                             collapse_threshold=0.0,
                             face_weight_threshold=0.0):
    """
    Build weighted edges using minimum-ball face selection.

    Returns:
        edges: (E, 2)
        edge_weights: (E,)
        diagnostics: dict of summary stats
    """
    knn_idx, knn_dist = build_knn_indices(vertices, k=k, chunk_size=knn_chunk_size)
    triangles, face_weights, diagnostics = build_minimum_ball_faces(
        vertices,
        k=k,
        alpha_min=alpha_min,
        max_triangles_per_vertex=max_triangles_per_vertex,
        knn_chunk_size=knn_chunk_size,
        tri_chunk_size=tri_chunk_size,
        face_weight_threshold=face_weight_threshold,
    )

    edges, edge_weights = faces_to_weighted_edges(triangles, face_weights)
    diagnostics['mb_num_edges'] = int(edges.shape[0])

    if edge_weights.numel() > 0:
        diagnostics.update({
            'mb_edge_w_min': float(edge_weights.min().item()),
            'mb_edge_w_mean': float(edge_weights.mean().item()),
            'mb_edge_w_max': float(edge_weights.max().item()),
        })

    if edges.numel() > 0:
        lengths = torch.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], dim=1)
        if collapse_threshold <= 0.0 and knn_dist.numel() > 0:
            collapse_threshold = 0.25 * torch.median(knn_dist[:, 0]).item()
        if collapse_threshold > 0.0:
            diagnostics['mb_collapse_ratio'] = float((lengths < collapse_threshold).float().mean().item())

    return edges, edge_weights, diagnostics
