"""
Constrained Delaunay Triangulation wrapper.

Given a planar simplicial complex L (vertices and constraint edges in R^2),
build a triangulation M of an enclosing disk such that L is a subcomplex of M.

Conventions
-----------
- L's vertex indices are preserved at the start of points_M, i.e.
  points_M[:n_L] == points_L exactly. Bounding-box vertices are appended
  at the end of points_M.
- M is returned in the same mixed-dimension format as getAlphaComplex:
  a list of sorted lists of vertex indices (vertices, edges, triangles).
- The bounding box is an axis-aligned square sized to enclose points_L,
  with a configurable padding factor (relative to the bounding-box extent).

Backend: triangle (Shewchuk's library) via the `triangle` Python package.
"""

import numpy as np
from itertools import combinations


def _bounding_box_square(points_L, padding_factor=0.5):
    """
    Build the four corner points of an axis-aligned square that encloses
    points_L with the given padding factor (0.5 = pad by half the extent
    on every side, so the square is twice as wide as the point cloud).

    Returns the corners in counter-clockwise order, suitable for use as
    a constraint polygon.
    """
    x_min, y_min = points_L.min(axis=0)
    x_max, y_max = points_L.max(axis=0)
    extent = max(x_max - x_min, y_max - y_min)
    pad = padding_factor * extent
    # Use the larger extent for a true square; centered on the point cloud.
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    half = 0.5 * extent + pad
    corners = np.array([
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ])
    return corners


def _faces_of(simplex_list, dim):
    """All sorted (dim+1)-tuples appearing as faces of any simplex in the list."""
    out = set()
    for s in simplex_list:
        if len(s) >= dim + 1:
            for face in combinations(sorted(s), dim + 1):
                out.add(face)
    return out


def constrained_delaunay(L_simplices, points_L, padding_factor=0.5,
                         forbid_steiner=True):
    """
    Extend L to a triangulated disk M via constrained Delaunay triangulation.

    Parameters
    ----------
    L_simplices : list of sorted lists of int
        Simplicial complex L as a list of mixed-dimension simplices, in the
        same format produced by getAlphaComplex.
    points_L : (n_L, 2) ndarray of float
        Coordinates of L's vertices.
    padding_factor : float, default 0.5
        Bounding-box padding as a fraction of the point cloud's extent.
    forbid_steiner : bool, default True
        If True, pass the 'YY' switch to `triangle` to forbid insertion of
        Steiner points on segments. This guarantees L's edges survive
        verbatim in M, which is required for L to be a subcomplex of M.
        If `triangle` cannot triangulate without Steiner points (e.g., a
        pathological input), it will raise an exception.

    Returns
    -------
    M_simplices : list of sorted lists of int
        Triangulated disk in mixed-dimension format (vertices + edges + triangles).
    points_M : (n_M, 2) ndarray of floats
        Vertex coordinates. The first n_L rows equal points_L exactly; the
        bounding-box corners are appended at the end.
    """
    points_L = np.asarray(points_L, dtype=float)
    n_L = points_L.shape[0]

    # Lazy import: the other helpers may be needed elsewhere
    import triangle as tr

    # The constraint edges are all the 1-simplices of L. Vertex indices in L are the
    # first n_L entries of the combined point array, so no offset is needed.
    L_edges = sorted(_faces_of(L_simplices, dim=1))
    constraint_edges = [list(e) for e in L_edges]

    # Bounding box: appended after L's points, so its indices are n_L..n_L+3.
    bbox = _bounding_box_square(points_L, padding_factor=padding_factor)
    points_combined = np.vstack([points_L, bbox])
    bbox_idx = n_L
    bbox_edges = [
        [bbox_idx + 0, bbox_idx + 1],
        [bbox_idx + 1, bbox_idx + 2],
        [bbox_idx + 2, bbox_idx + 3],
        [bbox_idx + 3, bbox_idx + 0],
    ]

    all_segments = constraint_edges + bbox_edges

    # define a planar straight-line graph
    pslg = dict(
        vertices=points_combined,
        segments=np.array(all_segments, dtype=int),
    )

    # Switches:
    #   'p' = treat input as a planar straight-line graph (use segments)
    #   'YY' = forbid Steiner points on any segment (preserves all input edges)
    switches = 'p'
    if forbid_steiner:
        switches += 'YY'

    result = tr.triangulate(pslg, switches)

    out_vertices = np.asarray(result['vertices'], dtype=float)
    out_triangles = np.asarray(result['triangles'], dtype=int)

    # Check: tr should not have inserted any new vertices since 'YY'
    # is passed and no triangle quality switches are used.
    if out_vertices.shape[0] != points_combined.shape[0]:
        raise RuntimeError(
            f"triangle inserted {out_vertices.shape[0] - points_combined.shape[0]} "
            f"Steiner points despite 'YY' switch. This usually means the input "
            f"PSLG is degenerate (overlapping or crossing constraint segments)."
        )
    # Also, the existing vertices should be in the same order
    if not np.allclose(out_vertices[:points_combined.shape[0]], points_combined):
        raise RuntimeError(
            "triangle reordered or perturbed input vertices. Cannot align L's "
            "indices with the output."
        )

    points_M = out_vertices  # same as points_combined, but use triangle's copy

    # Build M as mixed-dimension simplicial complex (vertices, edges, triangles).
    triangles_list = [sorted(t.tolist()) for t in out_triangles]
    edges_set = set()
    for t in triangles_list:
        for e in combinations(t, 2):
            edges_set.add(e)
    edges_list = [list(e) for e in sorted(edges_set)]
    vertices_list = [[i] for i in range(points_M.shape[0])]

    M_simplices = vertices_list + edges_list + triangles_list
    return M_simplices, points_M


def assert_subcomplex(L_simplices, M_simplices):
    """
    Assertion that checks every simplex of L appears in M (as a sorted tuple of 
    vertex indices). Returns True or raises AssertionError with a 
    list of missing simplices.
    """
    M_set = {tuple(sorted(s)) for s in M_simplices}
    missing = [s for s in L_simplices if tuple(sorted(s)) not in M_set]
    if missing:
        raise AssertionError(
            f"L is not a subcomplex of M. Missing simplices: {missing}"
        )
    return True
