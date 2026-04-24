"""
Example simplicial complexes L for testing the Alexander duality pipeline.

Each function returns (L_simplices, points_L) where:
  - L_simplices is a list of sorted lists of vertex indices, mixed dimensions
    (vertices, edges, triangles), matching the format produced by getAlphaComplex.
  - points_L is an (n, 2) numpy array of vertex coordinates.

The three examples are:
  A. A 2-manifold with boundary: an annulus (disk with a triangular hole).
  B. Non-manifold, two triangles joined at a single vertex (bowtie).
  C. Non-manifold, two triangles sharing one edge with no third triangle
     filling either side. NOT a 2-manifold-with-boundary: the shared edge
     has no half-disk neighborhood.
"""

import numpy as np
from itertools import combinations


def _close_under_faces(maximal_simplices):
    """
    Given a list of simplices (each is a list/tuple of vertex indices), return
    the full simplicial complex closure: every unique face of every input
    simplex, each as a sorted list. Ordering is by dimension
    (vertices first, then edges, then triangles).
    """
    seen = set()
    for s in maximal_simplices:
        s = sorted(s)
        for k in range(1, len(s) + 1):
            for face in combinations(s, k):
                seen.add(tuple(face))
    by_dim = sorted(seen, key=lambda t: (len(t), t))
    return [list(t) for t in by_dim]


def make_Annulus():
    """
    Manifold with boundary - a triangulated annulus.

    It's an outer hexagon (vertices 0..5) and an inner
    triangle (vertices 6..8). Triangulated by
    connecting each inner vertex to two outer vertices.
    """
    # Outer hexagon at radius 1, inner triangle at radius 0.35.
    outer_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    inner_angles = np.linspace(np.pi / 6, 2 * np.pi + np.pi / 6, 3, endpoint=False)
    outer = np.stack([np.cos(outer_angles), np.sin(outer_angles)], axis=1)
    inner = 0.35 * np.stack([np.cos(inner_angles), np.sin(inner_angles)], axis=1)
    points_L = np.vstack([outer, inner])

    # Triangles in the annular region. Vertices: outer 0..5, inner 6..8.
    # Each inner vertex i (i=6,7,8) is connected to two consecutive outer
    # vertices, and consecutive inner vertices share a connecting outer vertex.
    # Layout: inner 6 sits opposite outer 1, etc. We just enumerate by hand.
    triangles = [
        [0, 1, 6],
        [1, 6, 7],
        [1, 2, 7],
        [2, 3, 7],
        [3, 7, 8],
        [3, 4, 8],
        [4, 5, 8],
        [5, 0, 8],
        [0, 6, 8],
    ]
    L = _close_under_faces(triangles)
    return L, points_L


def make_Papillon():
    """
    The papillon - Not manifold.

    Vertices 0,1,2 form one triangle; vertices 2,3,4 form the other. They share
    vertex 2 only. The link of vertex 2 is two disjoint edges, so L is not
    a manifold there.
    """
    points_L = np.array([
        [-1.0, 0.0],   # 0
        [-1.0, 1.0],   # 1
        [0.0, 0.5],   # 2  (the shared vertex)
        [1.0, 1.0],   # 3
        [1.0, 0.0],   # 4
    ])
    edges = [
        [0, 1],
        [0,2],
        [1,2],
        [2,3],
        [2,4],
        [3,4]
    ]

    L = _close_under_faces(edges)

    # triangles = [
    #     [0, 1, 2],
    #     [2, 3, 4],
    # ]
    
    #L = _close_under_faces(triangles)
    
    return L, points_L


def make_Dumbbell():
    """
    Non-manifold: two triangles connected by one edge.

    Vertices 0,1,2 form one triangle; vertices 3,4,5 form another. They are
    connected by edge [1,3].
    """
    points_L = np.array([
        [-1.0, 0.0],    # 0
        [0.0, 0.5],     # 1
        [-1.0, 1.0],    # 2
        [1.0, 0.5],    # 3
        [2.0, 0.0],    # 4
        [2.0, 1.0]     # 5
    ])
    edges = [
        [0, 1],
        [0, 2],
        [1, 2],
        [3, 4],
        [3, 5],
        [4, 5],
        [1, 3]
    ]
    L = _close_under_faces(edges)
    # triangles = [
    #     [0, 1, 2],
    #     [1, 2, 3],
    # ]
    # L = _close_under_faces(triangles)
    return L, points_L


def make_3_pages():
    """
    Variant of C: three triangles sharing one edge ("book with three pages").
    Genuinely non-manifold along the entire shared edge. The third triangle
    sticks out of the plane geometrically; here we embed it in R^2 with a
    deliberate overlap (vertex 4 placed where the planar drawing crosses
    triangle [0,1,2]). The abstract simplicial complex is still well-defined
    and non-manifold; only the planar drawing is degenerate.

    Use this only if you want the genuine 3-pages topology and are willing
    to live with a self-intersecting drawing. CDT will refuse to triangulate
    this in R^2 with vertex 4 on the wrong side, so feed CDT only the planar
    examples A, B, C.
    """
    points_L = np.array([
        [0.0, 0.0],     # 0
        [1.0, 0.0],     # 1
        [1.0, 1.0],     # 2
        [2.0, 0.5],     # 3
        [0.5, 1.5],     # 4 -- on the same side as 0; geometric overlap
    ])
    triangles = [
        [0, 1, 2],
        [1, 2, 3],
        [1, 2, 4],
    ]
    L = _close_under_faces(triangles)
    return L, points_L
