import numpy as np
import itertools
from itertools import combinations
from gudhi import AlphaComplex as AC

from src.abstract_subdivision import AbstractSubdivision

from src.relative_subdivision import subdivision, RelAbsSub, RelSub, RelSubGeneral, compute_Lout


### Relative subdivision

def relative_subdivide_complex(M, pointsM, L, pointsL, Subs):
    
    relM, relPoints = RelSub(M, pointsM, L, pointsL, Subs)
    
    # L simplices are unchanged in relative subdivision
    sL = L.copy()
    sPointsL = pointsL.copy()
    
    return relM, relPoints, sL, sPointsL

def compute_outer_boundary(sM):
    
    edges = sorted([s for s in sM if len(s) == 2])
    triangles = sorted([s for s in sM if len(s) == 3])
    
    edge_index = {tuple(e): i for i, e in enumerate(edges)}
    
    boundary_count = np.zeros(len(edges), dtype=int)
    
    for T in triangles:
        for e in [(T[0],T[1]), (T[1],T[2]), (T[0],T[2])]:
            boundary_count[edge_index[tuple(sorted(e))]] += 1
    
    boundary_mod2 = boundary_count % 2
    
    marked_edges = [
        edges[i]
        for i in range(len(edges))
        if boundary_mod2[i] == 1
    ]
    
    return marked_edges

## This is Maunder's supplement

def compute_supplement(relM, relPoints, pointsL):
    
    n_L_vertices = pointsL.shape[0]
    
    # These are exactly the L vertices
    SubComplexIndices = list(range(n_L_vertices))
    
    # Keep simplices not touching L
    Lbar = [
        simplex for simplex in relM
        if all(v >= n_L_vertices for v in simplex)
    ]
    
    # Explicit 0-simplices
    verticesLbar = [
        [i] for i in range(n_L_vertices, relPoints.shape[0])
    ]
    
    Lbar = verticesLbar + Lbar
    
    return Lbar, SubComplexIndices

# 3. The general tight supplement

def compute_tight_supplement(L, M, points_M, Subs):
    """
    Compute the tight supplement tilde{L}.

    Builds K = L ∪ L_out, calls RelSubGeneral, then keeps only simplices with
    no vertex in V(L). Per the lemma, the result consists of:
      (a) chains of barycenters of simplices in M setminus (L ∪ L_out),
      (d) simplices of L_out,
      (e) cones from L_out simplices to chains of barycenters in M setminus (L ∪ L_out).

    Parameters
    ----------
    L, M : list of sorted lists of int
    points_M : (n_M, d) ndarray
    Subs : dict[int, list]

    Returns
    -------
    Ltilde : list of sorted lists of int
        The tight supplement.
    sub_simplices: list of sorted lists of int
         — the full relative derived complex (M, K)'
    sub_points : (n_M + |M setminus K with len>=2|, d) ndarray
        Vertex coordinates for the relative subdivision (M, K)'.
    L_vertex_indices : list of int
        Indices in sub_points (that is, in points_M's prefix) of L's vertices.
        Not really sure I use this actually.
    Lout_vertex_indices : list of int
        Indices into sub_points of L_out's vertices.
    """
    L_vertex_indices = sorted({v for s in L for v in s})
    L_vertex_set = set(L_vertex_indices)

    # L_out is the subcomplex of M of simplices with no vertex in Vertices(L).
    Lout = compute_Lout(L, M)
    Lout_vertex_indices = sorted({v for s in Lout for v in s})

    # K = L ∪ L_out as a list of simplices, uniques.
    K_seen = set()
    K = []
    for s in L + Lout:
        key = tuple(sorted(s))
        if key not in K_seen:
            K_seen.add(key)
            K.append(list(key))
    K_vertex_indices = sorted({v for s in K for v in s})

    # Relative derived complex (M, K)'.
    sub_simplices, sub_points = RelSubGeneral(
        M, points_M, K, K_vertex_indices, Subs
    )

    # Tight supplement: simplices with no vertex in V(L). The vertices of L
    # in sub_points are exactly L_vertex_indices (prefix are preserved)
    Ltilde = [
        s for s in sub_simplices
        if all(v not in L_vertex_set for v in s)
    ]

    # Add explicitly the 0-simplices for any vertex of sub_points not in V(L) that
    # appears in some simplex of Ltilde. Follows the convention of the other
    # supplement functions (so connected_components can find isolated vertices).
    used_verts = {v for s in Ltilde for v in s}
    existing_verts = {s[0] for s in Ltilde if len(s) == 1}
    missing = [[v] for v in sorted(used_verts) if v not in existing_verts]
    Ltilde = Ltilde + missing

    # Every barycenter of an M\K simplex is a 0-simplex of tilde{L} (case (a) of the
    # lemma). Some of these may already appear inside larger
    # simplices of Ltilde; that's fine. Others are genuine isolated 0-simplices,
    # representing a new connected component.
    n_M = points_M.shape[0]
    n_bary = sub_points.shape[0] - n_M
    bary_zero_simplices = [[n_M + j] for j in range(n_bary)]

    # Also every 0-simplex of L_out is a 0-simplex of tilde{L} (case (d) of the lemma).
    # These were added verbatim by RelSubGeneral but double-check:
    
    Lout_zero_simplices = [[v] for v in Lout_vertex_indices]
    
    # Deduplicate against Ltilde's existing 0-simplices.
    existing_zero = {s[0] for s in Ltilde if len(s) == 1}
    extra_zero = [s for s in (bary_zero_simplices + Lout_zero_simplices) 
                  if s[0] not in existing_zero]

    Ltilde = Ltilde + extra_zero

    return Ltilde, sub_simplices, sub_points, L_vertex_indices, Lout_vertex_indices


# Union-find structure to compute the connected components

def connected_components(Lbar):
    
    vertices = [s[0] for s in Lbar if len(s) == 1]
    edges = [s for s in Lbar if len(s) == 2]
    
    parent = {v: v for v in vertices}
    
    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    
    for e in edges:
        union(e[0], e[1])
    
    components = {}
    for v in vertices:
        root = find(v)
        components.setdefault(root, []).append(v)
    
    return list(components.values())

# Check which in the set of CC share vertices with the boundary
def discard_boundary_components(CC, markedEdges):
    
    boundary_vertices = set()
    for e in markedEdges:
        boundary_vertices.update(e)
    
    valid = []
    excluded = []
    
    for c in CC:
        if any(v in boundary_vertices for v in c):
            excluded.append(c)
        else:
            valid.append(c)
    
    return valid, excluded

# This case for when there is no Lout
def extract_representatives_relative(validCC, MminusL, pointsL, M):
    
    offset = pointsL.shape[0]
    
    edges = sorted([s for s in M if len(s) == 2])
    edge_index = {tuple(e): i for i,e in enumerate(edges)}
    
    repr_edges = [] # store edges that are representative cycles (for plotting)
    cycles = [] # store the repr cycles individually
    
    for comp in validCC:
        
        repr_triangles = []
        
        for v in comp:
            if v >= offset:
                simplex = MminusL[v - offset]   # <-- FIX HERE
                if len(simplex) == 3:
                    repr_triangles.append(simplex)
        
        boundary_count = np.zeros(len(edges), dtype=int)
        
        for T in repr_triangles:
            for e in [(T[0],T[1]), (T[0],T[2]), (T[1],T[2])]:
                boundary_count[edge_index[tuple(sorted(e))]] += 1
        
        boundary_mod2 = boundary_count % 2

        new_cycle = [ edges[i] for i in range(len(edges)) if boundary_mod2[i] == 1 ]
        repr_edges.extend(new_cycle)
        cycles.append(new_cycle)
        
    
    return repr_edges, cycles


# 4. Components-to-cycles map for the tight case, general Lout

def extract_representatives_tight(valid_CC, L, M, points_M, Lout_vertex_indices,
                                  n_M_points):
    """
    Pull each connected component of tilde{L} back to top-dim simplices of M, take
    Z/2 boundary, and return the L-edges appearing in the result.

    A vertex v of tilde{L} is one of:
      - A vertex of L_out (v < n_M_points), corresponding to a 0-simplex of M.
      - A barycenter (v >= n_M_points), corresponding to a positive-dim simplex
        of M setminus (L ∪ L_out).

    For each component, we collect the top-dim (here triangle) simplices of M that
    correspond to its barycenter vertices, plus -- for components touching
    L_out -- the top-dim simplices of M whose support is entirely in L_out.
    Then we Z/2-sum their boundaries and read off the edges that lie in L.

    Parameters
    ----------
    valid_CC : list of lists of int
        Connected components of tilde{L} (after discarding boundary components),
        each a list of vertex indices in the tilde{L} index range.
    L, M : list of sorted lists of int
    points_M : (n_M, d) ndarray
    Lout_vertex_indices : list of int
        Indices into points_M of L_out's vertices.
    n_M_points : int
        Number of vertices in points_M (= offset for barycenter indices).

    Returns
    -------
    repr_edges : list of sorted lists of int
        Edges of L that appear in any component's representative cycle.
    """
    L_keys = {tuple(sorted(s)) for s in L}
    L_edges = sorted([s for s in L if len(s) == 2])
    edge_index = {tuple(e): i for i, e in enumerate(L_edges)}

    Lout_vertex_set = set(Lout_vertex_indices)

    # Build M \ K once: triangles of M not in L and not in L_out.
    Lout = compute_Lout(L, M)
    Lout_keys = {tuple(sorted(s)) for s in Lout}
    M_triangles = [s for s in M if len(s) == 3]
    MminusK_triangles = [
        s for s in M_triangles
        if tuple(sorted(s)) not in L_keys and tuple(sorted(s)) not in Lout_keys
    ]
    MminusK_simplices_dim_geq_1 = [
        s for s in M
        if tuple(sorted(s)) not in L_keys and tuple(sorted(s)) not in Lout_keys
        and len(s) >= 2
    ]
    # Map barycenter index back to the simplex it represents
    bary_to_simplex = {
        n_M_points + j: s
        for j, s in enumerate(MminusK_simplices_dim_geq_1)
    }
    # Triangles of L_out (top-dim simplices entirely outside L).
    Lout_triangles = [s for s in Lout if len(s) == 3]
    Lout_triangle_keys = {tuple(sorted(s)) for s in Lout_triangles}

    repr_edges = [] # store edges that are representative cycles (for plotting)
    cycles = [] # store the repr cycles individually

    for comp in valid_CC:
        comp_set = set(comp)

        # Collect top-dim simplices to take boundary of.
        repr_triangles = []

        # (a) Barycenters in the component that correspond to triangles of M\K.
        for v in comp:
            if v >= n_M_points:
                sigma = bary_to_simplex.get(v)
                if sigma is not None and len(sigma) == 3:
                    repr_triangles.append(sigma)

        # (b) Triangles of L_out whose vertices belong to this component.
        # It suffices to check e.g. the first one, for speed
        # A triangle of L_out sits inside tilde{L} as a 2-simplex of K (case (d) of
        # the lemma).
        for T in Lout_triangles:
            if T[0] in comp_set:
                repr_triangles.append(T)

        # Z/2 boundary, restricted to L's edges.
        boundary_count = np.zeros(len(L_edges), dtype=int)
        for T in repr_triangles:
            T_sorted = sorted(T)
            for e in [(T_sorted[0], T_sorted[1]),
                      (T_sorted[0], T_sorted[2]),
                      (T_sorted[1], T_sorted[2])]:
                if e in edge_index:
                    boundary_count[edge_index[e]] += 1

        boundary_mod2 = boundary_count % 2

        new_cycle = [ L_edges[i] for i in range(len(L_edges)) if boundary_mod2[i] == 1 ]
        repr_edges.extend(new_cycle)
        cycles.append(new_cycle)

    return repr_edges, cycles

