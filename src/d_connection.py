import numpy as np
from itertools import combinations


def top_dim_connected_components(complex, LinM):
    """ Compute (d-1)-connected components of a simplicial complex.
    
    Two top-dim simplices are equivalent iff they share a codim-1 face
    AND that face is NOT in the LinM subcomplex.
    The relation is closed transitively.
    
    Parameters
    ----------
    complex : SimplicialComplex
    LinM : Inclusion
        An inclusion (subcomplex ⊆ complex). Codim-1 faces belonging to
        the subcomplex are NOT used to merge top-simplices.
    
    Returns
    -------
    components : list of np.ndarray
        Each entry is an int32 array of FLAT indices of top-dim simplices
        in the same component, sorted ascending.
    """
    top_dim = complex.dim
    if top_dim < 1:
        return []
    
    top_simplices = complex.simplices_in_dim[top_dim]
    n_top = len(top_simplices)
    if n_top == 0:
        return []
    
    # Build face -> list of top-simplex local indices.
    face_to_top = {}
    for i, sigma in enumerate(top_simplices):
        sigma_sorted = sorted(int(v) for v in sigma)
        for j in range(len(sigma_sorted)):
            face = tuple(sigma_sorted[:j] + sigma_sorted[j+1:])
            face_to_top.setdefault(face, []).append(i)
    
    # Optional: build the excluded face set, expressed in `complex`'s
    # vertex indexing.
    excluded_face_set = set()
    sub = LinM.small
    vmap = LinM.vertex_map  # sub-vertex -> complex-vertex
    if sub.dim >= top_dim - 1:
        for row in sub.simplices_in_dim[top_dim - 1]:
            # Translate to `complex`'s vertex indexing.
            face = tuple(sorted(int(vmap[int(v)]) for v in row))
            excluded_face_set.add(face)
    
    # Union-find on top-simplex local indices.
    parent = np.arange(n_top, dtype=np.int32)
    
    def find(v):
        root = v
        while parent[root] != root:
            root = parent[root]
        while parent[v] != root:
            parent[v], v = root, parent[v]
        return root
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    
    for face, top_idxs in face_to_top.items():
        if face in excluded_face_set:
            continue  # do NOT merge across L-edges
        for j in range(1, len(top_idxs)):
            union(top_idxs[0], top_idxs[j])
    
    # Translate to flat indices in `complex`.
    flat_offset = sum(complex.simplices_in_dim[k].shape[0] 
                      for k in range(top_dim))
    
    roots = np.array([find(v) for v in range(n_top)], dtype=np.int32)
    by_root = {}
    for i, r in enumerate(roots):
        by_root.setdefault(int(r), []).append(flat_offset + i)
    
    components = [np.array(sorted(idxs), dtype=np.int32) 
                  for idxs in by_root.values()]
    components.sort(key=lambda arr: int(arr[0]))
    return components

def discard_boundary_components_topdim(M, components):
    """ Partition top-dim-component lists into valid (no boundary contact)
    and excluded (touching ∂|M|).
    
    A component touches the boundary iff any of its top-dim simplices has a
    codim-1 face that lies on ∂|M|.
    
    Parameters
    ----------
    M : SimplicialComplex
        The ambient complex. Boundary computed via M.boundary().
    components : list of np.ndarray
        Each entry is an int32 array of flat indices into M of top-dim
        simplices, as returned by top_dim_connected_components(diff_incl.small)
        AFTER translation back to M's flat indices (see below).
    
    Returns
    -------
    valid : list of np.ndarray
    excluded : list of np.ndarray
    
    Notes
    -----
    The components must reference flat indices in M. If they were produced
    by `top_dim_connected_components(diff_incl.small)`, those indices are
    in the small complex's flat-index space, NOT M's. Use 
    `_components_to_M_flats(diff_incl, components)` to translate.
    """
    boundary_codim1 = M.boundary()
    boundary_face_set = {tuple(sorted(int(v) for v in face)) 
                         for face in boundary_codim1}
    
    valid = []
    excluded = []
    for comp in components:
        touches_boundary = False
        for flat in comp:
            sigma = M.simplex_by_flat_index(int(flat))
            sigma_sorted = sorted(int(v) for v in sigma)
            for j in range(len(sigma_sorted)):
                face = tuple(sigma_sorted[:j] + sigma_sorted[j+1:])
                if face in boundary_face_set:
                    touches_boundary = True
                    break
            if touches_boundary:
                break
        if touches_boundary:
            excluded.append(comp)
        else:
            valid.append(comp)
    return valid, excluded


def representative_cycles_topdim(M, valid_components, base_inclusion):
    """ Extract representative cycles in L from valid top-dim components of
    the complement complex.
    
    For each valid component, take the Z/2 boundary of its top-dim simplices
    and return the codim-1 simplices appearing an odd number of times. By
    Alexander duality, this lies in L (we assert it as a sanity check).
    
    Parameters
    ----------
    M : SimplicialComplex
    valid_components : list of np.ndarray
        Components as flat indices of top-dim simplices in M.
    base_inclusion : Inclusion
        The original L ⊆ M, used to translate cycle edges back to L's
        vertex indexing.
    
    Returns
    -------
    cycles : list of list of tuple
        Per-component lists of L-edges (sorted tuples in L's vertex indexing).
    """
    # L-edge flats in M, plus the m_to_l vertex map.
    L_edge_flats_in_M = set()
    if not base_inclusion.is_trivial:
        L = base_inclusion.small
        if L.dim >= 1:
            for L_edge in L.simplices_in_dim[1]:
                m_edge = tuple(sorted(int(base_inclusion.vertex_map[v]) 
                                      for v in L_edge))
                L_edge_flats_in_M.add(M._simplex_to_flat[m_edge])
    m_to_L = base_inclusion.m_to_l
    
    cycles = []
    for comp in valid_components:
        # Z/2 boundary count over codim-1 faces.
        boundary_count = {}
        for flat in comp:
            sigma = M.simplex_by_flat_index(int(flat))
            sigma_sorted = sorted(int(v) for v in sigma)
            for j in range(len(sigma_sorted)):
                face = tuple(sigma_sorted[:j] + sigma_sorted[j+1:])
                boundary_count[face] = boundary_count.get(face, 0) + 1
        
        cycle_faces_in_M = [face for face, cnt in boundary_count.items() 
                            if cnt % 2 == 1]
        
        # Sanity check: by Alexander duality, every cycle face is in L.
        for face in cycle_faces_in_M:
            assert M._simplex_to_flat[face] in L_edge_flats_in_M, \
                (f"Cycle face {face} is not in L — Alexander duality "
                 f"violated. Bug somewhere upstream.")
        
        # Translate to L's vertex indexing.
        cycle_in_L = []
        for face in cycle_faces_in_M:
            l_face = tuple(sorted(int(m_to_L[v]) for v in face))
            cycle_in_L.append(l_face)
        
        cycles.append(cycle_in_L)
    
    return cycles

def components_to_M_flats(diff_inclusion, small_components):
    """ Translate components from diff_inclusion.small flat indices to
    diff_inclusion.large (= M) flat indices.
    
    Parameters
    ----------
    diff_inclusion : Inclusion
        The (complement_complex ⊆ M) inclusion.
    small_components : list of np.ndarray
        Components as flat indices in diff_inclusion.small, as returned by
        top_dim_connected_components(diff_inclusion.small).
    
    Returns
    -------
    list of np.ndarray with flat indices in M.
    """
    small = diff_inclusion.small
    M = diff_inclusion.large
    vmap = diff_inclusion.vertex_map  # length small.n_points; small_v -> M_v
    
    out = []
    for comp in small_components:
        m_flats = []
        for small_flat in comp:
            small_sigma = small.simplex_by_flat_index(int(small_flat))
            m_sigma = tuple(sorted(int(vmap[int(v)]) for v in small_sigma))
            m_flats.append(M._simplex_to_flat[m_sigma])
        out.append(np.array(sorted(m_flats), dtype=np.int32))
    return out