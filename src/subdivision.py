""" Subdivision class

Represents a subdivision of a simplicial complex M, combinatorially, as
chains of flat indices into M. The geometric realization (with barycenters
as explicit points) is produced on demand via `to_simplicial_complex()`.

Three subdivision types are supported through distinct constructors:

  Subdivision.barycentric(M)    — the full barycentric subdivision M'.
  Subdivision.relative(incl)    — the relative derived complex (M, L)'.
  Subdivision.tight(incl)       — the tight subdivision (M, L ∪ L_out)'.

Combinatorial representation
----------------------------
A simplex of the subdivision is encoded as a chain

    (f_0, f_1, ..., f_{k-1})

of flat indices into M, with strictly increasing parent dimensions:

    dim(M.simplex(f_0)) < dim(M.simplex(f_1)) < ... < dim(M.simplex(f_{k-1})).

The resulting subdivision simplex has:
  - length k if the chain does NOT start at an L-simplex (type (i) of Maunder's
    classification): k barycenters, one per chain element.
  - length k + dim(σ_{f_0}) if the chain DOES start at an L-simplex (type (iii)):
    dim(σ_{f_0}) + 1 vertices of σ_{f_0} + (k - 1) barycenters.

Length-1 chains (f_0,) are simplices of L when σ_{f_0} ∈ L (type (ii)), or
single barycenters of M\\L simplices when σ_{f_0} ∈ M\\L.

Chains are stored grouped by chain length:

  chains_by_length[length] : (n_chains, length) int32 array.

The tight subdivision re-uses the relative-subdivision machinery by first
forming the inclusion L ∪ L_out ⊆ M internally.

Notes
-----
- "Flat index" means the SimplicialComplex.flat_* indexing: vertices first,
  then edges, then triangles, etc., concatenated by increasing dimension.
- An Inclusion of None (i.e., `inclusion is None`) means "no relative
  subcomplex", which is the barycentric case.
"""

import numpy as np
from itertools import combinations

from .simplicial_complex import SimplicialComplex
from .inclusion import Inclusion


class Subdivision:

    def __init__(self, parent, inclusion, chains_by_length):
        """ Direct constructor. Use the class methods barycentric(),
        relative(), or tight() instead of calling this directly.

        Parameters:
        parent : SimplicialComplex
            The complex M being subdivided.
        inclusion : Inclusion or None
            The subcomplex L ⊆ M if this is a relative subdivision, else None.
        chains_by_length : dict[int, np.ndarray]
            For each key `length` (>= 1), an array of shape (n_chains, length)
            of int32 flat indices into parent, each row a strictly-ascending
            (by parent dimension) chain.
        """
        self.parent = parent
        self.inclusion = inclusion
        self.chains_by_length = chains_by_length

    # ----------------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------------

    @classmethod
    def barycentric(cls, M):
        """ Barycentric subdivision M'. """
        chains = _build_chains(M, inclusion_mask=None)
        return cls(parent=M, inclusion=None, chains_by_length=chains)

    @classmethod
    def relative(cls, inclusion):
        """ Relative derived complex (M, L)' for the given inclusion L ⊆ M. """
        chains = _build_chains(inclusion.large, inclusion_mask=inclusion.simplex_mask)
        return cls(parent=inclusion.large, inclusion=inclusion,
                   chains_by_length=chains)

    @classmethod
    def tight(cls, inclusion):
        """ Tight subdivision (M, L ∪ L_out)'.

        Internally computes L_out (= simplices of M with no vertex in V(L)),
        forms the inclusion L ∪ L_out ⊆ M, and builds the relative
        subdivision against that.
        """
        K_mask = _build_tight_mask(inclusion)
        # Build a fresh Inclusion for L ∪ L_out so the Subdivision has the
        # right inclusion field. This also means the downstream "which simplices
        # are in the subcomplex" queries use L ∪ L_out, not just L.
        #
        # The small complex is materialized by restricting M's simplices
        # to those marked by K_mask and closing under faces (they are already
        # closed since L and L_out are closed, and disjoint in vertex set).
        M = inclusion.large
        K_inclusion = _inclusion_from_mask(M, K_mask)
        chains = _build_chains(M, inclusion_mask=K_mask)
        return cls(parent=M, inclusion=K_inclusion, chains_by_length=chains)

    # ----------------------------------------------------------------------
    # Introspection
    # ----------------------------------------------------------------------

    @property
    def n_chains(self):
        """ Total number of chains across all lengths. """
        return sum(arr.shape[0] for arr in self.chains_by_length.values())

    def __repr__(self):
        counts = {length: arr.shape[0]
                  for length, arr in self.chains_by_length.items()}
        return (f"Subdivision(parent={self.parent!r}, "
                f"inclusion={self.inclusion!r}, "
                f"chain_counts={counts})")

    # ----------------------------------------------------------------------
    # Materialization
    # ----------------------------------------------------------------------

    def to_simplicial_complex(self):
        """ Build a standalone SimplicialComplex whose vertices are the
        barycenters of all chain-start simplices (for type (i)/(iii)) plus
        the original vertices of L that appear in type-(iii) cones, and
        whose simplices realize each chain as described above.

        Returns
        -------
        complex : SimplicialComplex
            The geometric realization of the subdivision.
        provenance : dict
            A mapping 'vertex_of_output' -> either ('parent_vertex', i)
            meaning it's a parent vertex of M (index i), or
            ('barycenter_of', flat) meaning it's the barycenter of the
            parent simplex at flat index `flat`. Useful for debugging
            and for later constructions that need vertex lineage.
        """
        M = self.parent
        mask = (self.inclusion.simplex_mask
                if self.inclusion is not None
                else None)

        # Every vertex of the output is either:
        #   - An original M-vertex (used only when it's an L-vertex appearing
        #     in a type-(iii) chain).
        #   - A barycenter of an M-simplex.
        # We allocate indices by walking all chains and collecting distinct
        # (kind, id) pairs.

        out_vertex_of = {}   # (kind, id) -> output vertex index
        provenance = []      # output index -> (kind, id)

        def _get_vertex(kind, id):
            key = (kind, int(id))
            idx = out_vertex_of.get(key)
            if idx is None:
                idx = len(provenance)
                out_vertex_of[key] = idx
                provenance.append(key)
            return idx

        # Pass 1: collect vertices and build output simplices as we go.
        out_simplices = []

        for length, chains in sorted(self.chains_by_length.items()):
            for chain in chains:
                f0 = int(chain[0])
                starts_in_L = mask is not None and mask[f0]

                if starts_in_L:
                    # Type (ii) if length == 1 — the whole L-simplex σ_{f_0}.
                    # Type (iii) if length > 1 — L-simplex vertices + barycenters.
                    sigma_a = M.simplex_by_flat_index(f0)
                    out_verts = [_get_vertex('parent_vertex', int(v))
                                 for v in sigma_a]
                    for f in chain[1:]:
                        out_verts.append(_get_vertex('barycenter_of', int(f)))
                    out_simplices.append(sorted(out_verts))
                else:
                    # Type (i): each chain element becomes one barycenter vertex.
                    out_verts = [_get_vertex('barycenter_of', int(f))
                                 for f in chain]
                    out_simplices.append(sorted(out_verts))

        # Pass 2: compute coordinates for every output vertex.
        n_out = len(provenance)
        points = np.zeros((n_out, M.points.shape[1]), dtype=M.points.dtype)
        for i, (kind, id) in enumerate(provenance):
            if kind == 'parent_vertex':
                points[i] = M.points[id]
            elif kind == 'barycenter_of':
                sigma_verts = M.simplex_by_flat_index(id)
                points[i] = M.points[sigma_verts].mean(axis=0)
            else:
                raise RuntimeError(f"Unknown provenance kind: {kind}")

        # Filter out 0-simplices (SimplicialComplex builds those from points)
        # and emit the rest. Length-1 chains that produce a single output
        # vertex (i.e., type (ii) L-vertices or type (i) barycenters) do not
        # become explicit 0-simplices in the complex input; they are covered
        # by the points array. But length-1 L-simplices of higher dim (e.g.,
        # a full edge [v0, v1] of L, as type (ii)) DO produce higher-dim
        # simplices in the output.
        simplices_for_complex = [s for s in out_simplices if len(s) >= 2]

        complex = SimplicialComplex(points, simplices_for_complex)
        return complex, {i: provenance[i] for i in range(n_out)}


# --------------------------------------------------------------------------
# Core chain-building algorithm
# --------------------------------------------------------------------------

def _build_chains(M, inclusion_mask):
    """ Build all chains of simplices of M with strictly increasing dimension.

    If `inclusion_mask` is None: all chains are included (barycentric).

    If `inclusion_mask` is a boolean array of length M.n_simplices marking
    the subcomplex L: chains are filtered per Maunder's classification.
    A chain (f_0, f_1, ..., f_{k-1}) is kept iff:
       - (type i)   all f_i are in M\\L, or
       - (type ii)  k == 1 and f_0 is in L, or
       - (type iii) k >= 2 and f_0 is in L and f_1, ..., f_{k-1} are in M\\L.

    Returns
    -------
    chains_by_length : dict[int, np.ndarray]
        Keys are chain lengths >= 1; values are (n_chains, length) int32 arrays.
    """
    n_simp = M.n_simplices
    dim_of = M._flat_dim  # length n_simp, dim of each simplex (int array)

    # Precompute the proper-face-of relation:
    # for each flat index f, parents_of[f] = list of flat indices of simplices
    # σ such that σ_f is a proper face of σ. We only need to go up one dim
    # each step (chains increase dim by at least 1 at each step, but we
    # enumerate chains by repeated extension).
    # Rather than "covers" (one-dim-higher), we need ALL proper supersets,
    # because a chain may skip dimensions. So parents_of[f] contains every
    # simplex of M that strictly contains σ_f.
    parents_of = [[] for _ in range(n_simp)]
    for parent_flat in range(n_simp):
        parent_verts = set(M.simplex_by_flat_index(parent_flat).tolist())
        if len(parent_verts) <= 1:
            continue
        for sub in _all_proper_nonempty_subsets(parent_verts):
            sub_tuple = tuple(sorted(sub))
            child_flat = M._simplex_to_flat.get(sub_tuple)
            if child_flat is not None:
                parents_of[child_flat].append(parent_flat)

    # Enumerate chains by iterative extension.
    # Length-1 chains: all flat indices satisfying the type-(i)/(ii) rule
    # (with the L-filter applied).
    chains_by_length = {}

    # Length 1:
    # - barycentric: every simplex
    # - relative:   every simplex (type ii for L-simplices; single-barycenter
    #               for M\\L simplices)
    length_1 = np.arange(n_simp, dtype=np.int32).reshape(-1, 1)
    if inclusion_mask is not None:
        # No filtering needed for length 1: both type (i) singletons and
        # type (ii) singletons are legitimate subdivision simplices.
        pass
    chains_by_length[1] = length_1

    # Longer chains: extend each chain by appending a flat index corresponding
    # to a proper superset of the last element. For the relative case, only
    # f_0 may be in L; all later f_i must be in M\\L.
    current = length_1
    length = 1
    while True:
        length += 1
        new_rows = []
        for row in current:
            last = row[-1]
            for candidate in parents_of[last]:
                if inclusion_mask is not None:
                    # Later chain elements must be in M\\L.
                    if inclusion_mask[candidate]:
                        continue
                new_rows.append(list(row) + [candidate])
        if not new_rows:
            break
        new_arr = np.array(new_rows, dtype=np.int32)
        chains_by_length[length] = new_arr
        current = new_arr

    return chains_by_length


def _all_proper_nonempty_subsets(verts):
    """ Yield all nonempty proper subsets of a set of vertices, as sorted lists. """
    verts = sorted(verts)
    n = len(verts)
    for k in range(1, n):
        for combo in combinations(verts, k):
            yield list(combo)


# --------------------------------------------------------------------------
# Tight-case helpers
# --------------------------------------------------------------------------

def _build_tight_mask(inclusion):
    """ Given an inclusion L ⊆ M, build the boolean mask over M.n_simplices
    marking the simplices of L ∪ L_out.

    L_out = { σ ∈ M : V(σ) ∩ V(L) = ∅ }.

    Returns
    -------
    mask : np.ndarray of bool, shape (M.n_simplices,)
    """
    M = inclusion.large
    L_vertex_set = set(int(v) for v in inclusion.vertex_map)

    mask = inclusion.simplex_mask.copy()  # starts with L
    # Add L_out: for each simplex of M, if its vertices are disjoint from V(L), mark it.
    for flat in range(M.n_simplices):
        if mask[flat]:
            continue
        sigma = M.simplex_by_flat_index(flat)
        if all(int(v) not in L_vertex_set for v in sigma):
            mask[flat] = True

    return mask


def _inclusion_from_mask(M, mask):
    """ Build an Inclusion from M and a boolean mask over M.n_simplices
    identifying the subcomplex. The small complex is materialized by
    collecting all marked simplices of dim >= 1; its vertices are those
    appearing in any marked simplex (plus all vertex-dim marked simplices).

    This is used internally by Subdivision.tight to build the L ∪ L_out
    inclusion.
    """
    # Identify subcomplex vertices: all M-vertices marked in the mask.
    # They live at flat indices 0 .. M.n_points - 1.
    vertex_flats = np.nonzero(mask[:M.n_points])[0]
    vertex_map = vertex_flats.astype(np.int32)

    # Collect higher-dim simplices of the subcomplex, translated to
    # subcomplex-local vertex indices.
    m_to_small = np.full(M.n_points, -1, dtype=np.int32)
    m_to_small[vertex_flats] = np.arange(len(vertex_flats), dtype=np.int32)

    small_simplices = []
    for flat in np.nonzero(mask)[0]:
        if flat < M.n_points:
            continue  # 0-simplex handled by vertex_map
        sigma = M.simplex_by_flat_index(flat)
        small_verts = [int(m_to_small[int(v)]) for v in sigma]
        small_simplices.append(sorted(small_verts))

    small_points = M.points[vertex_flats]
    small = SimplicialComplex(small_points, small_simplices)

    return Inclusion(small, M, vertex_map)