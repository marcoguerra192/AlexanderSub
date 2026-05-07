""" Subdivision class

Represents a subdivision of a simplicial complex M. For now it stores two
representations:

1. `materialized` (SimplicialComplex): the geometric realization, built by
   translating the legacy `subdivision()` / `RelSubGeneral()` functions to
   the class-based inputs. This is the known-good representation.

2. `chains_by_length` (dict[int, np.ndarray]): the combinatorial "view on
   the parent" representation. Each entry is a (n_chains, length) int32
   array of flat indices into the parent M, each row a strictly-ascending
   chain by parent dimension.

Both are computed at construction. This is to check the transition betwen
the two. The method `verify_consistent()` compares
the two and returns a report; in the long run, I'd like to deprecate the 
materialized one and only compute barycenters lazily if needed. 

Two constructors, all taking an Inclusion:

  Subdivision.from_inclusion(incl)    — builds (M, L)' using the inclusion.
                                        If incl is trivial, this is the plain
                                        barycentric subdivision M'.
  Subdivision.tight(incl)             — builds (M, L ∪ L_out)'. Internally
                                        computes L_out and forms the
                                        L ∪ L_out inclusion.

Convenience aliases:
  Subdivision.barycentric(M)   = Subdivision.from_inclusion(Inclusion.trivial(M))
  Subdivision.relative(incl)   = Subdivision.from_inclusion(incl)

The `subdivision` and `RelSubGeneral` functions are the old ones. Thin 
adapters convert class-based inputs to the legacy list-of-lists format 
before calling them.
"""

import numpy as np
from itertools import combinations

from .simplicial_complex import SimplicialComplex
from .inclusion import Inclusion
from .abstract_subdivision import AbstractSubdivision

# Legacy algorithms live in the old src_old package.
from src_old.relative_subdivision import subdivision as _legacy_subdivision
from src_old.relative_subdivision import RelSubGeneral as _legacy_relsubgeneral

class Subdivision:

    def __init__(self, parent, inclusion, base_inclusion, materialized,
             materialized_provenance, chains_by_length):
        """ Direct constructor. Best to use the classmethods instead.

        Parameters
        ----------
        parent : SimplicialComplex
            The complex M being subdivided.
        inclusion : Inclusion
            The inclusion L ⊆ M used to build this subdivision. Set it
            to trivial for a pure barycentric subdivision.
        materialized : SimplicialComplex
            The geometric realization produced by the legacy algorithm.
        materialized_provenance : dict[int, tuple]
            Maps each vertex of `materialized` to a tuple describing its
            origin: ('parent_vertex', i) for an original vertex of M at
            index i, or ('barycenter_of', flat) for a barycenter of the
            M-simplex at flat index `flat`.
        chains_by_length : dict[int, np.ndarray]
            Combinatorial representation: for each chain length ≥ 1, a
            (n_chains, length) int32 array of flat indices into parent.
        """
        self.parent = parent
        self.inclusion = inclusion
        self.base_inclusion = base_inclusion
        self.materialized = materialized
        self.materialized_provenance = materialized_provenance
        self.chains_by_length = chains_by_length

    # Constructors

    @classmethod
    def from_inclusion(cls, inclusion):
        """ Build (M, L)' from the given inclusion.

        If inclusion is trivial, produces the barycentric subdivision M'.
        Otherwise, produces the relative subdivision (M, L)'.
        """
        M = inclusion.large
        Subs = _build_Subs(M.dim)

        # Legacy-format inputs.
        M_legacy = _complex_to_legacy_simplices(M)

        # Compute the materialized subdivision via the legacy code.
        if inclusion.is_trivial:
            # Pure barycentric: call the legacy `subdivision` function.
            legacy_simplices, legacy_points = _legacy_subdivision(
                M_legacy, M.points, Subs
            )
            materialized, provenance = _build_materialized_from_legacy(
                legacy_simplices, legacy_points, M, inclusion=None
            )
        else:
            # Relative: call RelSubGeneral with K = L (as seen from M).
            K_legacy, K_vertex_indices = _subcomplex_from_inclusion(inclusion)
            legacy_simplices, legacy_points = _legacy_relsubgeneral(
                M_legacy, M.points, K_legacy, K_vertex_indices, Subs
            )
            materialized, provenance = _build_materialized_from_legacy(
                legacy_simplices, legacy_points, M, inclusion=inclusion
            )

        # Compute the combinatorial chain representation independently.
        chains = _build_chains(M, inclusion_mask=inclusion.simplex_mask)

        return cls(
            parent=M,
            inclusion=inclusion,
            base_inclusion=inclusion, 
            materialized=materialized,
            materialized_provenance=provenance,
            chains_by_length=chains,
        )

    @classmethod
    def barycentric(cls, M):
        """ Pure barycentric subdivision M'. """
        return cls.from_inclusion(Inclusion.trivial(M))

    @classmethod
    def relative(cls, inclusion):
        """ Relative subdivision (M, L)'. Alias for from_inclusion. """
        return cls.from_inclusion(inclusion)

    @classmethod
    def tight(cls, inclusion):
        """ Tight subdivision (M, L ∪ L_out)'.

        inclusion computes L_out = {σ ∈ M : V(σ) ∩ V(L) = ∅}, forms the
        inclusion L ∪ L_out ⊆ M, and subdivides relative to that.
        """
        # if inclusion.is_trivial:
        #     raise ValueError(
        #         "Subdivision.tight requires a non-trivial inclusion "
        #     )
        L_union_Lout_in_M = inclusion.L_union_Lout_in_M()
        sub = cls.from_inclusion(L_union_Lout_in_M)
        sub.base_inclusion = inclusion 
        return sub

    # Introspection

    @property
    def n_chains(self):
        """ Total number of chains across all lengths. """
        return sum(arr.shape[0] for arr in self.chains_by_length.values())

    def __repr__(self):
        counts = {length: arr.shape[0]
                  for length, arr in self.chains_by_length.items()}
        return (f"Subdivision(parent={self.parent!r}, "
                f"inclusion={self.inclusion!r}, "
                f"n_materialized_simplices={self.materialized.n_simplices}, "
                f"chain_counts={counts})")

    # Materialization (for now, just return the cached materialized complex)

    def to_simplicial_complex(self):
        """ Return the materialized geometric realization.

        In the current implementation this just returns self.materialized.
        Later, once we trust the chain-based representation, this will be
        reimplemented to build a SimplicialComplex from chains directly.
        """
        return self.materialized

    ####
    # Computation of the supplement in abstract. Works for all cases.
    ###

    def supplement(self):
        """ The supplement of L in self.materialized.
        
        The supplement is the subcomplex of the materialized subdivision M'
        consisting of simplices whose vertices have no L-provenance, where:
          - 'L-provenance' means provenance ('parent_vertex', v) with v in V(L),
            OR provenance ('barycenter_of', f) with f a flat index of an L-simplex.
        
        The L referred to here is self.base_inclusion's small complex. For
        barycentric and relative subdivisions, base_inclusion == inclusion.
        For tight subdivisions, base_inclusion is the original L ⊆ M (NOT the
        L ∪ L_out used internally for the relative subdivision step).
        
        Returns
        -------
        Inclusion
            An inclusion (supplement ⊆ self.materialized).
        """
        base = self.base_inclusion
        L_vertex_set = set(int(v) for v in base.vertex_map)
        L_mask = base.simplex_mask  # boolean array over self.parent.n_simplices
    
        # Determine which vertices of materialized are "L-vertices".
        materialized = self.materialized
        is_L_vertex = np.zeros(materialized.n_points, dtype=bool)
        for i in range(materialized.n_points):
            kind, ident = self.materialized_provenance[i]
            if kind == 'parent_vertex':
                if int(ident) in L_vertex_set:
                    is_L_vertex[i] = True
            elif kind == 'barycenter_of':
                if L_mask[int(ident)]:
                    is_L_vertex[i] = True
            else:
                raise RuntimeError(f"Unknown provenance kind: {kind}")
    
        # The supplement's vertices are those NOT marked as L.
        supplement_vertex_indices = np.nonzero(~is_L_vertex)[0].astype(np.int32)
        supp_old_to_new = np.full(materialized.n_points, -1, dtype=np.int32)
        supp_old_to_new[supplement_vertex_indices] = np.arange(
            len(supplement_vertex_indices), dtype=np.int32
        )
    
        # Collect supplement simplices: those whose vertices are all non-L.
        supp_simplices = []
        for k in range(1, materialized.dim + 1):
            for row in materialized.simplices_in_dim[k]:
                if all(not is_L_vertex[int(v)] for v in row):
                    supp_simplices.append(sorted(int(supp_old_to_new[int(v)]) for v in row))
    
        # Materialize the supplement as a standalone SimplicialComplex.
        supp_points = materialized.points[supplement_vertex_indices]
        supp_complex = SimplicialComplex(supp_points, supp_simplices)
    
        # Inclusion: supp_complex ⊆ materialized, with vertex_map = supplement_vertex_indices.
        return Inclusion(supp_complex, materialized, supplement_vertex_indices)


    def representative_cycles(self, valid_components, supplement_inclusion):
        """ Extract representative cycles in L from valid components of the
        supplement.
        
        For each valid component, collects the top-dim simplices of self.parent
        (= M) "associated" with the component, takes their Z/2 boundary, and
        restricts to L's codim-1 simplices. The result is a Z/2 cycle in L for
        each component.
        
        Two kinds of associated top-dim M-simplices:
          (i)  M-triangles in M\\K whose barycenter (a vertex of self.materialized
               with provenance ('barycenter_of', f)) belongs to the component.
          (ii) M-triangles entirely in K (i.e., in L_out for the tight case)
               whose vertices appear in the component as ('parent_vertex', v).
               For tight subdivisions this means L_out triangles; for barycentric
               and relative this case is empty (M\\K = M\\L there has no triangles
               in K=L since L is at most 1-dimensional in our examples — but the
               code handles it generically).
        
        Parameters
        ----------
        valid_components : list of np.ndarray
            Components of the supplement, in supplement-vertex indices.
        supplement_inclusion : Inclusion
            The supplement-as-inclusion, returned by self.supplement().
        
        Returns
        -------
        cycles : list of list of tuple
            For each component, the list of L-edges (as sorted tuples of
            L-vertex indices) that form its representative cycle. Edges are
            in L's vertex indexing (translated through base_inclusion.vertex_map).
        """
        M = self.parent
        base = self.base_inclusion
        provenance = self.materialized_provenance
    
        # L-edges as a set, in M's vertex indexing.
        L_edge_flats_in_M = set()
        if not base.is_trivial:
            L = base.small
            if L.dim >= 1:
                for L_edge in L.simplices_in_dim[1]:
                    m_edge = tuple(sorted(int(base.vertex_map[v]) for v in L_edge))
                    L_edge_flats_in_M.add(M._simplex_to_flat[m_edge])
    
        # M-to-L vertex map for translating cycle edges back to L's vertex space.
        m_to_L = base.m_to_l  # length M.n_points; -1 where M-vertex isn't in L
    
        # K-mask: simplices marked as "in the subcomplex used for the relative
        # subdivision". This determines which top-dim M-simplices are in M\K
        # (their barycenters appear in the materialized subdivision) vs in K
        # (they appear verbatim, contributing parent_vertex entries).
        K_mask = self.inclusion.simplex_mask  # NOT base.simplex_mask
    
        # Build adjacency: from supplement-vertex index -> provenance.
        supp_to_provenance = {
            supp_v: provenance[int(supplement_inclusion.vertex_map[supp_v])]
            for supp_v in range(supplement_inclusion.small.n_points)
        }
    
        cycles = []
    
        for comp in valid_components:
            comp_set = set(int(v) for v in comp)
    
            # Collect associated top-dim M-simplices.
            repr_top_simplices = set()  # set of flat indices in M
    
            # Type (i): M\K top-dim simplices whose barycenter is in this component.
            for supp_v in comp:
                kind, ident = supp_to_provenance[int(supp_v)]
                if kind == 'barycenter_of':
                    f = int(ident)
                    # Only top-dim M-simplices contribute to the cycle's boundary.
                    # (Barycenters of edges / lower-dim sit on the supplement but
                    # their "boundary" is just their own vertices, which aren't
                    # L-edges. Skip.)
                    if M._flat_dim[f] == M.dim:
                        repr_top_simplices.add(f)
    
            # Type (ii): K-side top-dim M-simplices wholly in this component.
            # Triangle T is in K iff K_mask[flat_of_T]; it's in this component iff
            # any of its vertices (translated to supplement-vertex indices via the
            # supplement_inclusion's m_to_l) is in comp_set. Since T is connected,
            # checking the first vertex suffices.
            for f in range(M.n_simplices):
                if M._flat_dim[f] != M.dim:
                    continue
                if not K_mask[f]:
                    continue
                sigma = M.simplex_by_flat_index(f)
                # Look up sigma's first vertex in the materialized complex.
                # Since K-simplices are added verbatim, their vertices appear as
                # ('parent_vertex', v) in the provenance.
                v0 = int(sigma[0])
                # Find the supplement-vertex with provenance ('parent_vertex', v0).
                # If v0 is in V(L), it won't be in the supplement (filtered out).
                # If v0 is in V(L_out), it will be.
                # Use supplement_inclusion's m_to_l, which goes
                # (materialized vertex) -> (supplement vertex) or -1.
                #
                # First, find materialized index of vertex v0:
                # It's a parent_vertex entry, so its materialized index = v0
                # (by convention in the materialized output).
                materialized_v0 = v0
                supp_v0 = int(supplement_inclusion.m_to_l[materialized_v0])
                if supp_v0 == -1:
                    continue  # not in supplement (= in V(L))
                if supp_v0 in comp_set:
                    repr_top_simplices.add(f)
    
            # Take Z/2 boundary, restrict to L-edges.
            boundary_count = {}
            for f in repr_top_simplices:
                sigma = M.simplex_by_flat_index(f)
                sigma_sorted = sorted(int(v) for v in sigma)
                # All codim-1 faces of sigma (= edges, for triangles)
                for i in range(len(sigma_sorted)):
                    face = tuple(sigma_sorted[:i] + sigma_sorted[i+1:])
                    boundary_count[face] = boundary_count.get(face, 0) + 1
    
            cycle_edges_in_M = [face for face, cnt in boundary_count.items() if cnt % 2 == 1 ]

            # Sanity: by Alexander duality, every cycle edge must be in L.
            for face in cycle_edges_in_M:
                assert M._simplex_to_flat[face] in L_edge_flats_in_M, \
                    f"Cycle edge {face} is not in L — Alexander duality violated, bug upstream"
    
            # Translate cycle edges back to L's vertex indexing.
            cycle_edges_in_L = []
            for m_edge in cycle_edges_in_M:
                l_edge = tuple(sorted(int(m_to_L[v]) for v in m_edge))
                cycle_edges_in_L.append(l_edge)
    
            cycles.append(cycle_edges_in_L)
    
        return cycles

    # ----------------------------------------------------------------------
    # Consistency check between the two representations
    # ----------------------------------------------------------------------

    def verify_consistent(self):
        """ Check that the materialized representation matches what the
        chain representation would produce.

        Returns
        -------
        ConsistencyReport
            An object describing the comparison. Check `.is_consistent`
            for a single bool; inspect fields for details on mismatches.
        """
        report = ConsistencyReport()

        # Realize the chain representation as a set of simplices-as-sorted-tuples
        # in terms of the same provenance scheme used by `materialized`.
        chain_simplices = _realize_chains_as_provenance_simplices(
            chains_by_length=self.chains_by_length,
            parent=self.parent,
            inclusion=self.inclusion,
        )

        # Express the materialized complex's simplices in provenance terms too.
        materialized_simplices = _materialized_as_provenance_simplices(
            self.materialized, self.materialized_provenance
        )

        chain_set = set(chain_simplices)
        mat_set = set(materialized_simplices)

        report.chain_only = sorted(chain_set - mat_set)
        report.materialized_only = sorted(mat_set - chain_set)
        report.common = sorted(chain_set & mat_set)
        report.n_chain = len(chain_set)
        report.n_materialized = len(mat_set)
        report.is_consistent = (
            len(report.chain_only) == 0 and len(report.materialized_only) == 0
        )

        return report


class ConsistencyReport:
    """ Result of Subdivision.verify_consistent(). """

    def __init__(self):
        self.is_consistent = None
        self.n_chain = 0
        self.n_materialized = 0
        self.chain_only = []         # simplices (as provenance tuples) only in chain repr
        self.materialized_only = []  # simplices only in materialized repr
        self.common = []

    def __repr__(self):
        status = "CONSISTENT" if self.is_consistent else "MISMATCH"
        return (f"ConsistencyReport({status}: "
                f"n_chain={self.n_chain}, n_materialized={self.n_materialized}, "
                f"common={len(self.common)}, "
                f"chain_only={len(self.chain_only)}, "
                f"materialized_only={len(self.materialized_only)})")

    def summary(self):
        """ Print a human-readable diff. """
        print(self)
        if self.chain_only:
            print(f"  Simplices only in chain representation ({len(self.chain_only)}):")
            for s in self.chain_only[:10]:
                print(f"    {s}")
            if len(self.chain_only) > 10:
                print(f"    ... and {len(self.chain_only) - 10} more")
        if self.materialized_only:
            print(f"  Simplices only in materialized representation ({len(self.materialized_only)}):")
            for s in self.materialized_only[:10]:
                print(f"    {s}")
            if len(self.materialized_only) > 10:
                print(f"    ... and {len(self.materialized_only) - 10} more")


# Helpers

def _build_Subs(dim):
    """ Build the Subs dict expected by the old algorithms.
    Keys: simplex sizes (2 for edges, 3 for triangles, ...).
    """
    # Legacy code indexes Subs by simplex SIZE (len(sigma)), not dimension.
    # For complexes up to dimension `dim`, sizes go from 2 to dim+1.
    # A few extra entries don't hurt; the legacy code looks up Subs[l] where
    # l = len(sigma) for sigma in M, and l is always >= 2 after filtering.
    # We mirror the pattern in old driver code: range(2, 5).
    max_size = max(dim + 1, 4)  # ensure Subs[3] always exists for triangles
    return {i: AbstractSubdivision(i) for i in range(2, max_size + 1)}


def _complex_to_legacy_simplices(K):
    """ Convert a SimplicialComplex to the old list-of-sorted-lists format.
    Only simplices of dim >= 1 are included (the old code expects no 0-simplices).
    """
    out = []
    for k in range(1, K.dim + 1):
        for row in K.simplices_in_dim[k]:
            out.append(sorted(int(v) for v in row))
    return out


def _subcomplex_from_inclusion(inclusion):
    """ Given a (non-trivial) Inclusion L ⊆ M, return (K_legacy, K_vertex_indices)
    in the format expected by the old RelSubGeneral.

    K_legacy is a list of sorted lists of M-vertex indices (every L-simplex,
    translated through the vertex map).
    K_vertex_indices is a list of M-vertex indices that are L-vertices.
    """
    assert not inclusion.is_trivial

    M = inclusion.large
    L = inclusion.small
    vmap = inclusion.vertex_map

    K_legacy = []
    # dim 0: L-vertices as singletons in M coords
    for v in vmap:
        K_legacy.append([int(v)])
    # dims >= 1: translate L-simplices through vmap
    for k in range(1, L.dim + 1):
        for row in L.simplices_in_dim[k]:
            K_legacy.append(sorted(int(vmap[v]) for v in row))

    K_vertex_indices = sorted(int(v) for v in vmap)
    return K_legacy, K_vertex_indices


def _build_L_union_Lout_inclusion(inclusion):
    pass

# ---------------- materialized output + provenance --------------------------

def _build_materialized_from_legacy(legacy_simplices, legacy_points, M, inclusion):
    """ Convert the legacy algorithm's output into (SimplicialComplex, provenance).

    The legacy output convention:
      - legacy_points[0 : M.n_points] == M.points  (original vertices)
      - legacy_points[M.n_points : ]  == barycenters of M-simplices in some order.

    For the barycentric case (subdivision): barycenters correspond to
    *all* simplices of M in the order they appear in M_legacy. Since
    M_legacy = _complex_to_legacy_simplices(M), that order is
    dim 1 simplices, then dim 2 simplices, etc.

    For the relative case (RelSubGeneral): barycenters correspond to
    simplices in M \\ K (dim >= 1) in the order they appear in M_legacy.

    We recover this by walking M_legacy and matching barycenters to simplex
    flat indices.
    """
    n_M = M.n_points
    n_bary = legacy_points.shape[0] - n_M

    # Determine which M-simplices the barycenters correspond to.
    # M_legacy lists simplices of dim >= 1 in a fixed order; the legacy code
    # filters out simplices in K when building the barycenter list (RelSubGeneral
    # only).
    M_legacy = _complex_to_legacy_simplices(M)

    if inclusion is None or inclusion.is_trivial:
        # Barycentric: all simplices of dim >= 1 get a barycenter, in order.
        bary_source_simplices = M_legacy
    else:
        # Relative: only simplices not in K get barycenters.
        K_faces = set()
        for flat in np.nonzero(inclusion.simplex_mask)[0]:
            K_faces.add(tuple(sorted(int(v) for v in M.simplex_by_flat_index(flat))))
        bary_source_simplices = [
            s for s in M_legacy if tuple(sorted(s)) not in K_faces
        ]

    assert len(bary_source_simplices) == n_bary, (
        f"Barycenter count mismatch: legacy produced {n_bary}, "
        f"expected {len(bary_source_simplices)} from M \\ K."
    )

    # Build provenance: vertex i in legacy output corresponds to
    #   ('parent_vertex', i)                      if i < n_M
    #   ('barycenter_of', flat_of_bary_source[i - n_M])  otherwise
    provenance = {}
    for i in range(n_M):
        provenance[i] = ('parent_vertex', i)
    for j, s in enumerate(bary_source_simplices):
        flat = M._simplex_to_flat[tuple(sorted(s))]
        provenance[n_M + j] = ('barycenter_of', flat)

    # Build the materialized SimplicialComplex. Legacy output simplices may
    # include 0-simplices (vertex lists of length 1); SimplicialComplex gets
    # its 0-simplices from `points`, so we filter those out.
    materialized_simplices = [
        sorted(int(v) for v in s) for s in legacy_simplices if len(s) >= 2
    ]

    materialized = SimplicialComplex(legacy_points, materialized_simplices)
    return materialized, provenance


# ---------------- chain-based construction ---------------------------------

def _build_chains(M, inclusion_mask):
    """ Build all chains of simplices of M with strictly increasing dimension.

    If `inclusion_mask` is None or all-False: all chains are pure type-(i)
    (no L involvement).

    Otherwise, marks which simplices are in L; chains are filtered per
    Maunder's classification.
    """
    n_simp = M.n_simplices

    # Precompute proper-parent relation.
    parents_of = [[] for _ in range(n_simp)]
    for parent_flat in range(n_simp):
        parent_verts = set(int(v) for v in M.simplex_by_flat_index(parent_flat))
        if len(parent_verts) <= 1:
            continue
        for sub in _all_proper_nonempty_subsets(parent_verts):
            sub_tuple = tuple(sorted(sub))
            child_flat = M._simplex_to_flat.get(sub_tuple)
            if child_flat is not None:
                parents_of[child_flat].append(parent_flat)

    chains_by_length = {}

    # Length-1 chains: every simplex.
    length_1 = np.arange(n_simp, dtype=np.int32).reshape(-1, 1)
    chains_by_length[1] = length_1

    # Extend iteratively. Rule for the relative case:
    # - first element f_0 may be in L or M\L (either a type-(ii) seed or
    #   a type-(i)/type-(iii) seed)
    # - all subsequent elements must be in M\L
    current = length_1
    length = 1
    while True:
        length += 1
        new_rows = []
        for row in current:
            last = int(row[-1])
            for candidate in parents_of[last]:
                if inclusion_mask is not None and inclusion_mask[candidate]:
                    # Cannot extend into an L-simplex.
                    continue
                new_rows.append(list(int(x) for x in row) + [int(candidate)])
        if not new_rows:
            break
        new_arr = np.array(new_rows, dtype=np.int32)
        chains_by_length[length] = new_arr
        current = new_arr

    return chains_by_length


def _all_proper_nonempty_subsets(verts):
    verts = sorted(verts)
    n = len(verts)
    for k in range(1, n):
        for combo in combinations(verts, k):
            yield list(combo)


# ---------------- consistency comparison ------------------------------------

def _realize_chains_as_provenance_simplices(chains_by_length, parent, inclusion):
    out = []
    M = parent
    mask = inclusion.simplex_mask if inclusion is not None else None

    def _contribution(f):
        """Provenance keys contributed by a single chain element."""
        sigma = M.simplex_by_flat_index(f)
        is_zero_dim = len(sigma) == 1
        is_in_L = (mask is not None) and bool(mask[f])
        if is_zero_dim or is_in_L:
            return [('parent_vertex', int(v)) for v in sigma]
        else:
            return [('barycenter_of', int(f))]

    for length, chains in sorted(chains_by_length.items()):
        for chain in chains:
            verts = []
            for f in chain:
                verts.extend(_contribution(int(f)))
            out.append(tuple(sorted(set(verts))))

    return out


def _materialized_as_provenance_simplices(materialized, provenance):
    """ Express each simplex of `materialized` as a sorted tuple of provenance
    keys (using the provenance dict).
    """
    out = []
    # Include 0-simplices derived from points (materialized doesn't store them
    # explicitly; they're implicit in materialized.points). Each point's
    # provenance is a single-vertex "simplex".
    for i in range(materialized.n_points):
        out.append((provenance[i],))

    for k in range(1, materialized.dim + 1):
        for row in materialized.simplices_in_dim[k]:
            out.append(tuple(sorted(provenance[int(v)] for v in row)))

    return out
