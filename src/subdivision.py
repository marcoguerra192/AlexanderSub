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

    def __init__(self, parent, inclusion, materialized, materialized_provenance,
                 chains_by_length):
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
        return cls.from_inclusion(L_union_Lout_in_M)

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
