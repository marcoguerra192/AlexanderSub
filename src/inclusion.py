""" Inclusion class

Represents an inclusion L ⊆ M of simplicial complexes.

The contract is as follows:

1.
    L must be a subcomplex of M - every simplex of L, after applying
    the vertex map, must exist as a simplex of M. This is checked at
    construction (it would otherwise cause silent correctness bugs later).

2.
    None can be passed as the smaller subcomplex, this represents the
    inclusion (emptyset, M). Equivalently, can be called with the class
    method Inclusion.trivial(M)

3.
    The vertex map goes L -> M: `vertex_map[i]` is the index (into M.points)
    of the M-vertex corresponding to L-vertex i. It must be injective.
    This is assumed, not enforced. Will break if violated. 

4.
    A class method from_prefix is given that assumes the vertices of L are 
    the first L.n_points vertices of M. I.e. that the vertices of L are a 
    prefix of those of M. This is the case we use in both the Alpha construction
    (where the vertices are the same) and the CDT case. 

5.
    The inclusion pre-computes a boolean mask over M.n_simplices indicating
    which of M's simplices belong to L. Downstream code should prefer the
    mask for vectorized work and the `contains` method for single queries.

6.
    Instances are not mutable. Changing `small`, `large`, or `vertex_map`
    after construction will produce incorrect results.
"""

import numpy as np
from itertools import combinations
from .simplicial_complex import SimplicialComplex


class Inclusion:

    def __init__(self, small, large, vertex_map):
        """ Constructor

        Parameters:
        small : SimplicialComplex
            The subcomplex L, or None for the trivial inclusion.
        large : SimplicialComplex
            The ambient complex M.
        vertex_map : np.ndarray of int32, shape (small.n_points,)
            vertex_map[i] is the index in large.points of the M-vertex
            corresponding to L-vertex i.
        """
        self.small = small
        self.large = large
        self.vertex_map = np.asarray(vertex_map, dtype=np.int32)
        self.is_trivial = (small is None)

        # Build the simplex mask by translating every L-simplex through
        # the vertex map and looking up its flat index in M. Simultaneously
        # record the flat indices for the complement ops.
        self._simplex_mask = np.zeros(large.n_simplices, dtype=bool)

        if not self.is_trivial:
            # Dimension 0: L-vertex i lives in M at vertex_map[i].
            # Flat index of M-vertex v is just v (vertices come first in M's
            # flat ordering by SimplicialComplex convention)
            for v_in_M in self.vertex_map:
                self._simplex_mask[large.flat_from_dim_index(0, int(v_in_M))] = True
    
            # Dimensions >= 1: translate each L-simplex, sort, look up in M.
            for k in range(1, small.dim + 1):
                for L_simp in small.simplices_in_dim[k]:
                    m_simp = tuple(sorted(int(self.vertex_map[v]) for v in L_simp))
                    flat = large._simplex_to_flat[m_simp]   # raises if not present
                    self._simplex_mask[flat] = True

        self._l_flats_in_m = np.nonzero(self._simplex_mask)[0].astype(np.int32)

        # Cached M -> L inverse vertex map, -1 where M-vertex has no L counterpart.
        self._m_to_l_cache = None

    @classmethod
    def from_prefix(cls, small, large):
        """ Constructor for the usual case where L's vertices
        are the first small.n_points rows of large.points, in order.
        The vertex map is then trivially the identity.
        """
        if small is None:
            raise ValueError("This constructor does not accept an emtpy complex")
        vertex_map = np.arange(small.n_points, dtype=np.int32)
        return cls(small, large, vertex_map)

    @classmethod
    def trivial(cls, large):
        """ Trivial inclusion: no subcomplex is marked. Passing this to a
        relative-style constructor yields a plain barycentric subdivision.
        """
        return cls(None, large, np.empty(0, dtype=np.int32))

    # Derived data

    @property
    def simplex_mask(self):
        """ Boolean array of length large.n_simplices. True at flat index
        `f` iff the simplex of M at that flat index belongs to L.
        """
        return self._simplex_mask

    @property
    def l_flats_in_m(self):
        """ Int array of flat indices into M of the simplices of L. """
        return self._l_flats_in_m

    @property
    def m_to_l(self):
        """ Int32 array of length large.n_points. m_to_l[j] is the L-vertex
        index corresponding to M-vertex j, or -1 if M-vertex j is not in L.
        """
        if self._m_to_l_cache is None:
            result = np.full(self.large.n_points, -1, dtype=np.int32)
            if not self.is_trivial:
                result[self.vertex_map] = np.arange(self.small.n_points, dtype=np.int32)
            self._m_to_l_cache = result
        return self._m_to_l_cache

    # Queries

    def contains(self, m_simplex):
        """ Is the M-simplex `m_simplex` in L?
        `m_simplex` is a list/tuple/array of M-vertex indices.
        """
        m_key = tuple(sorted(int(v) for v in m_simplex))
        flat = self.large._simplex_to_flat.get(m_key)
        if flat is None:
            return False
        return bool(self._simplex_mask[flat])

    def contains_flat(self, flat_idx):
        """ O(1) version of `contains` when the flat index is already known. """
        return bool(self._simplex_mask[flat_idx])

    def complement_mask(self):
        """ Boolean array: True at flat index f iff the M-simplex at f is
        NOT in L. Computed on demand. """
        return ~self._simplex_mask

    def complement_flats(self):
        """ Int array of flat indices into M of the simplices NOT in L. """
        return np.nonzero(~self._simplex_mask)[0].astype(np.int32)

    # Printing / equality

    def __repr__(self):
        if self.is_trivial:
            return f"Inclusion.trivial(large={self.large!r})"
        return (f"Inclusion(small={self.small!r}, large={self.large!r}, "
                f"n_L_simplices={int(self._simplex_mask.sum())})")

    ## Computing the L_out subcomplex

    def Lout_mask(self):
        """ Boolean mask over self.large.n_simplices marking simplices of L_out
        (M-simplices whose vertices are entirely outside V(L)).
        """
        if self.is_trivial:
            # Vacuously: every simplex of M has all vertices outside V(L),
            # so L_out = M and it returns an all-True mask.
            # However, this 
            return np.ones(self.large.n_simplices, dtype=bool)
            
        L_vertex_set = set(int(v) for v in self.vertex_map)
        mask = np.zeros(self.large.n_simplices, dtype=bool)
        for flat in range(self.large.n_simplices):
            sigma = self.large.simplex_by_flat_index(flat)
            if all(int(v) not in L_vertex_set for v in sigma):
                mask[flat] = True
        return mask

    def L_union_Lout_in_M(self):
        """ 
        Return a new Inclusion representing (L ∪ L_out) ⊆ M,
        where L_out = {σ ∈ M : V(σ) ∩ V(L) = ∅}.
        """
        if self.is_trivial:
            # Lout is all of M, so the resulting inclusion is (M,M)
            return Inclusion(self.large, self.large, np.arange(self.large.n_points))

        M = self.large
        # the vertices of L, as seen in M
        L_vertex_set = set(int(v) for v in self.vertex_map)
    
        # Mark L-simplices and all M-simplices
        # whose vertices are disjoint from V(L).
        mask = self.simplex_mask.copy()
        
        for flat in range(M.n_simplices):
            if mask[flat]: # skip the simplices that were in L, obviously
                continue
            sigma = M.simplex_by_flat_index(flat) # simplex as a tuple of indices
            if all(int(v) not in L_vertex_set for v in sigma): # check not in L
                mask[flat] = True

        # Now mask indicates the simplices in L \cup L_out
        # Realize (L ∪ L_out) as a standalone SimplicialComplex.
        vertex_flats = np.nonzero(mask[:M.n_points])[0]
        vertex_map = vertex_flats.astype(np.int32)

        # map from m to the new subcomplex
        m_to_small = np.full(M.n_points, -1, dtype=np.int32)
        m_to_small[vertex_flats] = np.arange(len(vertex_flats), dtype=np.int32)
    
        small_simplices = []
        for flat in np.nonzero(mask)[0]:
            if flat < M.n_points: # these are vertices, we skip them
                continue
            sigma = M.simplex_by_flat_index(flat) # find the simplex as a tuple
            small_verts = [int(m_to_small[int(v)]) for v in sigma]
            small_simplices.append(sorted(small_verts))
    
        small_points = M.points[vertex_flats]
        small = SimplicialComplex(small_points, small_simplices)
    
        return Inclusion(small, M, vertex_map)

    def discard_boundary_components(self, parent_subdivision, components):
        """ Partition the components of `self.small` into those that touch the
        boundary of `parent_subdivision.parent` (= |M|) and those that don't.
        
        `self` is expected to be the supplement inclusion: small ⊆ materialized
        subdivision, where `parent_subdivision` is the Subdivision whose
        materialized complex is self.large.
        
        Parameters
        ----------
        parent_subdivision : Subdivision
            The subdivision that produced self.large; needed to look up vertex
            provenance.
        components : list of np.ndarray
            Connected components of self.small, as returned by
            self.small.connected_components().
        
        Returns
        -------
        valid : list of np.ndarray
            Components that do NOT touch the boundary of |M|.
        excluded : list of np.ndarray
            Components that touch the boundary.
        """
        M = parent_subdivision.parent
        provenance = parent_subdivision.materialized_provenance
    
        # Compute the boundary subcomplex of M as a set of flat indices.
        boundary_codim1 = M.boundary()  # codim-1 simplices on ∂|M|
        boundary_M_flats = set()
        boundary_M_vertices = set()
        for face in boundary_codim1:
            face_sorted = tuple(sorted(int(v) for v in face))
            # The face itself
            boundary_M_flats.add(M._simplex_to_flat[face_sorted])
            # Its vertices
            for v in face_sorted:
                boundary_M_vertices.add(v)
                boundary_M_flats.add(M.flat_from_dim_index(0, v))
    
        # Translate the supplement's vertex indices into the materialized 
        # subdivision's vertex indices.
        supp_to_materialized = self.vertex_map  # length self.small.n_points
    
        # Mark which supplement-vertices are on ∂|M| via provenance.
        is_boundary_supp_vertex = np.zeros(self.small.n_points, dtype=bool)
        for supp_v in range(self.small.n_points):
            materialized_v = int(supp_to_materialized[supp_v])
            kind, ident = provenance[materialized_v]
            if kind == 'parent_vertex':
                if int(ident) in boundary_M_vertices:
                    is_boundary_supp_vertex[supp_v] = True
            elif kind == 'barycenter_of':
                if int(ident) in boundary_M_flats:
                    is_boundary_supp_vertex[supp_v] = True
            else:
                raise RuntimeError(f"Unknown provenance kind: {kind}")
    
        # Partition components.
        valid = []
        excluded = []
        for comp in components:
            if any(is_boundary_supp_vertex[int(v)] for v in comp):
                excluded.append(comp)
            else:
                valid.append(comp)
    
        return valid, excluded

        
    def complement_complex(self):
        """ Build the closure of the top-dim simplices of M not in L, as an
        Inclusion(complement_complex ⊆ M).
        
        The complement complex is the smallest subcomplex of M containing every
        top-dim simplex of M that is not in L. By construction it is closed
        under taking faces.
        
        For the manifold case (L is at most codim-1 in M), this is the natural
        "set-difference" complex used for the cheap (no-subdivision) pipeline.
        
        Returns
        -------
        Inclusion(complement_complex ⊆ self.large)
        """
        M = self.large
        top_dim = M.dim
        
        # Top-dim M-simplices NOT in L.
        top_simplices_not_in_L = []
        for row in M.simplices_in_dim[top_dim]:
            m_simp = tuple(sorted(int(v) for v in row))
            flat = M._simplex_to_flat[m_simp]
            if not self._simplex_mask[flat]:
                top_simplices_not_in_L.append(m_simp)
        
        # Closure: every face of every top-simplex.
        closure_faces = set()
        for sigma in top_simplices_not_in_L:
            closure_faces.add(sigma)
            for k in range(1, len(sigma)):
                for face in combinations(sigma, k):
                    closure_faces.add(face)
        
        # Vertices appearing in the closure.
        closure_vertices = sorted({int(v) for face in closure_faces for v in face})
        
        # Build the small complex: reindex M's vertices to local indices.
        m_to_local = {int(v): i for i, v in enumerate(closure_vertices)}
        
        small_simplices = []
        for face in closure_faces:
            if len(face) >= 2:  # 0-simplices implicit in points
                small_simplices.append(sorted(m_to_local[int(v)] for v in face))
        
        small_points = M.points[np.array(closure_vertices, dtype=np.int32)]
        small = SimplicialComplex(small_points, small_simplices)
        
        vertex_map = np.array(closure_vertices, dtype=np.int32)
        return Inclusion(small, M, vertex_map)
