''' Simplicial complex class

The contract is as follows:

1.
    The simplex list must be closed by subsets. The class does not accept
    top-dimensional representations. If not enforced, it will crash.

2.
    0-simplices in simplices are immaterial: points determines the existence
    of 0-simplices: if a point is never referenced in simplices, it will 
    nonetheless be included, and the ordering will reflect this.  

3.
    We are assuming that all simplicial complexes can be embedded in the
    d-disk. Therefore, the dimension of every simplicial complex cannot 
    exceed the ambient dimension. If any simplex is passed that exceeds
    the ambient dimension, it will be ignored.
    Currently, only dimension 2 and 3 are implemented. 
    
4. 
    Simplices are to be passed as a list of sorted lists. This is assumed
    true and not checked. If not enforced, it can fail or crash. The same
    is assumed when accessing objects.
   
'''

import numpy as np
from itertools import combinations

class SimplicialComplex:

    def __init__(self, points, simplices ):

        ''' Constructor

        Parameters:
        points: np.array of floats, shape N x d, with d=1,2 or 3
        simplices : list of sorted lists.
        '''

        if not isinstance(points, np.ndarray):
            raise ValueError("points must be a numpy array")

        if not np.issubdtype(points.dtype, np.floating):
            raise ValueError("points must be a numpy array of floats")

        self.ambient_dim = points.shape[1]
        
        if self.ambient_dim not in [2,3]:
            raise ValueError("points must be N x d with d=2,3")

        # store the points coordinates -> these fix the vertices positions
        self.points = points
        self.n_points = points.shape[0]

        if self.n_points == 0:
            raise ValueError("Empty complex - error")

        # processing the simplices
        
        if not isinstance(simplices, list):
            raise ValueError("simplices must be a list of sorted lists")

        if not all([ isinstance(l, list) for l in simplices]):
            raise ValueError("simplices must be a list of sorted lists")

        self._simplices_by_dim = {}
        self._cumulative = [0]

        

        # treat 0-simplices separately
        self._simplices_by_dim[0] = np.arange(self.n_points, dtype=np.int32).reshape(-1, 1)
        self._cumulative.append(self.n_points)

        # CHECK FOR DEBUGGING, but then can remove
        max_dim = max((len(s) for s in simplices), default=1) - 1
        assert max_dim <= self.ambient_dim

        # dimension of the complex - increased as simplices are read
        self.dim = 0

        
        # divide by dimension
        for k in range(1, self.ambient_dim + 1):
            simp_of_dim_k = [ s for s in simplices if len(s) == k+1 ]
            if simp_of_dim_k:
                self._simplices_by_dim[k] = np.array(simp_of_dim_k, dtype=np.int32)
                self.dim = k
            else:
                self._simplices_by_dim[k] = np.empty((0,k+1), dtype=np.int32)
                
            self._cumulative.append(self._cumulative[-1] + len(simp_of_dim_k))

        # Store total simplices
        self.n_simplices = self._cumulative[-1]

        ### LOOKUP dictionaries - Check whether it's worth making O(dim)
        ## instead of O(1), with a large memory saving

        # _flat_dim[i] gives the dimension of the flat index i
        self._flat_dim = np.concatenate([
            np.full(len(self._simplices_by_dim[k]), k, dtype=np.int32)
            for k in sorted(self._simplices_by_dim)
        ])

        # _flat_within[i] is the index of sigma of flat index i, within 
        # the list of appropriate dimension
        self._flat_within = np.concatenate([
            np.arange(len(self._simplices_by_dim[k]), dtype=np.int32)
            for k in sorted(self._simplices_by_dim)
        ])

        # map from a list of vertex indices to flat index
        self._simplex_to_flat = {}
        flat = 0
        for k in sorted(self._simplices_by_dim):
            for row in self._simplices_by_dim[k]:
                self._simplex_to_flat[tuple(row)] = flat
                flat += 1

    @property
    def simplices_in_dim(self):
        return _DimView(self)
    
    def flat_index_by_simplex(self, simplex):
        '''
        simplex is a list of ints, each a vertex.
        returns the global flat index
        '''
        return self._simplex_to_flat[tuple(simplex)]
        
    def flat_from_dim_index(self, dim, idx):
        '''
        dim is the dimension, idx the index within that dimension
        returns the global flat index
        '''
        return self._cumulative[dim] + idx
        
    def simplex_by_flat_index(self, idx):
        '''
        idx is the global flat index. 
        returns the simplex as a list of ints, each a vertex
        '''
        dim = self._flat_dim[idx]
        index = self._flat_within[idx]

        return self._simplices_by_dim[dim][index]

    def connected_components(self):
        """ Compute the connected components of the 1-skeleton.
        
        Every vertex of the complex is initially its own component; edges merge
        components. Isolated vertices (no incident edges) end up as singleton
        components.
        
        Returns
        -------
        components : list of np.ndarray
            Each entry is an int32 array of vertex indices forming one component,
            sorted in ascending order. The list itself is sorted by smallest
            vertex in each component.
        """
        n = self.n_points
        parent = np.arange(n, dtype=np.int32)
        
        def find(v):
            # Iterative path compression
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
        
        # Walk the 1-skeleton (only edges, not higher-dim simplices, are needed).
        if self.dim >= 1:
            edges = self.simplices_in_dim[1]
            for e in edges:
                union(int(e[0]), int(e[1]))
        
        # Group vertices by root.
        roots = np.array([find(v) for v in range(n)], dtype=np.int32)
        by_root = {}
        for v, r in enumerate(roots):
            by_root.setdefault(int(r), []).append(v)
        
        components = [np.array(sorted(verts), dtype=np.int32)
                      for verts in by_root.values()]
        components.sort(key=lambda arr: int(arr[0]))
        return components

    def boundary(self):
        """ The boundary of the complex as a Z/2 chain on codim-1 simplices.
        
        Computes which codim-1 simplices have an odd number of incident top-dim
        simplices. For a triangulation of a d-disk, this returns the codim-1
        simplices on ∂|M|.
        
        For a 2D complex (top-dim = triangles), returns boundary edges.
        For a 1D complex (top-dim = edges), returns boundary vertices.
        For a complex with no top-dim simplices, returns an empty array.
        
        Returns
        -------
        boundary_simplices : np.ndarray of int32, shape (n_boundary, top_dim)
            The codim-1 simplices on the boundary, each as a sorted row of
            vertex indices. Empty array of shape (0, top_dim) if no boundary.
        """
        top_dim = self.dim
        if top_dim < 1:
            # No top-dim simplices to take boundary of.
            return np.empty((0, max(top_dim, 1)), dtype=np.int32)
        
        codim1 = top_dim - 1
        top_simplices = self.simplices_in_dim[top_dim]
        
        if codim1 == 0:
            # Top simplices are edges; codim-1 simplices are vertices.
            # Each vertex's count = number of incident edges.
            counts = np.zeros(self.n_points, dtype=np.int64)
            for e in top_simplices:
                counts[int(e[0])] += 1
                counts[int(e[1])] += 1
            boundary_verts = np.nonzero(counts % 2 == 1)[0].astype(np.int32)
            return boundary_verts.reshape(-1, 1)
        
        # General case: top simplices have top_dim+1 vertices, codim-1 faces have top_dim vertices.
        # Count, mod 2, how many top simplices contain each codim-1 face.
        face_count = {}
        for sigma in top_simplices:
            sigma_sorted = sorted(int(v) for v in sigma)
            # Each face is obtained by deleting one vertex.
            for i in range(len(sigma_sorted)):
                face = tuple(sigma_sorted[:i] + sigma_sorted[i+1:])
                face_count[face] = face_count.get(face, 0) + 1
        
        boundary_faces = sorted(face for face, cnt in face_count.items() if cnt % 2 == 1)
        if not boundary_faces:
            return np.empty((0, top_dim), dtype=np.int32)
        return np.array(boundary_faces, dtype=np.int32)
            
''' Trick to not actually store the vertices. 
If asked the 0-simplices, generates an arange on the fly
Otherwise, returns the appropriate _simplices_by_dim
from the SimplicialComplex instance 
'''
class _DimView:
    def __init__(self, Complex):
        self._c = Complex
        
    def __getitem__(self, k):
        if k == 0:
            return np.arange(self._c.n_points, dtype=np.int32)
        return self._c._simplices_by_dim[k] 

        