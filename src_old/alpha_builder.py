# Alpha builder

import numpy as np
from scipy.spatial import Delaunay
import scipy as sp
import itertools
from itertools import combinations
import copy

from gudhi import AlphaComplex as AC

def AlphaComplex(points, thresh):
    """
    Takes a point cloud (Npoints x Ndims) and returns:
      - TriList: list of np.array([i,j,k]) triangles
      - EdgeList: list of np.array([i,j]) edges

    using Gudhi's AlphaComplex backend.

    thresh is the radius threshold (same meaning as in your original code).
    """
    # Gudhi uses squared radius as filtration value
    thresh_sq = thresh ** 2

    # Build Gudhi alpha complex
    ac = AC(points=points)
    st = ac.create_simplex_tree()

    TriList = []
    EdgeList = []

    # Iterate over simplices and filtration values
    for simplex, filt in st.get_filtration():
        dim = len(simplex) - 1

        # Only keep simplices with filtration <= threshold^2
        if filt <= thresh_sq:
            if dim == 1:
                # edge
                e = np.array(sorted(simplex), dtype=int)
                EdgeList.append(e)
            elif dim == 2:
                # triangle
                t = np.array(sorted(simplex), dtype=int)
                TriList.append(t)

    return TriList, EdgeList



def getAlphaComplex(points, thresh):
    TrianglesM, EdgesM = AlphaComplex(points, thresh)
    trianglesM = [ sorted(t.tolist()) for t in TrianglesM ]
    edgesM = [ sorted(e.tolist()) for e in EdgesM ]
    M = trianglesM + edgesM
    
    return M
