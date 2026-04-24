# Alexander subdivision

import numpy as np
from scipy.spatial import Delaunay
import scipy as sp
import itertools
from itertools import combinations
import copy

from .abstract_subdivision import AbstractSubdivision, RelAbsSub


def subdivision( M, points, Subs):
    
    ''' Documentation
    '''

    
    n_points = points.shape[0]
    
    # vertices
    
    N_Verts = n_points + len(M)
    SubPoints = np.zeros((N_Verts, 2))
    
    SubPoints[ 0 : n_points , : ] = points
    
    
    for i in range(len(M)):
        
        bary = np.mean(points[M[i],:], axis=0)
        
        SubPoints[i+n_points,:] = bary
        
    subdivision = []

    # compute map for indices once
    M_simplex_index = { tuple(sorted(simplex)): idx for idx, simplex in enumerate(M) }
        
    # iterate over all simplices
    for s, sigma in enumerate(M):
        
        #print("We consider simplex ", sigma)
        l = len(sigma)
        #print("Simplex on ",l," points")
        
        for sub in Subs[l]:
            
            #print("This sub is ",sub)
            newSimplex = []
            for face in sub:
                
                thisFace = [ sigma[h] for h in face ]
                #print("This face is ",thisFace)
                
                if len(thisFace)>1:
                    #thisIndex = M.index(thisFace)
                    thisIndex = M_simplex_index[tuple(sorted(thisFace))]
                    
                    #print("That is index ", thisIndex)
                
                    newSimplex.append(thisIndex + n_points)
                    
                else:
                    thisIndex = thisFace[0]
                    newSimplex.append(thisIndex)
                
            
            #print("So we add simplex ", newSimplex)    
                
            subdivision.append( sorted(newSimplex) )
            
    
    return subdivision, SubPoints

# 1. Relative subdivision when L_out is empty, so the vertices are in common

def RelSub( M, pointsM, L, pointsL, Subs ):
    
    ''' Documentation
    '''

    
    # generate abstract triangle cases
    RelSubs = {}

    for i in range(4):
        RelSubs[(3,i)] = RelAbsSub( 3, i, Subs )
    
    
    n_pointsL = pointsL.shape[0]

    L_faces = { tuple(sorted(s)) for s in L }
    
    MminusL = [ x for x in M if x not in L ] # M minus L
    NVerts = pointsL.shape[0] + len(MminusL) # How many vertices in the relative sub
    
    SubPoints = np.zeros((NVerts, 2))
    
    # the first points are just the points of L
    SubPoints[ 0 : n_pointsL , : ] = pointsL.copy()
    
    for i in range(len(MminusL)): # for every simplex in the subset
        
        bary = np.mean(pointsM[MminusL[i],:], axis=0) # compute barycenter
        SubPoints[i+n_pointsL,:] = bary # add it in position
        
    subdivision = []
    
    subdivision.extend(L) # first, add all L

    MminusL_simplex_index = { tuple(sorted(simplex)): idx for idx, simplex in enumerate(MminusL) }

    L_edges = { tuple(sorted(s)) for s in L if len(s)==2 }
    
    # iterate over all simplices in the difference
    for s, sigma in enumerate(MminusL):

        ## added
        key = tuple(sorted(sigma))
        if key in L_faces:
            continue
        
        l = len(sigma)
        #print('* Simplex to subdivide is ', sigma)
        
        if l == 3: # if triangle
            # find how many and which edges of sigma are in L
            flag1 = tuple(sorted([sigma[0],sigma[1]])) in L_edges
            
            flag2 = tuple(sorted([sigma[1],sigma[2]])) in L_edges
            
            flag3 = tuple(sorted([sigma[0],sigma[2]])) in L_edges
            
            nLocked = sum( [ 1 if x else 0 for x in [flag1, flag2, flag3] ] )

            # map the vertices accordingly
            
            if nLocked == 0: # if None are in L
                
                Map = [0,1,2] # the basic ordering is fine
                
            elif nLocked == 3: # if all are in L 
            
                Map = [0,1,2] # the basic ordering is fine as well
            
            elif nLocked == 1:
                
                if flag1:
                    Map = [0,1,2] # the basic ordering is fine
                elif flag2:
                    Map = [1,2,0] # first permutation
                elif flag3:
                    Map = [2,0,1] # second permutation
            elif nLocked == 2:
                
                if not flag3: # if it's NOT flag 3, so 1 and 2
                    Map = [0,1,2] # the basic ordering is fine
                elif not flag1: # if it's NOT flag 1, so 2 and 3
                    Map = [1,2,0] # first permutation
                elif not flag2:
                    Map = [2,0,1] # second permutation
                
            #print('nLocked is: ', nLocked)
            #print('The map is ', Map)
            # Then use the appropriate abstract subdivision of a triangle
            for sub in RelSubs[(3, nLocked)]:
                
                #print(' Abstract sub is: ',sub)
                
                newSimplex = []
                
                for face in sub:
                    
                    #print('  Abstract vertex: ', face)
                    
                    # HERE WE MUST USE THE MAP to take into account that the Relative Abstract Subdivision is on the standard triangle
                    thisFace = [ sigma[Map[h]] for h in face ]
                    
                    # AND THEN REORDER
                    thisFace = list(sorted(thisFace))
                     
                    #print('  This vertex: ', thisFace)
                    
                    if len(thisFace)>1: # if it's not a vertex
                        
                        #print('    Barycenter')

                        key = tuple(sorted(thisFace))
                        if key not in MminusL_simplex_index:
                            print("Face not in MminusL:", key)

                        if key in L_faces:        # face in L
                            newSimplex.extend(key)

                        else:                     # face in M\L
                            newSimplex.append(
                                MminusL_simplex_index[key] + n_pointsL
                            )
                        
                        #thisIndex = MminusL.index(thisFace)
                        #thisIndex = MminusL_simplex_index[tuple(sorted(thisFace))]

                        #print("    Goes to index ", thisIndex)

                        #newSimplex.append(thisIndex + n_pointsL)
                        
                        #print("    Translating to ", thisIndex+ n_pointsL)

                    else:
                        #print('    Original Vertex')
                        thisIndex = thisFace[0]
                        newSimplex.append(thisIndex)
                        #print('    Goes to index ', thisIndex)
                        
                newSimplex = list(sorted(newSimplex))
        
                subdivision.append( newSimplex )
                #print('Adding ',newSimplex)
                
        
        else: # all else must be subdivided normally
            for sub in Subs[l]:
                
                #print(' Abstract sub is: ', sub)
            
                newSimplex = []
                for face in sub:
                    
                    #print('  Abstract vertex: ', face)

                    thisFace = [ sigma[h] for h in face ]
                     
                    #print('  This vertex: ', thisFace)

                    if len(thisFace)>1: # if it's not a vertex
                        
                        #print('    Barycenter')
                        
                        #thisIndex = MminusL.index(thisFace)
                        thisIndex = MminusL_simplex_index[tuple(sorted(thisFace))]

                        #print("    Goes to index ", thisIndex)

                        newSimplex.append(thisIndex + n_pointsL)
                        
                        #print("    Translating to ", thisIndex+ n_pointsL)

                    else:
                        #print('    Original Vertex')
                        thisIndex = thisFace[0]
                        newSimplex.append(thisIndex)
                        #print('    Goes to index ', thisIndex)
                        
                newSimplex = list(sorted(newSimplex))
        
                subdivision.append( newSimplex )
                #print('Adding ',newSimplex)
        
    return subdivision, SubPoints

# 2. Generalized relative subdivision - does not assume the same set of vertices

def RelSubGeneral(M, points_M, K, K_vertex_indices, Subs):
    """
    Relative derived complex (M, K)' where K is an arbitrary subcomplex of M.

    Unlike the original `RelSub`, this version does not assume K's vertices are
    the prefix of points_M. Instead, the caller supplies K_vertex_indices, the
    explicit list of indices into points_M that belong to K.

    Output convention
    -----------------
    The output's vertex array is points_M (full, in original order) followed by
    barycenters of simplices in M setminus K. So:

        sub_points[i] = points_M[i]              for i in 0..n_M-1
        sub_points[n_M + j] = barycenter(M setminus K)_j   for j in 0..|M setminus K|-1

    So K's vertex indices in the output are exactly K_vertex_indices,
    unchanged. Barycenter indices live in [n_M, n_M + |M setminus K|).

    Parameters
    ----------
    M : list of sorted lists of int
        Ambient simplicial complex in mixed-dimension format, i.e. the disk
    points_M : (n_M, d) ndarray
        Vertex coordinates for M.
    K : list of sorted lists of int
        Subcomplex of M.
    K_vertex_indices : iterable of int
        Indices into points_M of the vertices of K. Must equal len(K).
    Subs : dict[int, list]
        Abstract subdivision patterns, as built by AbstractSubdivision(n).

    Returns
    -------
    subdivision : list of sorted lists of int
        Simplices of (M, K)'.
    sub_points : (n_M + |M\\K with len>=2|, d) ndarray
        Vertex coordinates for the subdivision (only positive-dim simplices
        of M\\K contribute barycenters).
    """
    points_M = np.asarray(points_M, dtype=float)
    n_M = points_M.shape[0]

    # Generate per-triangle relative subdivision patterns once.
    RelSubs = {}
    for i in range(4):
        RelSubs[(3, i)] = RelAbsSub(3, i, Subs)

    K_vertex_set = set(K_vertex_indices)
    K_faces = {tuple(sorted(s)) for s in K}
    K_edges = {t for t in K_faces if len(t) == 2}

    # M \ K: simplices of M that do not belong to K. Only simplices of dim >= 1
    # contribute barycenters (a 0-simplex has no barycenter distinct from itself
    # and 0-simplices of M that aren't in K become 0-simplices of L_out, but
    # they're already vertices of M).
    MminusK = [s for s in M if tuple(sorted(s)) not in K_faces and len(s) >= 2]

    # Index map from a M\K simplex (as sorted tuple) to its barycenter index in
    # the output vertex array.
    MminusK_index = {tuple(sorted(s)): n_M + j for j, s in enumerate(MminusK)}

    # Compute and stack barycenters.
    n_bary = len(MminusK)
    sub_points = np.zeros((n_M + n_bary, points_M.shape[1]), dtype=float)
    sub_points[:n_M] = points_M
    for j, s in enumerate(MminusK):
        sub_points[n_M + j] = points_M[s].mean(axis=0)

    subdivision = []

    # Add the simplices of K verbatim (case (ii) of the relative-derived def).
    subdivision.extend([list(s) for s in K])

    # For each simplex of M \ K, emit its abstract subdivision, with the
    # locked-face logic applied for triangles whose edges belong to K.
    for sigma in M:
        key = tuple(sorted(sigma))
        if key in K_faces:
            continue
        l = len(sigma)
        if l < 2:
            # 0-simplex of M \ K; this would be a vertex of L_out only, but
            # by construction L_out ⊆ K so this branch is dead. We handle it
            # defensively: emit it as a singleton.
            subdivision.append([sigma[0]])
            continue

        if l == 3:
            # Triangle case: dispatch by which of its three edges lie in K.
            sigma_sorted = sorted(sigma)
            e01 = (sigma_sorted[0], sigma_sorted[1])
            e12 = (sigma_sorted[1], sigma_sorted[2])
            e02 = (sigma_sorted[0], sigma_sorted[2])
            flag1 = e01 in K_edges
            flag2 = e12 in K_edges
            flag3 = e02 in K_edges
            nLocked = int(flag1) + int(flag2) + int(flag3)

            # The same Map logic as in the original RelSub: rotate so the
            # locked edges occupy the configuration RelAbsSub expects.
            if nLocked == 0 or nLocked == 3:
                Map = [0, 1, 2]
            elif nLocked == 1:
                if flag1:
                    Map = [0, 1, 2]
                elif flag2:
                    Map = [1, 2, 0]
                else:  # flag3
                    Map = [2, 0, 1]
            else:  # nLocked == 2
                if not flag3:
                    Map = [0, 1, 2]
                elif not flag1:
                    Map = [1, 2, 0]
                else:  # not flag2
                    Map = [2, 0, 1]

            for sub in RelSubs[(3, nLocked)]:
                newSimplex = []
                for face in sub:
                    thisFace = sorted(sigma_sorted[Map[h]] for h in face)
                    if len(thisFace) > 1:
                        face_key = tuple(thisFace)
                        if face_key in K_faces:
                            # Cone vertex (case (iii) of the relative def):
                            # contributes the K-face's vertices directly.
                            newSimplex.extend(face_key)
                        else:
                            # Barycenter of an M\K face.
                            if face_key not in MminusK_index:
                                raise RuntimeError(
                                    f"Triangle face {face_key} of sigma={sigma_sorted} "
                                    f"is neither in K nor in M\\K. M may be missing "
                                    f"this face as an explicit simplex."
                                )
                            newSimplex.append(MminusK_index[face_key])
                    else:
                        newSimplex.append(thisFace[0])
                subdivision.append(sorted(newSimplex))
        else:
            # Generic (non-triangle) case: standard barycentric pattern with
            # no locked-face dispatch. This branch handles edges of M \ K
            # (which become "edge from vertex to its barycenter" pairs) and,
            # in higher dimensions, simplices of dim != 2.
            for sub in Subs[l]:
                newSimplex = []
                for face in sub:
                    thisFace = sorted(sigma[h] for h in face)
                    if len(thisFace) > 1:
                        face_key = tuple(thisFace)
                        if face_key in K_faces:
                            newSimplex.extend(face_key)
                        else:
                            if face_key not in MminusK_index:
                                raise RuntimeError(
                                    f"Face {face_key} of sigma={sigma} "
                                    f"is neither in K nor in M\\K."
                                )
                            newSimplex.append(MminusK_index[face_key])
                    else:
                        newSimplex.append(thisFace[0])
                subdivision.append(sorted(newSimplex))

    return subdivision, sub_points

def tightSupp_NoLOut( L , pointsL, M, pointsM, Subs ):
    ''' Assuming L < M is a subcomplex of M, compute the "tight" supplement of L in M, 
    that is the set of simplices in (M,L cup L_out)' that do not have any vertex in L.
    For our current case L_out is empty, so this reduces to sigma in (M,L)' such that
    no vertex of sigma is a simplex of L (eq. has a vertex in L').
    
    '''
    
    # compute vertices of L
    L0 = list(set( [ vert for sigma in L for vert in sigma ] ))
    
    # Compute Lout (IT WILL BE EMPTY!)
    Lout = [ simp for simp in M if all( x not in L0 for x in simp)  ]

    
    relsubs, newpoints = RelSub(M, pointsM, L, pointsL, Subs)
    
    # Find which vertices in sM are vertices of L
    
    SubComplexIndices = []
    for i in range(newpoints.shape[0]):
        for j in range(pointsL.shape[0]):
            if np.allclose(newpoints[i, :], pointsL[j, :], atol=1e-12):
            
                SubComplexIndices.append(i)
            
    RelSupPoints = newpoints.copy()
    
    # keep simplices if they are not in the smaller complex
    Lbar = [ m for m in relsubs if all([s not in SubComplexIndices for s in m ])  ]
    
    # Add vertices to Lbar!
    verticesLbar = [ [n] for n in range(newpoints.shape[0]) if n not in SubComplexIndices ] 
    
    Lbar = verticesLbar + Lbar
            
    
    return Lbar, SubComplexIndices

# 1. L_out

def compute_Lout(L, M):
    """
    Return L_out = { σ ∈ M : σ has no vertex in V(L) }.

    L_out is automatically closed under taking faces (the property "no vertex
    in V(L)" is hereditary), so the returned list is a bona fide subcomplex.

    Parameters
    ----------
    L, M : list of sorted lists of int
        Simplicial complexes in mixed-dimension format.

    Returns
    -------
    Lout : list of sorted lists of int
        Subcomplex of M with no vertex in V(L).
    """
    L_vertices = {v for s in L for v in s}
    Lout = [s for s in M if all(v not in L_vertices for v in s)]
    return Lout

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
    isolated = [[v] for v in sorted(used_verts) if v not in existing_verts]
    Ltilde = isolated + [s for s in Ltilde if len(s) > 1]

    return Ltilde, sub_points, L_vertex_indices, Lout_vertex_indices
