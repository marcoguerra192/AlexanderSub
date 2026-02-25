# Alexander subdivision

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import scipy as sp
import itertools
from itertools import combinations
import copy

from gudhi import AlphaComplex as AC




# def AlphaComplex(points, thresh):
#     ''' Takes a point cloud (Npoints x Ndims) and returns the Alpha complex at threshold thresh '''
    
#     NPoints = points.shape[0]
    
#     #build the circumsphere
#     def circumsphereRad(P):
#         '''
#         Inspired from Yohai Rehani
#         :param P: array Nxd of N points of dimension d
#         :return: center and squared radius of the circum-sphere of P
#         '''
#         p1 = P[0, :]
#         A = P[1:, :] - p1
#         Pnorm = np.sum(np.power(P, 2), 1)
#         b = 0.5*(Pnorm[1:] - Pnorm[0])
#         invA = np.linalg.pinv(A)
#         c0 = invA.dot(b)
#         F = sp.linalg.null_space(A)
#         if F.size != 0:
#             z = np.transpose(F).dot(p1-c0)
#             c = c0 + F.dot(z)
#         else:
#             c = c0
#         R = np.sum(np.power(p1-c, 2))
#         return R

#     Del = Delaunay(points)
#     #triangles = points[Del.simplices]
    
#     TriList = [ t for t in Del.simplices if np.sqrt(circumsphereRad(points[t])) <= thresh ]
#     TriList = [ np.array(sorted(t)) for t in TriList ]
    
#     EdgeList = []
#     CandidateEdges = []
    
#     for t in Del.simplices:
        
        
#         e1 = tuple(sorted((t[0] ,t[1])))
#         e2 = tuple(sorted((t[0] ,t[2])))
#         e3 = tuple(sorted((t[1] ,t[2])))  
        
#         CandidateEdges.extend( [e1,e2,e3] )
        
#     CandidateEdges = list(set(CandidateEdges))
    
#     for e in CandidateEdges:
        
#         if circumsphereRad(points[[e[0],e[1]],:]) <= thresh**2 :
            
#             EdgeList.append(  (e[0],e[1]) )
                  
                
#     EdgeList = [ np.array(e) for e in EdgeList ]
    
#     return TriList, EdgeList








def tightSupp( L , pointsL, M, pointsM ):
    ''' Assuming L < M is a subcomplex of M, compute the "tight" supplement of L in M, 
    that is the set of simplices in (M,L \cup L_out)' that do not have any vertex in L.
    For our current case L_out is empty, so this reduces to \sigma \in (M,L)' such that
    no vertex of \sigma is a simplex of L (eq. has a vertex in L').
    
    '''
    
    
    # compute vertices of L
    L0 = list(set( [ vert for sigma in L for vert in sigma ] ))
    
    # Compute Lout (IT WILL BE EMPTY!)
    Lout = [ simp for simp in M if all( x not in L0 for x in simp)  ]

    
    relsubs, newpoints = RelSub(M, pointsM, L, pointsL)
    
    # Find which vertices in sM are vertices of L
    
    SubComplexIndices = []
    for i in range(newpoints.shape[0]):
        
        if any( all(newpoints[i,:] == pointsL[j,:]) for j in range(pointsL.shape[0]) ):
            
            SubComplexIndices.append(i)
            
    RelSupPoints = newpoints.copy()
    
    # keep simplices if they are not in the smaller complex
    Lbar = [ m for m in relsubs if all([s not in SubComplexIndices for s in m ])  ]
    
    # Add vertices to Lbar!
    verticesLbar = [ [n] for n in range(newpoints.shape[0]) if n not in SubComplexIndices ] 
    
    Lbar = verticesLbar + Lbar
            
    
    return Lbar, SubComplexIndices


