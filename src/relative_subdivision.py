# Alexander subdivision

import numpy as np
from scipy.spatial import Delaunay
import scipy as sp
import itertools
from itertools import combinations
import copy

def subdivision( M, points ):
    
    ''' Docstring
    '''

    '''
    Assumes Subs has already been defined as a global variable, as in
    Subs = {i: AbstractSubdivision(i) for i in range(2,5)}
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
                    thisIndex = M.index(thisFace)
                    
                    #print("That is index ", thisIndex)
                
                    newSimplex.append(thisIndex + n_points)
                    
                else:
                    thisIndex = thisFace[0]
                    newSimplex.append(thisIndex)
                
            
            #print("So we add simplex ", newSimplex)    
                
            subdivision.append( newSimplex )
            
    
    return subdivision, SubPoints

def RelSub( M, pointsM, L, pointsL ):

    try:
        Subs
    except NameError:
        Subs = {}

        for i in range(2,4+1):
            Subs[i] = AbstractSubdivision(i)

    
    # generate abstract triangle cases
    RelSubs = {}

    for i in range(4):
        RelSubs[(3,i)] = RelAbsSub( 3, i )
    
    
    n_pointsL = pointsL.shape[0]
    
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
    
    # iterate over all simplices in the difference
    for s, sigma in enumerate(MminusL):
        
        l = len(sigma)
        #print('* Simplex to subdivide is ', sigma)
        
        if l == 3: # if triangle
            # find how many and which edges of sigma are in L
            flag1 = ( [sigma[0],sigma[1]] in L)

            flag2 = ( [sigma[1],sigma[2]] in L)
            
            flag3 = ( [sigma[0],sigma[2]] in L)
            
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
                        
                        thisIndex = MminusL.index(thisFace)

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
                        
                        thisIndex = MminusL.index(thisFace)

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