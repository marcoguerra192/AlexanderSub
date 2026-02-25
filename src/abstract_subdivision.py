import numpy as np
from scipy.spatial import Delaunay
import scipy as sp
import itertools
from itertools import combinations
import copy


def AbstractSubdivision( n ):
    ''' Generate the arrows from the element [n] in Delta
    
    '''
    
    simplex = list(range(n))
    
    faces = []
    faces.append(simplex)
    
    for i in range(1,n):
        
        for x in combinations(range(n), i):
            
            face = [s for s in simplex if s not in x]
            faces.append(face)
    
    n_faces = len(faces)
    graph = np.zeros((n_faces, n_faces), dtype=int)

    # generate a graph of face relations (it's just the edges)
    for i in range(n_faces):
        
        for j in range(n_faces):
            
            if all(x in faces[i] for x in faces[j]):
                
                graph[i,j] = 1
                
    # remove the diagonal
    np.fill_diagonal(graph, 0)
    
    # now follow all possible paths!
    subdivision = []
    
    def recTraverse( neighbours , tally ):
        
        #print("New function call")
        
        nextSteps = np.nonzero(neighbours)[0].tolist()
        
        #print("The neighbours are ", nextSteps)
        
        for i in nextSteps:
            
            #print("Considering index ",i, " that is face ", faces[i])
            
            newTally = tally.copy()
            newTally.append(i)
            
            #print("Current tally is ", newTally, "Which corresponds to ", [faces[h] for h in newTally ] )
            
            subdivision.append(newTally)
            
            if len(faces[i]) == 1:
                
                #print('Leaf node. Tally is ', [faces[h] for h in newTally ])
                pass
                
            #print("Calling the next branch with i = ", i)
            recTraverse(graph[i,:] , newTally.copy())
            
    
    for i in range(n_faces): # for each row of the matrix
        
        #print('Working on the ',i, " th row")
        
        tally = [i]
        
        neighbours = graph[i,:]
        
        recTraverse(neighbours, tally)
        
    subdivision = [ [faces[i] for i in s] for s in subdivision ]   
        
    return subdivision

def RelAbsSub(n , lockedFaces):
    '''Function for the relative abstract subdivision of a triangle
    
    Starts from the regular abstract subdivision, and removes the simplices that are not 
    there in the relative case. Currently only works for triangles, but the concept 
    should be general.
    Params:
    n (int), either 2 or 2: the dimension of the simplex to subdivide
    lockedFaces (int): 1,2 or 3. Assumes the ordering [0,1] , [0,1] [1,2], all edges
    '''
    
    if n <= 1 or n > 3:
        raise ValueError("Dimension ", n, " is not supported") 

    try:
        Subs
    except NameError:
        Subs = {}

        for i in range(2,4+1):
            Subs[i] = AbstractSubdivision(i)
        
            
    def lockFace(subsimps, face):
        '''Change the list Sub[n] by locking one specific face'''
        
        # For n = 3, the face can be [0,1], [0,2] or [1,2]
        
        
        subsimps = [ s for s in subsimps if s[0] != face ] # remove the two edges that start in face
        subsimps = [ s for s in subsimps if s[1] != face ] # remove the two triangles through face and the edge that ends in face
        subsimps.append( [ [face[0]] , [face[1]] ] )
        subsimps.append( [ [0,1,2] , [face[0]] , [face[1]] ] )
        
        return subsimps
        
    subsimps = Subs[n].copy()
    
    if lockedFaces == 0:
        pass
    
    if lockedFaces == 1: # assumes it's [0,1]
        subsimps = lockFace(subsimps, [0,1])
        
    if lockedFaces==2: # assumes it's [0,1] and [1,2]
        subsimps = lockFace(subsimps, [0,1])
        subsimps = lockFace(subsimps, [1,2])
        
    if lockedFaces==3: # it's all
        subsimps = lockFace(subsimps, [0,1])
        subsimps = lockFace(subsimps, [1,2])
        subsimps = lockFace(subsimps, [0,2])
        
    return subsimps