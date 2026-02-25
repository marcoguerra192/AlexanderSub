# Alexander subdivision

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import scipy as sp
import itertools
from itertools import combinations
import copy

from gudhi import AlphaComplex as AC


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

def subdivision( M, points ):
    
    if not Subs:
        Subs = {}

        for i in range(2,4+1):
            Subs[i] = AbstractSubdivision(i)
    
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


def draw_2d_simplicial_complex(simplices, pos=None, return_pos=False, fig = None, markedEdges=None):
    """
    Draw a simplicial complex up to dimension 2 from a list of simplices, as in [1].
        
        Args
        ----
        simplices: list of lists of integerss
            List of simplices to draw. Sub-simplices are not needed (only maximal).
            For example, the 2-simplex [1,2,3] will automatically generate the three
            1-simplices [1,2],[2,3],[1,3] and the three 0-simplices [1],[2],[3].
            When a higher order simplex is entered only its sub-simplices
            up to D=2 will be drawn.
        
        pos: dict (default=None)
            If passed, this dictionary of positions d:(x,y) is used for placing the 0-simplices.
            The standard nx spring layour is used otherwise.
           
        ax: matplotlib.pyplot.axes (default=None)
        
        return_pos: dict (default=False)
            If True returns the dictionary of positions for the 0-simplices.
            
        References
        ----------    
        .. [1] I. Iacopini, G. Petri, A. Barrat & V. Latora (2018)
               "Simplicial Models of Social Contagion".
               arXiv preprint arXiv:1810.07031..
    """

    
    #List of 0-simplices
    nodes =list(set(itertools.chain(*simplices)))
    
    #List of 1-simplices
    edges = list(set(itertools.chain(*[[tuple(sorted((i, j))) for i, j in itertools.combinations(simplex, 2)] for simplex in simplices])))

    #List of 2-simplices
    triangles = list(set(itertools.chain(*[[tuple(sorted((i, j, k))) for i, j, k in itertools.combinations(simplex, 3)] for simplex in simplices])))
    
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_xlim([0, 270])      
    ax.set_ylim([0, 270])
    ax.set_xlim([-1, 10])      
    ax.set_ylim([-1, 10])
    ax.get_xaxis().set_ticks([])  
    ax.get_yaxis().set_ticks([])
    ax.axis('on')
       
    if pos is None:
        # Creating a networkx Graph from the edgelist
        G = nx.Graph()
        G.add_edges_from(edges)
        # Creating a dictionary for the position of the nodes
        pos = nx.spring_layout(G)
        
    # Drawing the edges
    for i, j in edges:
        (x0, y0) = pos[i]
        (x1, y1) = pos[j]
        line = plt.Line2D([ x0, x1 ], [y0, y1 ],color = 'blue', zorder = 1, lw=1.5)
        ax.add_line(line);
    
    # Filling in the triangles
    for i, j, k in triangles:
        (x0, y0) = pos[i]
        (x1, y1) = pos[j]
        (x2, y2) = pos[k]
        tri = plt.Polygon([ [ x0, y0 ], [ x1, y1 ], [ x2, y2 ] ],
                          edgecolor = 'white', facecolor = plt.cm.Blues(0.6),
                          zorder = 2, alpha=0.4, lw=0.5)
        ax.add_patch(tri);
        
    # AGGIUNTA MIA 
    # HIGHLIGHTED EDGES
    if markedEdges is not None:
            for i,j in markedEdges:
                (x0 , y0) = pos[i]
                (x1 , y1) = pos[j]
                line = plt.Line2D([ x0, x1 ], [y0, y1 ],color = u'#ff7f0e', zorder = 3, lw=2)
                ax.add_line(line);

    # Drawing the nodes 
    #for i in nodes:
    for i in range(len(pos)):
        (x, y) = pos[i]
        #  radius was 0.1
        circ = plt.Circle([ x, y ], radius = 0.01, zorder = 4, lw=0.5,
                          edgecolor = 'Black', facecolor = u'#ff7f0e')
        ax.add_patch(circ);
        
#     for i in nodes:
        
#         (x, y) = posLabels[i]
#         string = ' $x_{' + str(i) + '}$'
        
#         ax.text(x,y,string, color='black')
        

    #return fig
def getAlphaComplex(points, thresh):
    TrianglesM, EdgesM = AlphaComplex(points, thresh)
    trianglesM = [ t.tolist() for t in TrianglesM ]
    edgesM = [ e.tolist() for e in EdgesM ]
    M = trianglesM + edgesM
    
    return M

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

def subdivision( M, points ):
    
    try:
        Subs
    except NameError:
        Subs = {}

        for i in range(2,4+1):
            Subs[i] = AbstractSubdivision(i)
    
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

def draw_Sup(simplices, subSimplices, supSimplices, subComplexIndices,  pos, posSub, boundaryComponent=None, excludedVerts=None, return_pos=False, fig = None):
    """
    Draw a simplicial complex up to dimension 2 from a list of simplices, as in [1].
        
        Args
        ----
        simplices: list of lists of integerss
            List of simplices to draw. Sub-simplices are not needed (only maximal).
            For example, the 2-simplex [1,2,3] will automatically generate the three
            1-simplices [1,2],[2,3],[1,3] and the three 0-simplices [1],[2],[3].
            When a higher order simplex is entered only its sub-simplices
            up to D=2 will be drawn.
        
        pos: dict (default=None)
            If passed, this dictionary of positions d:(x,y) is used for placing the 0-simplices.
            The standard nx spring layour is used otherwise.
           
        ax: matplotlib.pyplot.axes (default=None)
        
        return_pos: dict (default=False)
            If True returns the dictionary of positions for the 0-simplices.
            
        References
        ----------    
        .. [1] I. Iacopini, G. Petri, A. Barrat & V. Latora (2018)
               "Simplicial Models of Social Contagion".
               arXiv preprint arXiv:1810.07031..
    """

    
    #List of 0-simplices
    nodes =list(set(itertools.chain(*simplices)))
    
    #List of 1-simplices
    edges = list(set(itertools.chain(*[[tuple(sorted((i, j))) for i, j in itertools.combinations(simplex, 2)] for simplex in simplices])))

    #List of 2-simplices
    triangles = list(set(itertools.chain(*[[tuple(sorted((i, j, k))) for i, j, k in itertools.combinations(simplex, 3)] for simplex in simplices])))
    
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_xlim([0, 270])      
    ax.set_ylim([0, 270])
    ax.set_xlim([-1, 10])      
    ax.set_ylim([-1, 10])
    ax.get_xaxis().set_ticks([])  
    ax.get_yaxis().set_ticks([])
    ax.axis('on')
       
    if pos is None:
        # Creating a networkx Graph from the edgelist
        G = nx.Graph()
        G.add_edges_from(edges)
        # Creating a dictionary for the position of the nodes
        pos = nx.spring_layout(G)
        
    # Drawing the edges
    for i, j in edges:
        (x0, y0) = pos[i]
        (x1, y1) = pos[j]
        line = plt.Line2D([ x0, x1 ], [y0, y1 ],color = 'blue', zorder = 1, lw=1.5)
        ax.add_line(line);
    
    # Filling in the triangles
    for i, j, k in triangles:
        (x0, y0) = pos[i]
        (x1, y1) = pos[j]
        (x2, y2) = pos[k]
        tri = plt.Polygon([ [ x0, y0 ], [ x1, y1 ], [ x2, y2 ] ],
                          edgecolor = 'white', facecolor = plt.cm.Blues(0.6),
                          zorder = 2, alpha=0.4, lw=0.5)
        ax.add_patch(tri);
        
    # Drawing the nodes 
    #for i in nodes:
    for i in range(len(pos)):
        (x, y) = pos[i]
        #  radius was 0.1
        circ = plt.Circle([ x, y ], radius = 0.01, zorder = 4, lw=0.5,
                          edgecolor = 'Black', facecolor = 'Blue')
        ax.add_patch(circ);
     
    # For the subcomplex
        
    #List of 0-simplices
    nodes =list(set(itertools.chain(*subSimplices)))
    
    #List of 1-simplices
    edges = list(set(itertools.chain(*[[tuple(sorted((i, j))) for i, j in itertools.combinations(simplex, 2)] for simplex in subSimplices])))

    #List of 2-simplices
    triangles = list(set(itertools.chain(*[[tuple(sorted((i, j, k))) for i, j, k in itertools.combinations(simplex, 3)] for simplex in subSimplices])))
    
    
    # Drawing the edges
    for i, j in edges:
        (x0, y0) = posSub[i]
        (x1, y1) = posSub[j]
        line = plt.Line2D([ x0, x1 ], [y0, y1 ],color = 'green', zorder = 2, lw=1.5)
        ax.add_line(line);
    
    # Filling in the triangles
    for i, j, k in triangles:
        (x0, y0) = posSub[i]
        (x1, y1) = posSub[j]
        (x2, y2) = posSub[k]
        tri = plt.Polygon([ [ x0, y0 ], [ x1, y1 ], [ x2, y2 ] ],
                          edgecolor = plt.cm.Greens(0.3), facecolor = plt.cm.Greens(0.6),
                          zorder = 2, alpha=0.4, lw=0.5)
        ax.add_patch(tri);
        
    # Drawing the nodes 
    #for i in nodes:
    for i in range(len(posSub)):
        (x, y) = posSub[i]
        #  radius was 0.1
        circ = plt.Circle([ x, y ], radius = 0.01, zorder = 4, lw=0.5,
                          edgecolor = 'Black', facecolor = 'Green')
        ax.add_patch(circ);
        
        
    #List of 0-simplices
    nodes =list(set(itertools.chain(*supSimplices)))
    
    #List of 1-simplices
    edges = list(set(itertools.chain(*[[tuple(sorted((i, j))) for i, j in itertools.combinations(simplex, 2)] for simplex in supSimplices])))

    #List of 2-simplices
    triangles = list(set(itertools.chain(*[[tuple(sorted((i, j, k))) for i, j, k in itertools.combinations(simplex, 3)] for simplex in supSimplices])))
    
    
    # Drawing the edges
    for i, j in edges:
        
        (x0, y0) = pos[i]
        (x1, y1) = pos[j]
        
        if excludedVerts is not None and i in excludedVerts:

            Color = u'#ff7f0e'
        else:
            Color = 'red'

        
        line = plt.Line2D([ x0, x1 ], [y0, y1 ],color = Color, zorder = 2, lw=1.5)
        ax.add_line(line);
    
    # Filling in the triangles
    for i, j, k in triangles:
        
        (x0, y0) = pos[i]
        (x1, y1) = pos[j]
        (x2, y2) = pos[k]
        
        if excludedVerts is not None and i in excludedVerts:

            Color = plt.cm.Oranges(0.6)
        else:
            Color = plt.cm.Reds(0.6)
                
       
        
        tri = plt.Polygon([ [ x0, y0 ], [ x1, y1 ], [ x2, y2 ] ],
                          edgecolor = 'white', facecolor = Color,
                          zorder = 2, alpha=0.4, lw=0.5)
        ax.add_patch(tri);
        
    # Drawing the nodes 
    #for i in nodes:
    for i in range(len(pos)):
         if i not in subComplexIndices:
            (x, y) = pos[i]
            #  radius was 0.1
            
            
            if excludedVerts is not None and i in excludedVerts:

                Color = u'#ff7f0e'
            else:
                Color = 'Red'
            
            circ = plt.Circle([ x, y ], radius = 0.01, zorder = 4, lw=0.5,
                              edgecolor = 'Black', facecolor = Color)
            ax.add_patch(circ);
     
    # MARKED EDGES and Vertices
    
    if boundaryComponent is not None:
            for i,j in boundaryComponent:
                (x0 , y0) = pos[i]
                (x1 , y1) = pos[j]
                line = plt.Line2D([ x0, x1 ], [y0, y1 ],color = u'#ff7f0e', zorder = 2, lw=2)
                ax.add_line(line);
                
                circ = plt.Circle([ x0, y0 ], radius = 0.01, zorder = 4, lw=0.5, edgecolor = 'Black', facecolor = u'#ff7f0e')
                ax.add_patch(circ);
                
                circ = plt.Circle([ x1, y1 ], radius = 0.01, zorder = 4, lw=0.5, edgecolor = 'Black', facecolor = u'#ff7f0e')
                ax.add_patch(circ);
                
            
    
        
#     for i in nodes:
        
#         (x, y) = posLabels[i]
#         string = ' $x_{' + str(i) + '}$'
        
#         ax.text(x,y,string, color='black')
        

    #return fig
