import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import scipy as sp
import itertools
from itertools import combinations
import copy


def draw_2d_simplicial_complex(simplices, pos=None, return_pos=False, markedEdges=None):
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

def draw_Sup(simplices, subSimplices, supSimplices, subComplexIndices,  pos, posSub, boundaryComponent=None, excludedVerts=None, return_pos=False):
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