""" Visualization for the class-based Alexander duality pipeline.

A single function `draw_pipeline` accepts whatever subset of the pipeline's
outputs you have computed and plots them in layered fashion. Inputs can range
from "just the subdivision" all the way to "the full pipeline with cycles".

Color conventions:
  - Parent M (background): light blue.
  - L (subcomplex of M, via base_inclusion): green.
  - Supplement (in the materialized subdivision): red.
  - Excluded supplement components (boundary-touching): orange.
  - Representative cycles (edges in L): bold orange.

Anything that isn't passed in is simply not drawn.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


# Style defaults
_NODE_RADIUS_FRAC = 0.003   # circle radius as fraction of plot extent
_PAD_FRAC = 0.05


def _collect_xy(*point_arrays):
    """ Concatenate x and y coordinates from a sequence of (n, 2) arrays. """
    xs, ys = [], []
    for arr in point_arrays:
        if arr is None or len(arr) == 0:
            continue
        xs.extend(arr[:, 0].tolist())
        ys.extend(arr[:, 1].tolist())
    return xs, ys


def _set_axes(ax, all_x, all_y):
    """ Auto-fit axes to content with padding, equal aspect, no ticks. """
    if not all_x:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    else:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        extent = max(x_max - x_min, y_max - y_min, 1e-9)
        pad = _PAD_FRAC * extent
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect('equal')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


def _node_radius(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    extent = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    return _NODE_RADIUS_FRAC * extent


def _draw_complex(ax, complex, edge_color, face_color, vertex_color,
                  edge_lw=1.5, alpha=0.4, zorder_edge=1, zorder_face=2,
                  zorder_vertex=4, vertex_filter=None):
    """ Draw a SimplicialComplex onto ax with the given colors.
    
    `vertex_filter`: if provided, an iterable of vertex indices to draw
    (others are skipped). If None, draw all vertices.
    """
    pos = complex.points
    radius = _node_radius(ax)

    # Edges
    if complex.dim >= 1:
        for e in complex.simplices_in_dim[1]:
            (x0, y0) = pos[int(e[0])]
            (x1, y1) = pos[int(e[1])]
            line = plt.Line2D([x0, x1], [y0, y1],
                              color=edge_color, zorder=zorder_edge, lw=edge_lw)
            ax.add_line(line)

    # Triangles
    if complex.dim >= 2:
        for t in complex.simplices_in_dim[2]:
            (x0, y0) = pos[int(t[0])]
            (x1, y1) = pos[int(t[1])]
            (x2, y2) = pos[int(t[2])]
            tri = plt.Polygon([[x0, y0], [x1, y1], [x2, y2]],
                              edgecolor='white', facecolor=face_color,
                              zorder=zorder_face, alpha=alpha, lw=0.5)
            ax.add_patch(tri)

    # Vertices
    vertex_iter = (range(complex.n_points) if vertex_filter is None
                   else vertex_filter)
    for i in vertex_iter:
        (x, y) = pos[int(i)]
        circ = plt.Circle([x, y], radius=radius, zorder=zorder_vertex,
                          lw=0.5, edgecolor='Black', facecolor=vertex_color)
        ax.add_patch(circ)


def _draw_marked_edges(ax, complex, edges_in_complex_indexing, color,
                       lw=2.0, zorder=5):
    """ Draw a list of edges (each a sorted tuple of vertex indices into
    `complex`) on top of `ax`. """
    if not edges_in_complex_indexing:
        return
    pos = complex.points
    for e in edges_in_complex_indexing:
        (x0, y0) = pos[int(e[0])]
        (x1, y1) = pos[int(e[1])]
        line = plt.Line2D([x0, x1], [y0, y1], color=color, zorder=zorder, lw=lw)
        ax.add_line(line)


def draw_pipeline(subdivision=None,
                  supplement=None,
                  valid_components=None,
                  excluded_components=None,
                  representative_cycles=None,
                  ax=None,
                  figsize=(7, 7)):
    """ Draw any subset of the pipeline's state.
    
    Parameters
    ----------
    subdivision : Subdivision, optional
        If given, plot the parent M (background) and, if non-trivial, the
        subcomplex L (highlighted).
    supplement : Inclusion, optional
        Supplement of L in the materialized subdivision (as returned by
        Subdivision.supplement()). If given, draw it on top of M.
    valid_components : list of np.ndarray, optional
        Connected components of the supplement that survived boundary discard.
        Vertices belonging to these components are drawn in red. Requires
        `supplement` to be given as well.
    excluded_components : list of np.ndarray, optional
        Components that were discarded (touch boundary). Drawn in orange.
        Requires `supplement` as well.
    representative_cycles : list of list of tuple, optional
        Per-component lists of L-edges (in L's vertex indexing). Drawn as
        bold orange edges on top of L. Requires `subdivision` (with non-trivial
        base_inclusion).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure is created.
    figsize : tuple
        Used only when ax is None.
    
    Returns
    -------
    ax : the axes used.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Pre-pass to gather all coordinates for auto-axes.
    all_x, all_y = [], []
    if subdivision is not None:
        x, y = _collect_xy(subdivision.parent.points)
        all_x.extend(x); all_y.extend(y)
    if supplement is not None:
        x, y = _collect_xy(supplement.large.points)
        all_x.extend(x); all_y.extend(y)
    _set_axes(ax, all_x, all_y)

    # Layer 1: parent complex M (background).
    if subdivision is not None:
        M = subdivision.parent
        _draw_complex(ax, M,
                      edge_color='steelblue',
                      face_color=plt.cm.Blues(0.4),
                      vertex_color='steelblue',
                      edge_lw=1.2, alpha=0.25,
                      zorder_edge=1, zorder_face=1, zorder_vertex=2)

    # Layer 2: L (subcomplex of M from base_inclusion), if non-trivial.
    if subdivision is not None and not subdivision.base_inclusion.is_trivial:
        base = subdivision.base_inclusion
        L = base.small
        _draw_complex(ax, L,
                      edge_color='green',
                      face_color=plt.cm.Greens(0.6),
                      vertex_color='green',
                      edge_lw=1.5, alpha=0.5,
                      zorder_edge=3, zorder_face=3, zorder_vertex=4)

    # Layer 3: supplement (in the materialized subdivision).
    if supplement is not None:
        supp_complex = supplement.small
        # Determine per-vertex color via component membership.
        valid_set = set()
        excluded_set = set()
        if valid_components is not None:
            for c in valid_components:
                valid_set.update(int(v) for v in c)
        if excluded_components is not None:
            for c in excluded_components:
                excluded_set.update(int(v) for v in c)

        # Draw all supplement edges and triangles in red (default), and
        # vertices colored by membership.
        _draw_complex(ax, supp_complex,
                      edge_color='red',
                      face_color=plt.cm.Reds(0.5),
                      vertex_color='red',
                      edge_lw=1.5, alpha=0.45,
                      zorder_edge=5, zorder_face=4, zorder_vertex=6,
                      vertex_filter=[])  # we'll draw vertices manually
        # Manual vertex pass with per-vertex color.
        radius = _node_radius(ax)
        pos = supp_complex.points
        for v in range(supp_complex.n_points):
            if v in excluded_set:
                color = u'#ff7f0e'   # orange
            elif v in valid_set:
                color = 'red'
            else:
                color = 'darkred'
            (x, y) = pos[int(v)]
            circ = plt.Circle([x, y], radius=radius * 1.3, zorder=7,
                              lw=0.5, edgecolor='Black', facecolor=color)
            ax.add_patch(circ)

    # Layer 4: representative cycles (drawn in L's vertex indexing).
    if representative_cycles is not None and subdivision is not None:
        if subdivision.base_inclusion.is_trivial:
            raise ValueError("Cannot draw representative_cycles without an L "
                             "(base_inclusion is trivial).")
        L = subdivision.base_inclusion.small
        # Flatten all cycles into one list of edges, keeping color uniform.
        all_cycle_edges = []
        for cyc in representative_cycles:
            all_cycle_edges.extend(cyc)
        _draw_marked_edges(ax, L, all_cycle_edges,
                           color=u'#ff7f0e', lw=3.0, zorder=8)

    return ax

def draw_setdiff_pipeline(inclusion=None,
                          diff_inclusion=None,
                          valid_components=None,
                          excluded_components=None,
                          representative_cycles=None,
                          ax=None,
                          figsize=(7, 7)):
    """ Draw the state of the set-difference (manifold-only) pipeline.

    Layered like `draw_pipeline`: any subset of inputs is allowed; missing
    layers are simply not drawn.

    Color conventions:
      - M (background): light blue.
      - L (subcomplex of M): green.
      - Top-dim simplices of the complement complex M\\L:
            - in a valid component: red
            - in an excluded (boundary-touching) component: orange
            - not classified yet (no components passed): dark red
      - Representative cycles (edges in L): bold orange.

    Parameters
    ----------
    inclusion : Inclusion, optional
        The original L ⊆ M. If given, M and L are drawn.
    diff_inclusion : Inclusion, optional
        The complement complex ⊆ M, as returned by inclusion.complement_complex().
        If given, the top-dim simplices of the complement are highlighted.
    valid_components : list of np.ndarray, optional
        Components of the complement, as flat indices into M. Vertices/triangles
        belonging to these are drawn in red.
    excluded_components : list of np.ndarray, optional
        Components touching ∂|M|, drawn in orange.
    representative_cycles : list of list of tuple, optional
        Per-component cycles in L's vertex indexing (sorted tuples of
        L-vertex indices). Drawn as bold orange edges.
    ax : matplotlib.axes.Axes, optional
        Drawing target. If None, a new figure is created.
    figsize : tuple
        Used only if ax is None.

    Returns
    -------
    ax : the axes used.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Pre-pass: collect all coordinates for auto-axes.
    all_x, all_y = [], []
    if inclusion is not None:
        x, y = _collect_xy(inclusion.large.points)
        all_x.extend(x); all_y.extend(y)
    if diff_inclusion is not None:
        x, y = _collect_xy(diff_inclusion.large.points)
        all_x.extend(x); all_y.extend(y)
    _set_axes(ax, all_x, all_y)

    # Layer 1: M (background).
    if inclusion is not None:
        M = inclusion.large
        _draw_complex(ax, M,
                      edge_color='steelblue',
                      face_color=plt.cm.Blues(0.4),
                      vertex_color='steelblue',
                      edge_lw=1.2, alpha=0.25,
                      zorder_edge=1, zorder_face=1, zorder_vertex=2)

    # Layer 2: L.
    if inclusion is not None and not inclusion.is_trivial:
        L = inclusion.small
        _draw_complex(ax, L,
                      edge_color='green',
                      face_color=plt.cm.Greens(0.6),
                      vertex_color='green',
                      edge_lw=1.5, alpha=0.5,
                      zorder_edge=3, zorder_face=3, zorder_vertex=4)

    # Layer 3: complement-complex top-dim simplices, colored by component.
    if diff_inclusion is not None:
        M = diff_inclusion.large

        # Build a per-top-dim-simplex color map.
        # M flat indices of top-dim simplices participating in valid/excluded
        # components.
        valid_top = set()
        excluded_top = set()
        if valid_components is not None:
            for c in valid_components:
                valid_top.update(int(f) for f in c)
        if excluded_components is not None:
            for c in excluded_components:
                excluded_top.update(int(f) for f in c)

        # Iterate over the complement-complex's top-dim simplices, expressed
        # in M's vertex/flat indexing for direct comparison with the
        # component sets.
        small = diff_inclusion.small
        vmap = diff_inclusion.vertex_map
        top_dim = small.dim
        if top_dim >= 2:
            for row in small.simplices_in_dim[top_dim]:
                # Translate to M's vertices, then to M's flat index.
                m_sigma = tuple(sorted(int(vmap[int(v)]) for v in row))
                m_flat = M._simplex_to_flat[m_sigma]

                if m_flat in excluded_top:
                    color = u'#ff7f0e'   # orange
                elif m_flat in valid_top:
                    color = 'red'
                else:
                    color = plt.cm.Reds(0.6)  # default: dark red

                # Draw the triangle on M's coordinates.
                pts = M.points
                (x0, y0) = pts[m_sigma[0]]
                (x1, y1) = pts[m_sigma[1]]
                (x2, y2) = pts[m_sigma[2]]
                tri = plt.Polygon([[x0, y0], [x1, y1], [x2, y2]],
                                  edgecolor='white', facecolor=color,
                                  zorder=5, alpha=0.6, lw=0.5)
                ax.add_patch(tri)

    # Layer 4: representative cycles (edges in L).
    if representative_cycles is not None and inclusion is not None:
        if inclusion.is_trivial:
            raise ValueError("Cannot draw representative_cycles without an L "
                             "(inclusion is trivial).")
        L = inclusion.small
        all_cycle_edges = []
        for cyc in representative_cycles:
            all_cycle_edges.extend(cyc)
        _draw_marked_edges(ax, L, all_cycle_edges,
                           color=u'#ff7f0e', lw=3.0, zorder=8)

    return ax