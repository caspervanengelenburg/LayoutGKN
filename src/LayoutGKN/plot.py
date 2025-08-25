import networkx as nx
from shapely.geometry import Polygon
import numpy as np

from LayoutGKN.constants import ROOM_COLORS


def draw_polygon(ax, poly, label=None, **kwargs):
    """Plots a polygon by filling it up. Edges of shapes are avoided to show exactly the area that
    the elements occupy."""
    x, y = poly.exterior.xy
    ax.fill(x, y, label=label, **kwargs)
    return


def draw_rooms(ax, polygons, colors, lw=None, ec="black"):
    """Draws the rooms of the floor plan layout."""

    for poly, color in zip(polygons, colors):
        draw_polygon(ax, poly, facecolor=color, edgecolor=ec, linewidth=lw)


def draw_graph(ax, G, fs, lw=0, s=20, w=2, ec="black", node_color='black', edge_colors=['black', 'white'], viz_rooms=True):

    # Extract information
    polygons = [Polygon(d) for _, d in G.nodes('polygon')]
    colors = [ROOM_COLORS[d] for _, d in G.nodes('category')]
    pos = {n: np.array(
        [Polygon(d).representative_point().x,
         Polygon(d).representative_point().y])
           for n, d in G.nodes('polygon')}

    # Draw room shapes
    if viz_rooms:
        draw_rooms(ax, polygons, colors, lw=lw, ec=ec)

    # Draw nodes
    if isinstance(s, list):
        nx.draw_networkx_nodes(G, pos, node_size=s, node_color=node_color, ax=ax)
    else:
        nx.draw_networkx_nodes(G, pos, node_size=fs*s, node_color=node_color, ax=ax)

    # Draw door edges
    edges = [(u, v) for u, v, d in G.edges(data="connectivity") if d == 1]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors[0],
                           width=w, ax=ax)

    # Draw door edges
    edges = [(u, v) for u, v, d in G.edges(data="connectivity") if d == 0]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors[1],
                           width=w, ax=ax)