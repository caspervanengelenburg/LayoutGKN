"""
tl;dr: Extracts richly-attributed access graphs from the original RPLAN images.

In order:
RPLAN image are converted into networkx graphs
The data is, subsequently, split into train, validation, and test part
Networkx graphs are converted into pytorch geometric graphs.
"""

import re, os, numpy as np, networkx as nx, random
from tqdm import tqdm
from itertools import combinations
from rasterio import features
from shapely import geometry, affinity
from shapely.geometry import Polygon, MultiPolygon, LineString, box
from torch_geometric.utils import from_networkx
from grakel import Graph
from grakel.kernels import GraphHopper
import torch, torch.nn.functional as F

from LayoutGKN.utils import polygon_to_list, load_image_rplan, save_pickle
from LayoutGKN.constants import CAT_RPLAN_ORIG, CAT_MAP
from LayoutGKN.config import load_cfg
from LayoutGKN.constants import CAT_RPLAN


# globs
POLY_IMAGE = Polygon(((0, 0), (0, 256), (256, 256), (256, 0), (0, 0)))
# for radial basis function
MU = 0.5
EPS = 1e-12


def check_validity_polygon(poly):
    """Check the validity of a room polygon.
    The polygon shouldn't contain the whole image mask."""
    if poly == POLY_IMAGE: return False
    poly_mapped = geometry.mapping(poly)
    if len(poly_mapped) != 1:
        for subpoly in poly_mapped["coordinates"]:
            subpoly = Polygon(subpoly)
            if subpoly == POLY_IMAGE: return False
            else: return True


def extract_shapes_from_mask(mask, min_area=10):
    """Extracts all possible shapes from a binary mask."""
    shapes = features.shapes(mask, connectivity=4)
    polygons = []
    for s, _ in shapes:
        poly = Polygon(geometry.shape(s))
        if min_area < poly.area and check_validity_polygon(poly):
            polygons.append(poly)
    return polygons


def extract_rooms_from_image(img,
                             cat_rplan_orig=CAT_RPLAN_ORIG,
                             cat_map=CAT_MAP,  # category mapping from original RPLAN to new
                             min_area=10):
    """Extract room shapes from RPLAN images."""
    polygons = []
    categories = []
    classes = list(cat_rplan_orig.keys())
    for y in classes:
        if y > 12: continue  # > 12 are not room types
        mask = (img == y).astype(np.uint8)
        polygons_y = extract_shapes_from_mask(mask, min_area=min_area)
        polygons.extend(polygons_y)
        categories.extend([cat_map[y]]*len(polygons_y))  # convert category to new category
    return polygons, categories


def split_rectilinear(poly):
    """Split an axis-aligned rectilinear polygon (e.g., L-shape) into rectangles."""
    def _split_one(P):
        xs = sorted({round(x, 8) for x, _ in P.exterior.coords})
        ys = sorted({round(y, 8) for _, y in P.exterior.coords})
        rects = []
        for x0, x1 in zip(xs, xs[1:]):
            for y0, y1 in zip(ys, ys[1:]):
                cell = box(x0, y0, x1, y1)
                if P.contains(cell):            # keep full cells entirely inside
                    rects.append(cell)
        return rects or [P]
    if isinstance(poly, MultiPolygon):
        out = []
        for p in poly.geoms:
            out.extend(_split_one(p))
        return out
    return _split_one(poly)


def n_largest_polygons(polygons, n=2):
    """Finds the n largest polygons given a list of polygons."""
    return sorted(polygons, key=lambda p: p.area, reverse=True)[:n]


def extract_doors_from_image(img):
    """Extract door shapes from RPLAN images."""
    mask = (img == 17).astype(np.uint8)
    polygons = extract_shapes_from_mask(mask)
    splits = []
    for poly in polygons:
        # splits L-joint geometries (1st) into TWO (2nd line) doors
        polys = split_rectilinear(poly)
        if len(polys) > 1: polys = n_largest_polygons(polys, n=2)
        splits.extend(polys)
    return splits


def transform_polygon(poly, shift=128, scale=1/256 * 18/10):
    """Transforms polygon based on a shift and scale."""
    poly_temp = np.array(poly.exterior.coords) - np.array(shift)
    poly_temp *= scale
    return Polygon(poly_temp)


def rotate_polygon_90k(poly, k=1):
    """Rotate a Polygon by 90°*k around its center (centroid)."""
    return affinity.rotate(poly, 90*(k % 4), origin='centroid')


def midline_longest_side(poly):
    """Line through the polygon center."""
    rect = poly.minimum_rotated_rectangle
    xs = list(rect.exterior.coords)[:4]
    e = max([(xs[i], xs[(i+1)%4]) for i in range(4)],
        key=lambda ab: np.hypot(ab[1][0]-ab[0][0], ab[1][1]-ab[0][1]))
    L = np.hypot(e[1][0]-e[0][0], e[1][1]-e[0][1]) or 1e-12
    ux, uy = (e[1][0]-e[0][0])/L, (e[1][1]-e[0][1])/L
    c = rect.centroid
    return LineString([(c.x-0.5*L*ux, c.y-0.5*L*uy), (c.x+0.5*L*ux, c.y+0.5*L*uy)])


def create_orthogonal_lines(p: Polygon, N: int):
    """Generates a set of N lines orthogonal to A,
    centered on line A, uniformly spaced along A, and all with length L.
    They do not start at the end-points but at length(A)/(2N) from the end-points."""
    l = midline_longest_side(p)
    L = l.length
    margin = L / N / 2
    distances = np.linspace(margin, L - margin, N)
    delta = 1e-6
    orthogonal_lines = []
    for d in distances:
        p1 = l.interpolate(max(0, d - delta))
        p2 = l.interpolate(min(L, d + delta))
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        # Orthogonal vector
        ox, oy = -dy, dx
        norm = np.hypot(ox, oy)
        ox, oy = ox / norm, oy / norm
        center = l.interpolate(d)
        offset = L / 2
        p_start = (center.x - ox * offset, center.y - oy * offset)
        p_end = (center.x + ox * offset, center.y + oy * offset)
        orthogonal_lines.append(LineString([p_start, p_end]))
    return orthogonal_lines


def get_geometry_feats(polygon):
    """Get geometrical features from polygon.
    Returns them in a list."""
    box = np.array(polygon.bounds)
    w = box[2] - box[0]  # width
    h = box[3] - box[1]  # height
    a =  np.sqrt(polygon.area)  # area
    cx, cy = polygon.centroid.xy  # center
    p = polygon.length / 4  # perimeter
    return np.array([cx[0], cy[0], w, h, a, p])


def check_access(p1, p2, polygons_door):
    """Checks if two room polygons are connected by a door."""
    for p_door in polygons_door:
        # creates N orthogonal lines w.r.t. door geometry
        lines = create_orthogonal_lines(p_door, N=3)
        if any([l.intersects(p1) and l.intersects(p2) for l in lines]): return True
    return False


def check_adjacency(p1, p2, b=0.05, aspect=2.0):
    """Checks if two room polygons are adjacent.
    It basically check whether the buffered overlap looks like a boundary segment:
    a strip, not a point / square-like geometry. Implemented as that the bounding box of the
    overlapping segment after dilating both room geometries should have
    a minimal aspect ratio between longest and shortest side."""
    segment = p1.buffer(b).intersection(p2.buffer(b))
    segment = split_rectilinear(segment)
    if len(segment) > 2: segment = n_largest_polygons(segment, n=1)[0]
    else: segment = segment[0]
    if segment.is_empty: return False
    segment_mrr = segment.minimum_rotated_rectangle
    xs = list(segment_mrr.exterior.coords)
    sides = [LineString([xs[i], xs[i+1]]).length for i in range(4)]
    return max(sides)/min(sides) >= aspect


def extract_access_graph(pid, polygons, categories, polygons_door):
    """Extracts the access graph from a set of room geometries
    and door geometries, including the room categories."""
    nodes = {}
    for n, (p, cat) in enumerate(zip(polygons, categories)):
        nodes[n] = {
            'polygon': polygon_to_list(p),
            'category': cat,
            'geometry': get_geometry_feats(p)
        }
    edges = []
    for i, j in combinations(nodes.keys(), 2):
        p1 = Polygon(nodes[i]['polygon'])
        p2 = Polygon(nodes[j]['polygon'])
        if check_access(p1, p2, polygons_door): edges.append([i, j, {"connectivity": 1}])
        elif check_adjacency(p1, p2): edges.append([i, j, {"connectivity": 0}])
        else: continue
    G = nx.Graph()
    G.graph["pid"] = pid  # plan ID as graph-level attribute
    G.add_nodes_from([(u, v) for u, v in nodes.items()])
    G.add_edges_from(edges)
    return G


def check_connectedness(G):
    """Checks whether an access graph is connected w.r.t. access.
    It checks, thus, whether each room could be reached from all others."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    # only add "door" edges to new graph
    edges = [(u, v) for u, v, d in G.edges(data=True) if d["connectivity"]==1]
    H.add_edges_from(edges)
    return len(list(nx.connected_components(H))) == 1


# DO NOT CHANGE SEED = 42
def split_ids(ids, seed=42):
    """Splits IDs into train, val, test (0.7, 0.2, 0.1)."""
    random.seed(seed); random.shuffle(ids)
    n = len(ids); a, b = int(n*0.7), int(n*0.9)
    return ids[:a], ids[a:b], ids[b:]


def nx_to_pyg(G):
    """Converts networkx graph to Pytorch Geometric graph."""
    G = G.copy()
    for n, d in G.nodes(data="geometry"):
        G.nodes[n]["geometry"] = torch.tensor(d)
    G = remove_attributes_from_graph(G, node_attr=["polygon"])
    return from_networkx(G)


def remove_attributes_from_graph(graph, node_attr=['polygon'], edge_attr=[]):
    """Removes attribute(s) from graph"""
    for attr in node_attr:
        for n in graph.nodes(): # delete irrelevant node features
            try: del graph.nodes[n][attr]
            except: pass
    for attr in edge_attr:
        for u, v in graph.edges(): # delete irrelevant edge features
            try: del graph.edges[u, v][attr]
            except: pass
    return graph


def pyg_to_grakel(G):
    """Converts a PyG graph to a Grakel-compatible graph format."""
    edge_index = G.edge_index.cpu().numpy()
    mask = G.connectivity.cpu().numpy() == 1  # only when connected by door!
    edge_index = edge_index[:, mask]
    num_nodes = G.num_nodes
    edges = {i: [] for i in range(num_nodes)}
    for u, v in edge_index.T:
        edges[u].append(v)
    node_attributes = {i: 0 for i in range(num_nodes)}
    return Graph(edges, node_labels=node_attributes)


def add_combined_vector(G):
    geom = G.geometry
    cats = F.one_hot(G.category, num_classes=len(CAT_RPLAN))
    # Only normalize geometric attribute
    mean = geom.mean(dim=0, keepdim=True)
    std = geom.std(dim=0, unbiased=False, keepdim=True).clamp_min(EPS)
    geom_norm = (geom - mean) / std
    # Concatenate both
    G.vecs = torch.cat([geom_norm, cats], dim=1)
    return G


def get_shortest_path_matrices(graphs):
    """Computes the shortest path matrices for all nodes in the graphs"""
    # Converts all PyG graphs to GraKel-compatible graphs
    graphs_G = [pyg_to_grakel(G) for G in graphs]
    # Sets kernel (arbitrary values are fine)
    d = 19; sigma = (d / 2) ** 0.5; mu = 1 / (2 * (sigma ** 2))
    gh = GraphHopper(kernel_type=("gaussian", mu))

    # Get shortest-path (ShP) matrices
    gh._method_calling = 1; gh._max_diam = 5; gh.calculate_norm_ = False
    outs = gh.parse_input(graphs_G)
    return outs


def main():
    cfg = load_cfg()
    dir_save = os.path.join(cfg.path_data, "rplan")
    print(f"Directory for saving exists? {os.path.exists(dir_save)}")

    # 1. Extract access graphs from RPLAN images
    ids = [int(re.search(r"\d+", p).group()) for p in os.listdir(cfg.path_rplan)]
    graphs = []
    for pid in tqdm(ids):
        try:
            # load image and flip it vertically
            img = load_image_rplan(pid, cfg.path_rplan)
            img = np.flipud(img)
            # extract room geometries / semantics, and door geometries
            polygons, categories = extract_rooms_from_image(img)
            polygons_door = extract_doors_from_image(img)
            polygons = [transform_polygon(p) for p in polygons]
            polygons_door = [transform_polygon(p) for p in polygons_door]
            # extract access graph
            G = extract_access_graph(pid, polygons, categories, polygons_door)
            graphs.append(G)
        except:
            print(f"Couldn't successfully extract graph for plan ID = {pid}")
    save_pickle(graphs, os.path.join(dir_save, "nx_graphs.pkl"))

    # 2. Convert NX graphs into PYG graphs (and remove non-connected plans)
    ids = [G.graph["pid"] for G in graphs]
    # Check graphs' validity: is each room reachable from any other?
    ids_valid = []
    print(f"Checking connectedness of graphs ...")
    for G in tqdm(graphs):
        if check_connectedness(G): ids_valid.append(G.graph["pid"])
    # splits IDs into train, val, test
    train_ids, val_ids, test_ids = split_ids(ids_valid)
    train_set, val_set = set(train_ids), set(val_ids)
    # aggregates list of PyG graphs for training, validation, and test
    print(f"Converting NX graphs to PyG graphs and splitting them in train/val/test ...")
    pyg_graphs_train, pyg_graphs_val, pyg_graphs_test = [], [], []
    for pid in tqdm(ids_valid):
        G = graphs[ids.index(pid)]
        G_pyg = nx_to_pyg(G)
        if pid in train_set:
            pyg_graphs_train.append(G_pyg)
        elif pid in val_set:
            pyg_graphs_val.append(G_pyg)
        else:
            pyg_graphs_test.append(G_pyg)
    torch.save((train_ids, pyg_graphs_train), os.path.join(dir_save, "pyg_graphs_train.pt"))
    torch.save((val_ids, pyg_graphs_val), os.path.join(dir_save, "pyg_graphs_val.pt"))
    torch.save((test_ids, pyg_graphs_test), os.path.join(dir_save, "pyg_graphs_test.pt"))

    # 3. Add shortest-path histogram matrix for GraphHopper kernel loss
    for mode in ["train", "val", "test"]:
        if mode == "train": pyg_graphs = pyg_graphs_train
        elif mode == "val": pyg_graphs = pyg_graphs_val
        else: pyg_graphs = pyg_graphs_test
        outs = get_shortest_path_matrices(graphs)

        # Adds shortest path matrices (for a given max path length) and adds them to the graph
        delta = 4  # max path length considered
        for (shp, _), G in tqdm(zip(outs, pyg_graphs)):
            shp = torch.tensor(shp[:, :delta, :delta])
            G.shp = shp.view(shp.shape[0], -1)  # flatten the shortest-path matrices

        torch.save((ids, graphs), os.path.join(dir_save, f"pyg_graphs_{mode}_Ms.pt"))
        print("Saved SHP-attributed graphs at <<<../data/pyg_graphs_{mode}_Ms.pt>>>\n")


if __name__ == "__main__":
    main()