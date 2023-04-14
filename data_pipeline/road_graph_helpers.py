#  Copyright 2022 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import copy
import sys
from collections import defaultdict

import geojson
import networkx as nx
import numpy as np
import osmnx as ox
import osmnx.graph
import osmnx.truncate
from data_helpers import get_intersecting_grid_cells
from h5_helpers import load_h5_file
from pyproj import Geod
from shapely.geometry import LineString
from shapely.geometry import Point
from sklearn.neighbors import BallTree

# +
geod_wgs84 = Geod(ellps="WGS84")


def distance_meters(x1, y1, x2, y2):
    _, _, dist = geod_wgs84.inv(x1, y1, x2, y2)
    return dist


def distance_node_counter(g, node, counter):
    mp = g.nodes[node]
    return distance_meters(mp["x"], mp["y"], counter.x, counter.y)


def generate_id(t):
    h = hash(t)
    if h < 0:
        h += sys.maxsize
    return h


def cut(line, distance):
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [LineString(coords[: i + 1]), LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [LineString(coords[:i] + [(cp.x, cp.y)]), LineString([(cp.x, cp.y)] + coords[i:])]


def get_edge_geometry(g, edge, edge_data=None):
    if edge not in g.edges:
        return None
    if not edge_data:
        edge_data = g.edges[edge]
    if "geometry" not in edge_data:
        n1, n2, ekey = edge
        mp1 = g.nodes[n1]
        mp2 = g.nodes[n2]
        line = LineString([Point(mp1["x"], mp1["y"]), Point(mp2["x"], mp2["y"])])
    else:
        line = edge_data["geometry"]
    return line


def check_counters_on_nodes(g, debug=True):
    counts = defaultdict(lambda: 0)
    unique_counters = set()
    for n in g.nodes:
        nd = g.nodes[n]
        if "counter_info" in nd:
            for ci in nd["counter_info"].split(","):
                unique_counters.add(ci)
            cnstr = str(nd["counter_nodes"])
            cnss = cnstr.split(",")
            counts[cnstr] += len(cnss)
    if debug:
        for k, v in counts.items():
            print(f"spread {k:3}: {v} rows")
        print(f"Graph has {len(unique_counters)} nodes with counters")
    return len(unique_counters), dict(counts)


def set_counter_on_node(g, node_id, node_dist, lon, lat, site_info, site_count=1):
    node_data = g.nodes[node_id]
    if "counter_distance_m" in node_data:
        node_data["counter_info"] = f"{node_data['counter_info']},{site_info}"
        node_data["counter_distance_m"] = f"{node_data['counter_distance_m']},{node_dist}"
        node_data["counter_lon"] = f"{node_data['counter_lon']},{lon}"
        node_data["counter_lat"] = f"{node_data['counter_lat']},{lat}"
        node_data["counter_nodes"] = f"{node_data['counter_nodes']},{site_count}"
    else:
        node_data["counter_info"] = str(site_info)
        node_data["counter_distance_m"] = str(node_dist)
        node_data["counter_lon"] = str(lon)
        node_data["counter_lat"] = str(lat)
        node_data["counter_nodes"] = str(site_count)


def split_edge(g, edge, np, site_info, split_reverse=True, snap_if_possible=True, nearest_node=None, nearest_node_dist=1e12):
    n1, n2, ekey = edge
    if edge not in g.edges:
        return

    ed = g.edges[edge]
    if "geometry" not in ed:
        mp1 = g.nodes[n1]
        mp2 = g.nodes[n2]
        ed["geometry"] = LineString([Point(mp1["x"], mp1["y"]), Point(mp2["x"], mp2["y"])])
        print(f"Added geometry to {edge}: {ed}")
    line = ed["geometry"]
    np_dist = line.project(np)
    np_orig = np
    np_fract = np_dist / line.length
    np = line.interpolate(np_dist)

    if (np_dist < 0.0001 or np_dist >= (line.length - 0.0001)) and snap_if_possible:
        ls_dist1 = distance_node_counter(g, n1, np_orig)
        ls_dist2 = distance_node_counter(g, n2, np_orig)
        if ls_dist1 < ls_dist2:
            # Snap to start node
            set_counter_on_node(g, n1, ls_dist1, np.x, np.y, site_info, site_count=1)
        else:
            # Snap to end node
            set_counter_on_node(g, n2, ls_dist2, np.x, np.y, site_info, site_count=1)
        return

    lps = cut(line, np_dist)
    l1, l2 = lps

    np_orig_dist = distance_meters(np_orig.x, np_orig.y, np.x, np.y)
    if nearest_node and np_orig_dist > nearest_node_dist:
        print(f"Distance to split point {np_orig_dist}m greater than to nearest node {nearest_node_dist}m")
        set_counter_on_node(g, nearest_node, nearest_node_dist, np_orig.x, np_orig.y, site_info, site_count=1)
        return

    # Create the new node and add it to the graph
    nn = generate_id((np.x, np.y))
    g.add_node(nn, x=np.x, y=np.y, street_count=2)
    if site_info:
        set_counter_on_node(g, nn, np_orig_dist, np_orig.x, np_orig.y, site_info, site_count=1)

    # Create the new edges and insert them to the graph, remove the old edge
    ed1 = copy.copy(ed)
    ed1.pop("key", None)
    ed1["geometry"] = l1
    ed1["length"] *= np_fract
    ed1["travel_time"] *= np_fract
    ed1["replaces_edge"] = edge
    g.add_edge(n1, nn, **ed1)
    ed2 = copy.copy(ed)
    ed2["geometry"] = l2
    ed2["length"] *= 1 - np_fract
    ed2["travel_time"] *= 1 - np_fract
    ed2["replaces_edge"] = edge
    g.add_edge(nn, n2, **ed2)
    g.remove_edge(n1, n2, key=ekey)
    oneway = ed["oneway"]
    assert type(oneway) == bool
    if split_reverse and not oneway:
        # If we add the reverse edges we need to make sure to use the same data as we're in a directed graph
        # https://github.com/gboeing/osmnx/blob/d2e0ab396b53988b0107907f61e60f0accf5cf00/osmnx/utils_graph.py#L382
        g.add_edge(nn, n1, **ed1)
        g.add_edge(n2, nn, **ed2)
        if (n2, n1, ekey) in g.edges:
            g.remove_edge(n2, n1, key=ekey)


def split_edges(g, edges_and_pts, split_reverse=True, snap_if_possible=True):
    missing_edges = []
    for e_and_pt in edges_and_pts:
        if len(e_and_pt) == 7:
            edge, _, lon, lat, site_info, nearest_node, nearest_node_dist = e_and_pt
        else:
            edge, _, lon, lat, site_info = e_and_pt
            nearest_node = None
            nearest_node_dist = 1e12
        # Compute the location of the new point on the edge
        counter_pt = Point(lon, lat)
        edge_line = get_edge_geometry(g, edge)
        if not edge_line:
            missing_edges.append((lon, lat, site_info))
            continue

        # Split the road geometry at the new node, try also for the inverse direction
        split_edge(
            g,
            edge,
            counter_pt,
            site_info,
            split_reverse=split_reverse,
            snap_if_possible=snap_if_possible,
            nearest_node=nearest_node,
            nearest_node_dist=nearest_node_dist,
        )
    return missing_edges


# +
LENGTH_TOLERANCE = 50  # do not delete shorter roads to avoid deleting e.g. roundabouts upfront
MIN_MAX_VOLUME = 10  # delete roads with less volume in the corresponding heading channel
FILTER_HIGHWAY_CLASSES = ["residential", "unclassified"]


def print_counts(g, prefix=None):
    if prefix:
        print(f"{prefix}: graph has {len(g.nodes)} nodes and {len(g.edges)} edges")
    else:
        print(f"Graph has {len(g.nodes)} nodes and {len(g.edges)} edges")


def edge_has_counter(g, edge):
    n1, n2, _ = edge
    if "counter_info" in g.nodes[n1]:
        return True
    if "counter_info" in g.nodes[n2]:
        return True
    return False


def is_filter_candidate(g, edge, length_tolerance, highway_classes=FILTER_HIGHWAY_CLASSES):
    ed = g.edges[edge]
    highway = ed["highway"]
    if type(highway) == list:
        if not any(hw in highway_classes for hw in highway):
            return False
    elif highway not in highway_classes:
        return False
    if length_tolerance > 0 and ed["length"] < length_tolerance:
        return False
    if edge_has_counter(g, edge):
        return False
    return True


def filter_edge_candidates(g, edges, length_tolerance=-1, highway_classes=FILTER_HIGHWAY_CLASSES):
    filtered = []
    for edge in edges:
        if is_filter_candidate(g, edge, length_tolerance=length_tolerance, highway_classes=highway_classes):
            filtered.append(edge)
    return filtered


def compute_max_daily_volume(cells, heatmap):
    max_daily_vol = 0
    for row, column, heading, _ in cells:
        heading_volume = heatmap[row, column, heading]
        max_daily_vol = max(max_daily_vol, heading_volume)
    return max_daily_vol


def clean_edges_with_low_volume(g, heatmap, rotate, lon_min, lat_min, min_max_volume=MIN_MAX_VOLUME):
    delete_edges = []
    for edge in g.edges:
        if not is_filter_candidate(g, edge, length_tolerance=LENGTH_TOLERANCE):
            continue

        line = get_edge_geometry(g, edge)
        cells_forward = get_intersecting_grid_cells(line, lon_min, lat_min, reverse=False, rotate=rotate)
        max_daily_vol_forward = compute_max_daily_volume(cells_forward, heatmap)
        cells_reverse = get_intersecting_grid_cells(line, lon_min, lat_min, reverse=True, rotate=rotate)
        max_daily_vol_reverse = compute_max_daily_volume(cells_reverse, heatmap)
        max_daily_vol = max(max_daily_vol_forward, max_daily_vol_reverse)

        if max_daily_vol < min_max_volume:
            delete_edges.append(edge)

    print(f"Found {len(delete_edges)}/{len(g.edges)} segments to delete (low volume)")
    g.remove_edges_from(delete_edges)
    return g


invalid_access_chars = ["'", '"', "[", "]", " "]
restricted_access = ["no", "private", "official", "permit", "delivery", "designated", "emergency"]


def clean_edges_with_no_access(g):
    delete_edges = []
    for edge in g.edges:
        if not is_filter_candidate(g, edge, length_tolerance=-1):
            continue
        ed = g.edges[edge]
        access = str(ed["access"]) if "access" in ed else ""
        access = "".join(x for x in access if x not in invalid_access_chars)
        for x in access.split(","):
            if x in restricted_access:
                delete_edges.append(edge)
    print(f"Found {len(delete_edges)}/{len(g.edges)} segments to delete (no access)")
    g.remove_edges_from(delete_edges)
    return g


def _clean_dead_end_edges(g, dead_ends):
    set(dead_ends)
    dead_end_edges = list(g.in_edges(dead_ends, keys=True))
    dead_end_edges += list(g.out_edges(dead_ends, keys=True))
    print(f"Found {len(dead_ends)} dead-ends and {len(dead_end_edges)} edge candidates")
    delete_edges = filter_edge_candidates(g, dead_end_edges)
    print(f"Deleting {len(delete_edges)} edges")
    g.remove_edges_from(delete_edges)
    return len(delete_edges)


def clean_dead_end_edges(g):
    edges_deleted = 1
    while edges_deleted > 0:
        streets_per_node = ox.stats.count_streets_per_node(g)
        dead_end_nodes = [node for node, count in streets_per_node.items() if count == 1]
        edges_deleted = _clean_dead_end_edges(g, dead_end_nodes)
    return g


def clean_isolates(g):
    isolates = list(nx.isolates(g))
    print(f"Found {len(isolates)} isolates to remove")
    print_counts(g, "Before")
    g.remove_nodes_from(isolates)
    print_counts(g, "After")
    return g


def clean_self_loops(g):
    cycles = list(nx.selfloop_edges(g, keys=True))
    print(f"Found {len(cycles)} self-loops candidates to remove")
    cycles = filter_edge_candidates(g, cycles, length_tolerance=300)  # we keep large loops
    print(f"Deleting {len(cycles)} self-loops")
    print_counts(g, "Before")
    g.remove_edges_from(cycles)
    print_counts(g, "After")
    return g


def get_all_neighbors(g, n):
    return list(set(list(g.successors(n)) + list(g.predecessors(n))))


def clean_no_neighbors(g):
    no_neighbors = [n for n in g.nodes if len(get_all_neighbors(g, n)) in [1]]
    print(f"Found {len(no_neighbors)} no_neighbors to remove")
    print_counts(g, "Before")
    _clean_dead_end_edges(g, no_neighbors)
    print_counts(g, "After")
    return g


def clean_sub_graphs(g):
    sub_graphs = sorted(nx.connected_components(g.to_undirected()), key=len, reverse=True)
    print(f"Main graph has {len(sub_graphs[0])} nodes")
    print(f"Largest sub-graph has {len(sub_graphs[1])} nodes")
    sub_graphs = sub_graphs[1:]
    subgraph_nodes = [n for sg in sub_graphs for n in sg]
    print(f"Found {len(sub_graphs)} with {len(subgraph_nodes)} nodes to remove")
    print_counts(g, "Before")
    g.remove_nodes_from(subgraph_nodes)
    print_counts(g, "After")
    return g


def get_edge_data(g, n1, n2):
    try:
        return g.edges[(n1, n2, 0)]
    except Exception:
        try:
            return g.edges[(n2, n1, 0)]
        except Exception:
            return None


def get_length(g, n1, n2):
    ed = get_edge_data(g, n1, n2)
    if not ed:
        return 1000
    return ed["length"]


def clean_circle_ramps(g, highway_classes=FILTER_HIGHWAY_CLASSES):  # noqa: C901
    ramp_candidates = []
    for edge in g.edges:
        ed = g.edges[edge]
        length = ed["length"]
        if length > 30:
            continue

        n1, n2, key = edge
        neighbors1 = get_all_neighbors(g, n1)
        if len(neighbors1) != 3:
            continue
        neighbors2 = get_all_neighbors(g, n2)
        if len(neighbors2) != 3:
            continue

        commons = [x for x in neighbors1 if x in neighbors2]
        if len(commons) != 1:
            continue
        cn = commons[0]
        if len(get_all_neighbors(g, cn)) != 2:
            continue

        ed1 = get_edge_data(g, n1, cn)
        if ed1["highway"] not in highway_classes:
            continue
        length1 = ed1["length"]

        ed2 = get_edge_data(g, n2, cn)
        if ed2["highway"] not in highway_classes:
            continue
        length2 = ed2["length"]

        if length1 > 50 or length2 > 50:
            continue
        if abs(length1 - length2) > (length1 * 0.2):
            continue
        ramp_candidates.append(cn)

    print(f"Found {len(ramp_candidates)} ramps")
    print_counts(g, "Before")
    g.remove_nodes_from(ramp_candidates)
    print_counts(g, "After")
    return g


def traverse_neighbors(g, node, node_list, num_neighbors_greater_two=0):
    if len(node_list) == 5:
        return False
    node_list[-1] if node_list else None
    node_list.append(node)
    neighbors = get_all_neighbors(g, node)
    if len(neighbors) > 2:
        num_neighbors_greater_two += 1
        if num_neighbors_greater_two > 1:
            return False
    for n in neighbors:
        length = get_length(g, node, n)
        if length > 150:
            return False
        if len(node_list) > 2 and n == node_list[0]:
            return True
        if n in node_list:
            continue
        return traverse_neighbors(g, n, node_list, num_neighbors_greater_two)
    return False


def clean_end_circles(g):
    circle_count = 0
    circle_candidates = []
    for edge in g.edges:
        ed = g.edges[edge]
        length = ed["length"]
        if length > 20:
            continue
        n1, n2, key = edge
        node_list = []
        is_circle = traverse_neighbors(g, n1, node_list)
        if is_circle:
            circle_count += 1
            circle_candidates.extend(node_list)
    print(f"Found {circle_count} end circles")
    print_counts(g, "Before")
    g.remove_nodes_from(circle_candidates)
    print_counts(g, "After")
    return g


def fix_duplicate_osm_edges(g, multi_edges):
    for edges in multi_edges:
        line1 = get_edge_geometry(g, edges[0])
        for edge in edges[1:]:
            line = get_edge_geometry(g, edge)
            equal_edge = line1 == line
            if equal_edge:
                g.remove_edge(edge[0], edge[1], key=edge[2])


def is_edge_unique(g, edge):
    u, v, key = edge
    for k in range(key):
        if (u, v, k) in g.edges:
            return False
    return True


def get_multi_edges(g, edge):
    # Returns all multi edges sorted by length and priority
    u, v, key = edge
    candidates = []
    for k in range(key + 1):
        if (u, v, k) in g.edges:
            ed = g.edges[(u, v, k)]
            highway = str(ed["highway"])
            prio = ed["length"]
            if "residential" in highway or "unclassified" in highway:
                prio += 10000
            candidates.append((prio, (u, v, k)))
    candidates = sorted(candidates)
    return [c[1] for c in candidates]


def find_multi_edges(g):
    multi_edges = defaultdict(lambda: [])
    for edge in g.edges:
        u, v, key = edge
        if key == 0:
            continue
        if is_edge_unique(g, edge):
            continue
        candidate = get_multi_edges(g, edge)
        if (u, v) in multi_edges:
            # If there are more than two edges, we only keep the longest one
            if len(multi_edges[(u, v)]) < len(candidate):
                multi_edges[(u, v)] = candidate
        else:
            multi_edges[(u, v)] = candidate
    print(len(multi_edges))
    all_multi_edges = [x for me in multi_edges.values() for x in me]
    print(len(all_multi_edges))
    return list(multi_edges.values())


def split_multi_edges(g, multi_edges):
    edges_and_pts = []
    for edges in multi_edges:
        # Split all edges but the first one (sorted above)
        for edge in edges[1:]:
            g.edges[edge]
            line = get_edge_geometry(g, edge)
            coords = list(line.coords)
            if len(coords) > 2:
                lon, lat = coords[int(len(coords) / 2)]
            else:
                pt = line.interpolate(line.length / 2)
                lon, lat = pt.coords[0]
            edges_and_pts.append((edge, 0, lon, lat, None))
    print(f"Found {len(edges_and_pts)} to split")
    split_edges(g, edges_and_pts, split_reverse=False, snap_if_possible=False)


def fix_keys(g):
    update_edges = []
    for edge in g.edges:
        u, v, k = edge
        if k != 0:
            update_edges.append(edge)
    for edge in update_edges:
        u, v, k = edge
        ed = g.edges[edge]
        g.remove_edge(u, v, key=k)
        g.add_edge(u, v, **ed)


def clean_multi_edges(g):
    print_counts(g, "Before removing OSM duplicates")
    multi_edges = find_multi_edges(g)
    fix_duplicate_osm_edges(g, multi_edges)
    print_counts(g, "Before splitting edges")
    multi_edges = find_multi_edges(g)
    split_multi_edges(g, multi_edges)
    print_counts(g, "After splitting edges")
    fix_keys(g)
    print_counts(g, "After fixing keys")
    return g


def check_graph_directed(g):
    g_directed = nx.DiGraph(g)
    print_counts(g_directed)
    assert len(g_directed.edges) == len(g.edges)
    assert len(g_directed.nodes) == len(g.nodes)


def is_counter(g, n):
    nd = g.nodes[n]
    if "counter_info" in nd:
        return True
    else:
        return False


def compute_depth_from_counter(g, nodes_to_visit, depth):
    if len(nodes_to_visit) == 0 or depth > 200:
        return
    if depth % 10 == 0 or depth > 70:
        print(f"{depth}: {len(nodes_to_visit)}")
    nodes_to_visit_next = []
    for ntv in nodes_to_visit:
        g.nodes[ntv]["depth"] = depth
        for n in get_all_neighbors(g, ntv):
            nd = g.nodes[n]
            if "counter_info" in nd:
                continue
            if "depth" in nd and nd["depth"] <= depth:
                continue
            nodes_to_visit_next.append(n)
    compute_depth_from_counter(g, list(set(nodes_to_visit_next)), depth + 1)


def process_heatmap_filter(g, heatmap_path, rotate, lon_min, lat_min):
    heatmap = load_h5_file(heatmap_path)
    print(f"Loaded heatmap with type {heatmap.dtype}")
    heatmap = heatmap.astype("float64")

    if "simplified" not in g.graph or not g.graph["simplified"]:
        g = ox.simplify_graph(g)  # simplification is required before filtering.
    print(f"Simplified road graph is {g}")

    g = clean_edges_with_no_access(g)
    g = clean_edges_with_low_volume(g, heatmap, rotate, lon_min, lat_min)
    g = clean_dead_end_edges(g)
    g = clean_isolates(g)
    g = clean_self_loops(g)
    g = clean_isolates(g)  # needs to be repeated again after self-loops removal.
    g = clean_dead_end_edges(g)  # needs to be repeated again after self-loops removal.
    g = clean_no_neighbors(g)
    g = clean_sub_graphs(g)
    g = clean_dead_end_edges(g)  # needs to be repeated again after no-neighbors and sub-graphs.
    for _ in range(4):  # repeat ramp and end-circle removel 4 times to also catch nested ones.
        g = clean_circle_ramps(g)
        g = clean_end_circles(g)
    g = clean_isolates(g)  # needs to be repeated again after circle and ramp removal.
    g = clean_multi_edges(g)
    check_graph_directed(g)
    compute_depth_from_counter(g, [n for n in g.nodes if is_counter(g, n)], 0)
    return g


# +
def is_counter_blacklisted(blacklist, x, y):
    key = str(x)[:9] + "," + str(y)[:9]
    return key in blacklist


def load_counters(counter_geojson, counter_blacklist=None):
    if not counter_blacklist:
        counter_blacklist = []
    blacklisted_count = 0
    with open(counter_geojson, "r") as f:
        fc = geojson.load(f)
        for f in fc["features"]:
            lat = f["geometry"]["coordinates"][1]
            lon = f["geometry"]["coordinates"][0]
            if is_counter_blacklisted(counter_blacklist, lon, lat):
                blacklisted_count += 1
                continue
            id = f["properties"]["id"]
            yield lat, lon, id
    print(f"Cleaned {blacklisted_count} blacklisted counters")
    assert blacklisted_count == len(counter_blacklist)


# Adopted from https://github.com/gboeing/osmnx/blob/main/osmnx/distance.py
def get_k_nearest_nodes(g, x, y, k=10, min_dist=40, max_dist_diff_factor=0.5):
    EARTH_RADIUS_M = 6_371_009
    X = np.array(x)
    Y = np.array(y)
    nodes = ox.utils_graph.graph_to_gdfs(g, edges=False, node_geometry=False)[["x", "y"]]
    nodes_rad = np.deg2rad(nodes[["y", "x"]])
    points_rad = np.deg2rad(np.array([Y, X]).T)
    dists, poss = BallTree(nodes_rad, metric="haversine").query(points_rad, k=k)
    dists = dists * EARTH_RADIUS_M  # convert radians -> meters
    # Converting index to numpy (instead of nodes.index[poss]) to avoid future warning on multi-dimensional indexing
    nns = nodes.index.to_numpy()[poss]
    res_nns = []
    res_dists = []
    for nn, dd in zip(nns, dists):
        res_nn = []
        res_dd = []
        prev_dist = dd[0]
        for n, d in zip(nn, dd):
            if d > min_dist:
                break
            if d - prev_dist > prev_dist * max_dist_diff_factor:
                break
            res_nn.append(n)
            res_dd.append(d)
        res_nns.append(res_nn)
        res_dists.append(res_dd)
    return res_nns, res_dists


def process_counter_merge(g, counter_geojson, assign_all_neighbors=False, counter_blacklist=None):
    Xs = []
    Ys = []
    Ms = []
    for lat, lon, site_info in load_counters(counter_geojson, counter_blacklist):
        Xs.append(lon)
        Ys.append(lat)
        Ms.append(site_info)

    # Check the nodes and collect edges that need to be split (eXs, eYs and eMs)
    iteration = 0
    while len(Xs) > 0:
        k_nearest = 1
        if assign_all_neighbors:
            k_nearest = 10
        print(f"Searching nearest nodes for {len(Xs)} counters (interation {iteration})")
        node_ids, node_dists_m = get_k_nearest_nodes(g, Xs, Ys, k=k_nearest, min_dist=5000)
        edge_ids, edge_dists_m = ox.distance.nearest_edges(g, Xs, Ys, return_dist=True)
        edges_and_pts = []
        for node_id, node_dist, edge_id, edge_dist, lon, lat, site_info in zip(node_ids, node_dists_m, edge_ids, edge_dists_m, Xs, Ys, Ms):
            if not node_dist or node_dist[0] > 40:
                edges_and_pts.append((edge_id, edge_dist, lon, lat, site_info, node_id[0], node_dist[0]))
            else:
                # E.g. in Melbourne if len(node_id) > 1 we distribute the counter to all nodes
                if assign_all_neighbors:
                    for n_id, n_dist in zip(node_id, node_dist):
                        set_counter_on_node(g, n_id, n_dist, lon, lat, site_info, site_count=len(node_id))
                else:
                    set_counter_on_node(g, node_id[0], node_dist[0], lon, lat, site_info)

        if len(edges_and_pts) > 0:
            print(f"Found {len(edges_and_pts)}/{len(Xs)} counters with distance > 40m to next node")

        missing_edges = split_edges(g, edges_and_pts)
        if not missing_edges:
            break
        Xs, Ys, Ms = zip(*missing_edges)
        iteration += 1

    counter_count, counter_counts = check_counters_on_nodes(g, debug=False)
    print(f"Added {counter_count} counters to the road graph: {counter_counts}")
    return g
