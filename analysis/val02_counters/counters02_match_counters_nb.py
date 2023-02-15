# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python [conda env:t4c] *
#     language: python
#     name: conda-env-t4c-py
# ---

# +
import pandas as pd
import numpy as np
import geojson
import geopandas
import osmnx as ox
import networkx as nx
from pathlib import Path
from sklearn.neighbors import BallTree
from datetime import datetime, timedelta
from geographiclib.geodesic import Geodesic
from shapely.geometry import Point, LineString

import warnings
warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')

RELEASE_PATH = Path('/private/data/mets10/release20221026_residential_unclassified')
COUNTER_PATH = Path('/private/data/mets10/loop_counters')


# +
# Adopted from https://github.com/gboeing/osmnx/blob/main/osmnx/distance.py in order to return
# multiple candidate edges.
def get_k_nearest_edges(g, x, y, k=5, min_dist=40):
    EARTH_RADIUS_M = 6_371_009
    X = np.array(x)
    Y = np.array(y)
    
    geoms = ox.utils_graph.graph_to_gdfs(g, nodes=False)["geometry"]

    # interpolate points along edges to index with k-d tree or ball tree
    uvk_xy = list()
    for uvk, geom in zip(geoms.index, geoms.values):
        # Interpolate every ~5 meters
        uvk_xy.extend((uvk, xy) for xy in ox.utils_geo.interpolate_points(geom, 0.00005))
    labels, xy = zip(*uvk_xy)
    vertices = pd.DataFrame(xy, index=labels, columns=["x", "y"])
    
    search_k = k
    if k > 1:
        # If more than one result is desired the haversine query needs to select more candidates
        # as every edge will consist of many points of which many might be close.
        search_k = min(int((len(vertices) / len(geoms)) * 1.2) * k, len(vertices))

    if BallTree is None:  # pragma: no cover
        raise ImportError("scikit-learn must be installed to search an unprojected graph")
    # haversine requires lat, lng coords in radians
    vertices_rad = np.deg2rad(vertices[["y", "x"]])
    points_rad = np.deg2rad(np.array([Y, X]).T)
    dists, poss = BallTree(vertices_rad, metric="haversine").query(points_rad, k=search_k)
    dists = dists * EARTH_RADIUS_M  # convert radians -> meters
    nes = vertices.index.to_numpy()[poss]
    
    res_nes = []
    res_dists = []
    for ne, dist in zip(nes, dists):
        ne = pd.Series(ne)
        dist = pd.Series(dist)
        # Remove duplicates
        mask = ne.duplicated()
        ne = ne[~mask]
        dist = dist[~mask]
        # Remove points too far
        mask = dist > min_dist
        ne = ne[~mask]
        dist = dist[~mask]
        # Cap selection to k elements and add to the output.
        # BallTree query results are already sorted by default, so no need to do it here.
        ne = ne[:k]
        dist = dist[:k]
        res_nes.append(ne.tolist())
        res_dists.append(dist.tolist())

    return res_nes, res_dists


# In-notebook unit tests
def test_get_k_nearest_edges():
    g = nx.MultiDiGraph(crs="epsg:4326")
    g.add_nodes_from([(1, {'x': 0.1, 'y': 0.1}), (2, {'x': 0.18, 'y': 0.1}), (3, {'x': 0.14, 'y': 0.17})])
    g.add_edges_from([(1, 2), (1, 3), (2, 3)])
    
    print("get_k_nearest_edges(g, [0.12], [0.12], k=1, min_dist=4000)")
    nn, dd = get_k_nearest_edges(g, [0.12], [0.12], k=1, min_dist=4000)
    print((nn, dd))
    assert(nn == [[(1, 3, 0)]])
    np.testing.assert_almost_equal(dd, [[827.5247451916221]])
    
    print("get_k_nearest_edges(g, [0.12], [0.12], k=2, min_dist=8000)")
    nn, dd = get_k_nearest_edges(g, [0.12], [0.12], k=2, min_dist=8000)
    print((nn, dd))
    assert(nn == [[(1, 3, 0), (1, 2, 0)]])
    np.testing.assert_almost_equal(dd, [[827.5247451916221, 2223.9016744838271]])
    
    print("get_k_nearest_edges(g, [0.12], [0.12], k=3, min_dist=8000)")
    nn, dd = get_k_nearest_edges(g, [0.12], [0.12], k=3, min_dist=8000)
    print((nn, dd))
    assert(nn == [[(1, 3, 0), (1, 2, 0), (2, 3, 0)]])
    np.testing.assert_almost_equal(dd, [[827.5247451916221, 2223.9016744838271, 4689.289257479777]])
    
    print("get_k_nearest_edges(g, [0.12], [0.12], k=3, min_dist=4000)")
    nn, dd = get_k_nearest_edges(g, [0.12], [0.12], k=3, min_dist=4000)
    print((nn, dd))
    assert(nn == [[(1, 3, 0), (1, 2, 0)]])
    np.testing.assert_almost_equal(dd, [[827.5247451916221, 2223.9016744838271]])
    
    print("get_k_nearest_edges(g, [0.12], [0.12], k=3, min_dist=2000)")
    nn, dd = get_k_nearest_edges(g, [0.12], [0.12], k=3, min_dist=2000)
    print((nn, dd))
    assert(nn == [[(1, 3, 0)]])
    np.testing.assert_almost_equal(dd, [[827.5247451916221]])
    
    print("get_k_nearest_edges(g, [0.12, 0.18], [0.12, 0.12], k=3, min_dist=2000)")
    nn, dd = get_k_nearest_edges(g, [0.12, 0.18], [0.12, 0.12], k=3, min_dist=2000)
    print((nn, dd))
    assert(nn == [[(1, 3, 0)], [(2, 3, 0)]])
    np.testing.assert_almost_equal(dd, [[827.5247451916221], [1103.3630281000585]])


test_get_k_nearest_edges()


# +
def read_counter_data(counter_file):
    counter_df = pd.read_parquet(counter_file)
    if 'day' in counter_df.columns:
        counter_df['month'] = counter_df['day'].str[:7]
    else:
        counter_df['month'] = counter_df['time_bin'].str[:7]
    if 'heading' not in counter_df.columns:
        counter_df['heading'] = -1
    if 'name' not in counter_df.columns:
        counter_df['name'] = ''
    counter_df = counter_df[['id', 'lat', 'lon', 'month', 'heading', 'name']]
    return counter_df


def get_counter_locations(city):
    loop_counter_files = sorted(list((COUNTER_PATH / city).glob('**/counters_*.parquet')))
    cdfs = []
    for lcf in loop_counter_files:
        cdfs.append(read_counter_data(lcf))
    counter_df = pd.concat(cdfs)
    counter_locations_df = counter_df[['id', 'lat', 'lon', 'heading', 'name']].groupby(by=['id']).last().reset_index()
    return counter_locations_df


def get_bearing(lat1, lon1, lat2, lon2):
    return Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)["azi1"] % 360


def bearing_diff(b1, b2):
    res= (b1-b2) % 360
    if res < 180:
        return res
    else:
        return (360-res)


def get_edge_geometry(g, edge, edge_data=None):
    if not edge_data:
        edge_data = g.edges[edge]
    if 'geometry' not in edge_data:
        n1, n2, ekey = edge
        mp1 = g.nodes[n1]
        mp2 = g.nodes[n2]
        line = LineString([Point(mp1['x'],mp1['y']), Point(mp2['x'],mp2['y'])])
        edge_data['geometry'] = line
    else:
        line = edge_data['geometry']
    return line


def find_nearest_ways(g, df):
    xs = df['lon'].tolist()
    ys = df['lat'].tolist()
    edges, dists = get_k_nearest_edges(g, xs, ys)
    headings = df['heading']
    names = df['name']
    
    way = []
    way_dist = []
    us = []
    vs = []
    for edge_candidates, dist_candidates, heading, counter_name in zip(edges, dists, headings, names):
        if not edge_candidates:
            way.append(-1)
            way_dist.append(-1)
            us.append(-1)
            vs.append(-1)
            continue
        best_edge = edge_candidates[0]
        best_dist = 1e7
        best_diff = best_dist + 360
        for edge, dist in zip(edge_candidates, dist_candidates):
            # Check if there's a better edge than the closest
            ed = g.edges[edge]
            if counter_name and 'ref' in ed:
                # If there's a name compare them
                ref = ed['ref']
                if ref.lower() in counter_name.lower():
                    match_diff = 0
            if heading >= 0:
                # if there's a heading use it
                line = get_edge_geometry(g, edge)
                p1 = line.coords[0]
                p2 = line.coords[1]
                bearing = get_bearing(p1[1], p1[0], p2[1], p2[0])
                angle_diff = abs(bearing_diff(bearing, heading))
                match_diff = angle_diff + dist
            if match_diff < best_diff:
                best_edge = edge
                best_dist = dist
                best_diff = match_diff
        ed = g.edges[best_edge]
        way.append(ed['osmid'])
        way_dist.append(best_dist)
        u, v, key = best_edge
        us.append(u)
        vs.append(v)
    
    df['ways'] = edges
    df['way_dists'] = dists
    df['way'] = way
    df['way_dist'] = way_dist
    df['u'] = us
    df['v'] = vs
    return df


def save_matched_locations(df, city_path):
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.lon, df.lat))
    gdf['ways'] = gdf['ways'].astype(str)
    gdf['way_dists'] = gdf['way_dists'].astype(str)
    if gdf['id'].dtype == np.dtype('O'):
        gdf['id'] = gdf['id'].astype(str)
    gdf.to_parquet(city_path / 'counters_matched.parquet', compression="snappy")
    gdf.to_file(city_path / 'counters_matched.geojson')
    return gdf


def simplify_graph(g):
    if "simplified" not in g.graph or not g.graph["simplified"]:
        g = ox.simplify_graph(g)
        print(f"Simplified road graph {len(g.nodes)} nodes and {len(g.edges)} edges")
    return g


# -

# # Madrid Locations

MADRID_GRAPH_PATH = RELEASE_PATH / '2022' / 'road_graph' / 'madrid'
madrid_g = ox.load_graphml(MADRID_GRAPH_PATH / 'road_graph.graphml')
print(f'Road graph has {len(madrid_g.nodes)} nodes and {len(madrid_g.edges)} edges')

madrid_g = simplify_graph(madrid_g)

madrid_locations_df = get_counter_locations('madrid')
# id	lat	lon	heading	time_bin	type	volume	occupation	congestion_level	speed_avg
madrid_locations_df

# !!! This takes 10-15 minutes for the whole graph !!!
madrid_matched_df = find_nearest_ways(madrid_g, madrid_locations_df)
madrid_matched_df[madrid_matched_df['way'] != -1]

madrid_matched_df['way'] = madrid_matched_df['way'].astype(str)

save_matched_locations(madrid_matched_df, MADRID_GRAPH_PATH)

# # London Locations

LONDON_GRAPH_PATH = RELEASE_PATH / '2022' / 'road_graph' / 'london'
london_g = ox.load_graphml(LONDON_GRAPH_PATH / 'road_graph.graphml')
print(f'Road graph has {len(london_g.nodes)} nodes and {len(london_g.edges)} edges')

london_g = simplify_graph(london_g)

london_locations_df = get_counter_locations('london')
# day	time_bin	id	flow_15m	sat_bandings	det_no	num_det	detector_rate	ts	lat	lon
# id	name	lat	lon	heading	time_bin	volume	speed
london_locations_df

# !!! This takes 10-15 minutes for the whole graph !!!
london_matched_df = find_nearest_ways(london_g, london_locations_df)
london_matched_df[london_matched_df['way'] != -1]

london_matched_df['way'] = london_matched_df['way'].astype(str)

save_matched_locations(london_matched_df, LONDON_GRAPH_PATH)

# # Berlin Locations

BERLIN_GRAPH_PATH = RELEASE_PATH / '2021' / 'road_graph' / 'berlin'
berlin_g = ox.load_graphml(BERLIN_GRAPH_PATH / 'road_graph.graphml')
print(f'Road graph has {len(berlin_g.nodes)} nodes and {len(berlin_g.edges)} edges')

berlin_g = simplify_graph(berlin_g)

berlin_locations_df = get_counter_locations('berlin')
berlin_locations_df

# !!! This takes 10-15 minutes for the whole graph !!!
berlin_matched_df = find_nearest_ways(berlin_g, berlin_locations_df)
berlin_matched_df[berlin_matched_df['way'] != -1]

berlin_matched_df['way'] = berlin_matched_df['way'].astype(str)

save_matched_locations(berlin_matched_df, BERLIN_GRAPH_PATH)
