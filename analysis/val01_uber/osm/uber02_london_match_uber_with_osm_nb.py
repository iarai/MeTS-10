# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
# +
import ast
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osmnx.distance import great_circle_vec
from shapely.geometry import LineString

# +
pd.set_option("display.max_columns", None)
UBASEPATH = Path("/iarai/public/t4c/uber")
UFILE = "movement-speeds-hourly-london-2019-10.parquet"
# TBASEPATH = Path('/iarai/public/t4c/data_pipeline/release20221003_keep_all_edges_residential_unclassified/2022')

# TBASEPATH = Path('/iarai/public/t4c/data_pipeline/release20221022_historic/2022')

OBASEPATH = Path("/iarai/public/t4c/osm")
place = "england-200101-truncated"
# place = "england-latest-truncated"
ONODESFILE = "%s_nodes.parquet" % place
OWAYSFILE = "%s_ways.parquet" % place

# write_ways_gpkg = True
# write_nodes_gpkg = True

# print(TBASEPATH)

CITY = "london"

uspeeds_df = pd.read_parquet(UBASEPATH / UFILE)
uspeeds_df["osm_way_id"] = uspeeds_df["osm_way_id"].astype(str)
print(uspeeds_df.columns)
# # print(uspeeds_df)
# tedges_df = pd.read_parquet(TBASEPATH / "road_graph" / CITY / "road_graph_freeflow.parquet")
# tnodes_df = pd.read_parquet(TBASEPATH / "road_graph" / CITY / "road_graph_nodes.parquet")
# tedges_df = tedges_df.merge(tnodes_df, how='left', left_on="u", right_on="node_id", suffixes=("", "_u")).merge(
#     tnodes_df, how='left', left_on="v", right_on="node_id", suffixes=("_u", "_v"))

# # explode osmid lists
# tedges_df["osmid_bak"] = tedges_df["osmid"]
# tedges_df["osmid"] = [ast.literal_eval(osmid) for osmid in tedges_df["osmid"]]
# tedges_df = tedges_df.explode("osmid")
# tedges_df["osmid"] = tedges_df["osmid"].astype(str)

# # collapse any highway lists (can happen during graph simplification)
# # into string values simply by keeping just the first element of the list
# tedges_df["highway"] = [ast.literal_eval(highway) if highway.startswith("[") else highway for highway in
#                         tedges_df["highway"]]
# tedges_df["highway"] = tedges_df["highway"].map(lambda x: x[0] if isinstance(x, list) else x)

# print(tedges_df.columns)


uspeeds_df
# -

# df_nodes_osm = pd.read_parquet(OBASEPATH / ONODESFILE)
df_nodes_osm = gpd.read_parquet(OBASEPATH / ONODESFILE)
print(df_nodes_osm.columns)
# print(df_nodes_osm.dtypes)
# print(df_nodes_osm)

# df_ways_osm = pd.read_parquet(OBASEPATH / OWAYSFILE)
df_ways_osm = gpd.read_parquet(OBASEPATH / OWAYSFILE)
df_ways_osm["osmid"] = df_ways_osm["osmid"].astype(str)
df_ways_osm["refs"] = [ast.literal_eval(refs) if refs is not None and refs.startswith("[") else refs for refs in df_ways_osm["refs"]]
print(df_ways_osm.columns)

assert type(df_ways_osm) == gpd.geodataframe.GeoDataFrame

assert uspeeds_df.groupby(["osm_start_node_id", "osm_end_node_id", "osm_way_id", "year", "month", "day", "hour"]).count()["speed_mph_mean"].max() == 1

# +
uedges_df = uspeeds_df.groupby(["osm_start_node_id", "osm_end_node_id", "osm_way_id"]).first().reset_index()
uedges_df = uedges_df.drop(["year", "month", "day", "hour", "speed_mph_mean", "speed_mph_stddev"], axis=1)
# print(uedges_df.columns)

uedges_df = (
    df_ways_osm.merge(uedges_df, how="right", right_on="osm_way_id", left_on="osmid", suffixes=("_way", ""))
    .merge(df_nodes_osm.add_suffix("_start_node"), how="left", left_on="osm_start_node_id", right_on="osmid_start_node")
    .merge(df_nodes_osm.add_suffix("_end_node"), how="left", left_on="osm_end_node_id", right_on="osmid_end_node")
)
uedges_df
# -

uedges_df["length_meters"] = [
    great_circle_vec(x_u, y_u, x_v, y_v)
    for x_u, y_u, x_v, y_v in zip(uedges_df["x_start_node"], uedges_df["y_start_node"], uedges_df["x_end_node"], uedges_df["y_end_node"])
]

uedges_df["geometry_old"] = uedges_df["geometry"]

uedges_df.columns

# check it's a geopandas frame
assert type(uedges_df) == gpd.geodataframe.GeoDataFrame

# +
# print(df_ways_osm)
# print(uspeeds_df.columns)
# print(uspeeds_df.dtypes)
# uspeeds_df = uspeeds_df.merge(df_nodes_osm, how='left', left_on="osm_start_node_id", right_on="osmid",
#                              suffixes=("", "_start_node")) \
#    .merge(df_nodes_osm, how='left', left_on="osm_end_node_id", right_on="osmid",
#           suffixes=("_start_node", "_end_node")) \
#     .merge(df_ways_osm, how='left', left_on="osm_way_id", right_on="osmid", suffixes=("", "_way"))
# print(uspeeds_df.columns)
# print(uedges_df.columns)
# -

uedges_df.groupby("highway").agg(count=("osm_start_node_id", "count"), length_meters_mean=("length_meters", "mean"))

# ### Sanity check node degrees Uber

neighbors = defaultdict(lambda: set())
for u, v in zip(uedges_df["osm_start_node_id"], uedges_df["osm_end_node_id"]):
    neighbors[u].add(v)
    neighbors[v].add(u)

node_degrees = {u: len(n) for u, n in neighbors.items()}
node_degrees

counts, bins = np.histogram(list(node_degrees.values()), bins=range(10))
for lb, c in zip(bins, counts):
    print(f"{lb}: {c:10d} ({c/len(node_degrees)*100:5.2f}%)")

plt.hist(list(node_degrees.values()), bins=range(10))

# Most have degree 2, some have 3 and very few have 1 and 4. Looks plausible.

# ### How many Uber nodes and ways can we match with historic OSM?

unodes = set(uedges_df["osm_start_node_id"]).union(uedges_df["osm_end_node_id"])
uways = set(uedges_df["osm_way_id"])
onodes = set(df_nodes_osm["osmid"])
oways = set(df_ways_osm["osmid"])

uo_nodes = unodes.intersection(onodes)

len(uo_nodes) / len(unodes), len(unodes) - len(uo_nodes), len(uo_nodes), len(unodes)

uo_ways = uways.intersection(oways)

len(uo_ways) / len(uways), len(uways) - len(uo_ways), len(uo_ways), len(uways)

# ### extract geometry for Uber road (create road_graph_nodes.parquet and road_graph_edges.parquet) for Uber road graph

# +
ambiguous = []
projected_geometries = []
status = []
for x_start_node, y_start_node, x_end_node, y_end_node, geometry, osm_way_id, hw, refs, osm_start_node_id, osm_end_node_id in zip(
    uedges_df["x_start_node"],
    uedges_df["y_start_node"],
    uedges_df["x_end_node"],
    uedges_df["y_end_node"],
    uedges_df["geometry_old"],
    uedges_df["osm_way_id"],
    uedges_df["highway"],
    uedges_df["refs"],
    uedges_df["osm_start_node_id"],
    uedges_df["osm_end_node_id"],
):
    #     print(x_start_node,y_start_node,x_end_node,y_end_node)
    #     print(osm_way_id)
    #     print(hw)

    #
    if geometry is None:

        if hw is None:
            assert osm_way_id in uways
            assert osm_way_id not in oways
            status.append("no highway")
        else:
            status.append("no geometry")
        projected_geometries.append(None)
        continue
    coords = list(geometry.coords)
    #     print(list(geometry.coords))
    #     assert len(set(geometry.coords))==len(list(geometry.coords)), (osm_way_id,len(set(geometry.coords)),len(list(geometry.coords)))

    #     start_node_index = [i for i, coord in enumerate(coords) if coord==(x_start_node,y_start_node)]
    start_node_index = [i for i, ref in enumerate(refs) if ref == osm_start_node_id]
    if len(start_node_index) != 1:
        #         print(x_start_node,y_start_node,x_end_node,y_end_node)
        #         print(osm_start_node_id)
        #         print(osm_end_node_id)
        #         print(osm_way_id)
        #         print(refs)
        #         raise
        ambiguous.append(osm_way_id)
        projected_geometries.append(None)
        status.append("start node ambiguous")
        #         print(x_start_node,y_start_node,x_end_node,y_end_node,geometry,osm_way_id,hw)
        continue
    start_node_index = start_node_index[0]
    #     print(start_node_index)

    #     end_node_index = [i for i, coord in enumerate(coords) if coord==(x_end_node,y_end_node)]
    end_node_index = [i for i, ref in enumerate(refs) if ref == osm_end_node_id]

    if len(end_node_index) != 1:
        ambiguous.append(osm_way_id)
        projected_geometries.append(None)
        status.append("end node ambiguous")
        #         print(x_start_node,y_start_node,x_end_node,y_end_node,geometry,osm_way_id,hw)
        continue
    end_node_index = end_node_index[0]
    #     print(end_node_index)
    assert (start_node_index < end_node_index) or (start_node_index > end_node_index), (start_node_index, end_node_index)
    #     geometry = LineString(coords[start_node_index:end_node_index+1])
    #     print(coords)
    #     print(coords[4:2])
    # #     break
    #     break

    if start_node_index < end_node_index:
        projected_geometries.append(LineString(coords[start_node_index : end_node_index + 1]))
    elif start_node_index > end_node_index:
        projected_geometries.append(LineString(reversed(coords[end_node_index : start_node_index + 1])))
    else:
        raise
    status.append("ok")
    if len(projected_geometries[-1].coords) == 0:
        print(start_node_index, end_node_index)
        print((x_start_node, y_start_node, x_end_node, y_end_node, geometry, osm_way_id, hw, refs, osm_start_node_id, osm_end_node_id))
        raise
uedges_df["geometry"] = projected_geometries
uedges_df["status"] = status
# -

# TODO visual inspection!
# TODO can improve on the ambiguous?
matching_stats_df = uedges_df.groupby(["highway", "status"]).agg(count=("osmid", "count"), avg_length_meters=("length_meters", "mean"))
matching_stats_df["percentage"] = matching_stats_df["count"] / len(uedges_df) * 100
matching_stats_df = matching_stats_df.reset_index()
matching_stats_df

len(~uedges_df["geometry"].isnull())

uedges_df[uedges_df["geometry"].isnull()].groupby("highway").agg(
    count=("osmid", "count"), avg_length_meters=("length_meters", "mean"), med_length_meters=("length_meters", "median"), status=("status", set)
)

# +
uedges_df[(uedges_df["geometry"].isnull()) & (uedges_df["highway"] == "motorway")]

# refs [311070, 3744469281]
# https://www.openstreetmap.org/way/345803322#map=17/51.25284/-0.12408
# https://www.openstreetmap.org/node/202995#map=16/51.2518/-0.1270       -> probably just deleted between recording and OSM version
# https://www.openstreetmap.org/node/3744469281#map=19/51.25548/-0.12439 -> in OSM
# here we could fix if the missing one has just one outgoing edge.

# -

uedges_debug_gdf = gpd.GeoDataFrame.from_records(
    [
        {"name": name, "geometry": LineString(geometry.coords)}
        for name, geometry, status in zip(uedges_df["name"], uedges_df["geometry_old"], uedges_df["status"])
        if geometry is not None and "ambiguous" in status
    ]
)
uedges_debug_gdf

uedges_debug_gdf.to_parquet(f"{CITY}_road_graph_edges_debug.parquet")

uedges_debug_gdf.to_file(f"{CITY}_road_graph_edges_debug.gpkg", driver="GPKG", layer="edges")

# ## Add speed_kph

# +
edges = uedges_df

# Logic taken from
# https://github.com/gboeing/osmnx/blob/ad54852a3131800b1eedeb167eccaa0276abd998/osmnx/speed.py

from osmnx.speed import _collapse_multiple_maxspeed_values, _clean_maxspeed


hwy_speeds = None
fallback = np.nan
precision = 1
agg = np.mean

# collapse any highway lists (can happen during graph simplification)
# into string values simply by keeping just the first element of the list
edges["highway"] = edges["highway"].map(lambda x: x[0] if isinstance(x, list) else x)

if "maxspeed" in edges.columns:
    # collapse any maxspeed lists (can happen during graph simplification)
    # into a single value
    edges["maxspeed"] = edges["maxspeed"].apply(_collapse_multiple_maxspeed_values, agg=agg)

    # create speed_kph by cleaning maxspeed strings and converting mph to
    # kph if necessary
    edges["speed_kph"] = edges["maxspeed"].astype(str).map(_clean_maxspeed).astype(float)
else:
    # if no edges in graph had a maxspeed attribute
    edges["speed_kph"] = None

# if user provided hwy_speeds, use them as default values, otherwise
# initialize an empty series to populate with values
if hwy_speeds is None:
    hwy_speed_avg = pd.Series(dtype=float)
else:
    hwy_speed_avg = pd.Series(hwy_speeds).dropna()

# for each highway type that caller did not provide in hwy_speeds, impute
# speed of type by taking the mean of the preexisting speed values of that
# highway type
for hwy, group in edges.groupby("highway"):
    if hwy not in hwy_speed_avg:
        hwy_speed_avg.loc[hwy] = agg(group["speed_kph"])

# if any highway types had no preexisting speed values, impute their speed
# with fallback value provided by caller. if fallback=np.nan, impute speed
# as the mean speed of all highway types that did have preexisting values
hwy_speed_avg = hwy_speed_avg.fillna(fallback).fillna(agg(hwy_speed_avg))

# for each edge missing speed data, assign it the imputed value for its
# highway type
speed_kph = edges[["highway", "speed_kph"]].set_index("highway").iloc[:, 0].fillna(hwy_speed_avg)

# all speeds will be null if edges had no preexisting maxspeed data and
# caller did not pass in hwy_speeds or fallback arguments
if pd.isnull(speed_kph).all():
    raise ValueError(("this graph's edges have no preexisting `maxspeed` " "attribute values so you must pass `hwy_speeds` or " "`fallback` arguments."))

# add speed kph attribute to graph edges
edges["speed_kph"] = speed_kph.round(precision).values
# -

# ## Output road_graph_nodes.parquet and road_graph_edges.parquet in compliant format

# +
# edges_attributes_list = ["u", "v", "osmid", "speed_kph", "maxspeed", "highway", "oneway", "lanes", "tunnel", "length_meters", "geometry"]
# [["node_id", "x", "y"]]
# -

for u, v, osmid, highway, length_meters, geometry, status in zip(
    uedges_df["osm_start_node_id"],
    uedges_df["osm_end_node_id"],
    uedges_df["osm_way_id"],
    uedges_df["highway"],
    uedges_df["length_meters"],
    uedges_df["geometry"],
    uedges_df["status"],
):
    if status == "ok" and len(geometry.coords) == 0:
        print((u, v, osmid, highway, length_meters, geometry, status))
        break

uedges_gdf = gpd.GeoDataFrame.from_records(
    [
        {
            "gkey": f"{u}_{v}_{osmid}",  # TODO hacky
            "u": u,
            "v": v,
            "osmid": osmid,
            "speed_kph": speed_kph,
            "maxspeed": maxspeed,
            "highway": highway,
            "oneway": "",  # TODO
            "lanes": "",  # TODO
            "tunnel": "",  # TODO
            "length_meters": length_meters,
            "geometry": LineString(geometry.coords),
            "name": name,  # TODO in addition to data_pipeline - add there as well?
        }
        for u, v, osmid, speed_kph, maxspeed, highway, length_meters, geometry, status, name in zip(
            uedges_df["osm_start_node_id"],
            uedges_df["osm_end_node_id"],
            uedges_df["osm_way_id"],
            uedges_df["speed_kph"],
            uedges_df["maxspeed"],
            uedges_df["highway"],
            uedges_df["length_meters"],
            uedges_df["geometry"],
            uedges_df["status"],
            uedges_df["name"],
        )
        if status == "ok"
    ]
)
uedges_gdf

uedges_gdf[uedges_gdf["osmid"] == "129375498"]

uedges_gdf.dtypes

assert type(uedges_gdf) == gpd.geodataframe.GeoDataFrame

# TODO naming as "london/road_graph_edges.parquet"
uedges_gdf.to_parquet(f"{CITY}_road_graph_edges.parquet")

uedges_gdf.to_file(f"{CITY}_road_graph_edges.gpkg", driver="GPKG", layer="edges")

nodes_in_uedges_gdf = set(uedges_gdf["u"]).union(uedges_gdf["v"])
nodes_in_uedges_gdf

unodes_gdf = gpd.GeoDataFrame.from_records(
    [
        {"node_id": node_id, "x": x, "y": y, "geometry": geometry}
        for node_id, x, y, geometry in zip(df_nodes_osm["osmid"], df_nodes_osm["x"], df_nodes_osm["y"], df_nodes_osm["geometry"])
        if node_id in nodes_in_uedges_gdf
    ]
)
unodes_gdf

unodes_gdf.to_parquet(f"{CITY}_road_graph_nodes.parquet")

unodes_gdf.to_file(f"{CITY}_road_graph_nodes.gpkg", driver="GPKG", layer="nodes")

unodes_gdf[unodes_gdf["node_id"] == 2348324108]

uedges_gdf[uedges_gdf["u"] == 2348324108]

assert len(uedges_gdf) == matching_stats_df[matching_stats_df["status"] == "ok"]["count"].sum()
