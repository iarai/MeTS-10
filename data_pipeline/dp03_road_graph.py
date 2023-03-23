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
"""Generate road graph using OSM data for the given city.

This script generates a basic road graph using source data from OSM.

The output files `road_graph_nodes.parquet`, `road_graph_edges.parquet`, `road_graph.gpkg` and `road_graph.graphml` will be written to `road_graph/<city>`.

Usage:
  dp03_road_graph.py [-h] -d DATA_FOLDER -c CITY [-r]

Arguments:
  -d DATA_FOLDER, --data_folder DATA_FOLDER
                        Folder containing a subfolders road_graph/<city>
  -c city_name, --city city_name
                        Name of the city to be processed.
  -cf filter_string, --custom_filter filter_string
                        Optional custom OSM filter string.
  -pms, --parse_maxspeed
                        Use the improved maxspeed parsing logic instead of the default OSMNX one.
  -kae, --keep_all_edges
                        Disable simplification to keep all OSM segments as individual edges in the graph.
  -hf, --heatmap_filter
                        Use the heatmap to filter roads with too low volume.
  -cn, --counter_nodes GEOJSON_FILE
                        Use the points in this .geojson files as counters to be merged with the road graph.
  -of, --osm_file OSM_FILE
                        Use this .osm file instead of download. City bounding box will be applied on this.
  -cbb, --custom_bounding_box NORTH,SOUTH,ESAST,WEST
                        Custom bounding box instead of competition city bounding box.
  -f, --force_overwrite
                        Force overwriting existing files.
"""
import argparse
import ast
import hashlib
import logging
import numbers
import os
import sys
import warnings
from ast import literal_eval
from pathlib import Path
from typing import Optional
from typing import Tuple

import osmnx as ox
import osmnx.graph
import osmnx.truncate
from data_helpers import get_latlon_bounds
from osmnx.io import utils_graph
from road_graph_helpers import process_counter_merge
from road_graph_helpers import process_heatmap_filter
from shapely import wkb

# The following default max speeds are used when the parse_maxspeed option is used
# and the segment doesn't have a usable maxspeed attribute.
DEFAULT_MAXSPEEDS = {
    "motorway": 120,
    "motorway_link": 120,
    "primary": 80,
    "primary_link": 80,
    "residential": 40,
    "secondary": 60,
    "secondary_link": 60,
    "tertiary": 50,
    "tertiary_link": 50,
    "trunk": 100,
    "trunk_link": 100,
    "unclassified": 50,
}


def get_maxspeed(maxspeed, highway):
    # OSMNX does also offer a default speed logic in add_edge_speeds() but this still uses an average if a list
    # of speeds is parsed. In this logic here we try to get the true maximum speed instead.
    try:
        highway = literal_eval(highway)
        highway = highway[0]
    except ValueError:
        highway = str(highway)
    if not maxspeed:
        return DEFAULT_MAXSPEEDS[highway]
    maxspeed = maxspeed.replace(",", "|")
    maxspeed = "".join([i for i in maxspeed if i.isdigit() or i == "|"])
    maxspeeds = maxspeed.split("|")
    maxspeeds = [float(m) if m.isdigit() else -1 for m in maxspeeds]
    maxspeed = max(maxspeeds)
    if maxspeed < 0:
        return DEFAULT_MAXSPEEDS[highway]
    return maxspeed


# Use `hashlib` instead of plain `hash`  since the latter is not deterministic: https://stackoverflow.com/questions/45015180/deterministic-recursive-hashing-in-python
def gkey_hash(b: bytes):
    # 8 bytes = 64bit to
    h = hashlib.blake2b(b, digest_size=8).digest()
    return int.from_bytes(h, "big")


def process_road_graph(  # noqa: C901
    data_folder: Path,
    city: str,
    custom_filter: str,
    overwrite: bool = False,
    parse_maxspeed: bool = False,
    simplify: bool = True,
    heatmap_filter: bool = True,
    counter_nodes: Optional[Path] = None,
    counter_split: bool = False,
    osm_file: Optional[Path] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
):
    city_road_graph_folder = data_folder / "road_graph" / city
    if not overwrite and ((city_road_graph_folder / "road_graph_nodes.parquet").exists() or (city_road_graph_folder / "road_graph_edges.parquet").exists()):
        print(f"Road graph files for {city} exist already. Skipping ...")
        return
    graph_file = city_road_graph_folder / "road_graph.graphml"
    if bbox is None:
        bbox, rotate = get_latlon_bounds(city)
    south, north, west, east = bbox
    if graph_file.exists() and not overwrite:
        print(f"Loading road graph for {city} from {graph_file}")
        g = ox.load_graphml(graph_file)
    else:
        if osm_file is not None:
            print(f"Loading road graph for {city} from {osm_file}")
            g = ox.graph.graph_from_xml(osm_file, simplify=False, retain_all=True, bidirectional=True)
            print(f"Truncating road graph {g} for {city} to {bbox}")
            g = ox.truncate.truncate_graph_bbox(g, north, south, east, west)
        else:
            graph_file.parent.mkdir(exist_ok=True, parents=True)
            print(f"Downloading road graph for {city} with bbox {bbox}")
            # download street network data from OSM and construct a MultiDiGraph model
            g = ox.graph_from_bbox(north, south, east, west, network_type="drive", simplify=False, truncate_by_edge=True, custom_filter=custom_filter)
        # impute edge (driving) speeds and calculate edge traversal times
        g = ox.add_edge_speeds(g)  # We do not use hwy_speeds here but instead the custom get_maxspeed logic if requested.
        g = ox.add_edge_travel_times(g)
        ox.save_graphml(g, filepath=graph_file)
    print(f"Road graph for {city} is {g} in {graph_file}")

    if counter_nodes is not None:
        g = process_counter_merge(g, counter_nodes, assign_all_neighbors=counter_split)

    if heatmap_filter:
        g = process_heatmap_filter(g, data_folder / "movie_heatmap" / city / "probe_heatmap.h5", rotate=rotate, lon_min=west, lat_min=south)
        print(f"Filtered road graph for {city} is {g}")

    if simplify and ("simplified" not in g.graph or not g.graph["simplified"]):
        g = ox.simplify_graph(g)
        print(f"Simplified road graph for {city} is {g}")
    ox.save_graph_geopackage(g, filepath=(city_road_graph_folder / "road_graph.gpkg"), directed=True)

    nodes_df, edges_df = utils_graph.graph_to_gdfs(g)
    nodes_df = nodes_df.reset_index()
    nodes_df = nodes_df.rename(columns={"osmid": "node_id"})
    nodes_df = nodes_df[["node_id", "x", "y"]]
    edges_df = edges_df.reset_index()
    edges_df["highway"] = edges_df["highway"].astype(str)
    edges_df["maxspeed"] = edges_df["maxspeed"].astype(str)
    edges_df["lanes"] = edges_df["lanes"].astype(str)
    edges_df["tunnel"] = edges_df["tunnel"].astype(str)
    edges_df["osmid"] = edges_df["osmid"].astype(str)
    edges_df["oneway"] = edges_df["oneway"].astype(str)
    edges_df = edges_df.rename(columns={"length": "length_meters"})
    if parse_maxspeed:
        edges_df["speed_kph"] = [get_maxspeed(ms, hw) for ms, hw in zip(edges_df["maxspeed"], edges_df["highway"])]
    else:
        # TODO why does this happen? `add_edge_speeds` is supposed to return one value?
        edges_df["speed_kph"] = [s[0] if type(s) == list else s for s in edges_df["speed_kph"]]
    edges_attributes_list = ["u", "v", "osmid", "speed_kph", "maxspeed", "highway", "oneway", "lanes", "tunnel", "length_meters", "geometry"]
    edges_df = edges_df[edges_attributes_list]

    # gkey generation
    # As there are some rare notorious situations, where there are two different ways between the same two nodes
    # (u,v,osmid) is not unique, so let's generate hash based on geometry to differentiate.
    edges_df["gkey"] = [gkey_hash(wkb.dumps(g)) for g in edges_df["geometry"]]
    # Fiona has problems with uint64: https://github.com/Toblerity/Fiona/issues/365
    edges_df["gkey"] = edges_df["gkey"].astype("int64")

    # remove duplicate gkeys
    # As OSM is a topological data model, the multiple ways can be projected to the same geometry
    # Here is an example:
    #     https://www.openstreetmap.org/node/3747813554
    #     https://www.openstreetmap.org/node/2433128136
    #     https://www.openstreetmap.org/way/371170715
    #     https://www.openstreetmap.org/node/3747813554
    #     https://www.openstreetmap.org/node/2433128136
    #     https://www.openstreetmap.org/way/371882252
    df_by_key = (
        edges_df.groupby(["gkey"])
        .agg(
            count=("geometry", "count"),
            **{attr: (attr, "first") for attr in ["speed_kph", "maxspeed", "highway", "length_meters"]},
        )
        .reset_index()
    )
    duplicates = df_by_key[df_by_key["count"] > 1]

    print(f"Found {len(duplicates)} duplicate gkeys.")
    print(f"  highway {duplicates['highway'].max()}")
    print(f"  max length_meters {duplicates['length_meters'].max()}")
    edges_df["gkey_duplicate"] = False
    for rec in duplicates.head().to_dict("records"):
        edges_df.loc[edges_df["gkey"] == rec["gkey"], "gkey_duplicate"] = True
        for rec_ in edges_df[edges_df["gkey"] == rec["gkey"]].to_dict("records"):
            print(rec_)
            print(f"https://www.openstreetmap.org/node/{rec_['u']}")
            print(f"https://www.openstreetmap.org/node/{rec_['v']}")
            print(f"https://www.openstreetmap.org/way/{rec_['osmid']}")
    print(f"-> before taking the first of duplicates only: {len(edges_df)}")
    edges_df.drop_duplicates(["gkey"], inplace=True)
    print(f"-> after taking the first of duplicates only: {len(edges_df)}")

    # key constraint check after gkey duplicates removal
    df_by_key = (
        edges_df.groupby(["gkey"])
        .agg(
            count=("geometry", "count"),
            **{attr: (attr, "first") for attr in ["speed_kph", "maxspeed", "highway", "length_meters"]},
        )
        .reset_index()
    )
    duplicates = df_by_key[df_by_key["count"] > 1]
    assert df_by_key["count"].max() == 1, f"gkey not unique: {duplicates}"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*initial implementation of Parquet.*")
        nodes_fn = city_road_graph_folder / "road_graph_nodes.parquet"
        print(f"Saving {len(nodes_df)} nodes to {nodes_fn}")
        nodes_df.to_parquet(nodes_fn, compression="snappy")
        edges_fn = city_road_graph_folder / "road_graph_edges.parquet"
        print(f"Saving {len(edges_df)} edges to {edges_fn}")
        edges_df.to_parquet(edges_fn, compression="snappy")
    print("... finished creating road graph.")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="This script creates a road graph from OSM.")
    parser.add_argument(
        "-d",
        "--data_folder",
        type=str,
        help="Folder containing T4c data folder structure",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--city",
        type=str,
        help="City to be processed",
        required=True,
        default=None,
    )
    parser.add_argument(
        "-cf",
        "--custom_filter",
        type=str,
        help="Custom OSM filter",
        required=False,
        default='["highway"~"motorway|motorway_link|trunk|primary|secondary|tertiary|residential|unclassified"]',
    )
    parser.add_argument("-pms", "--parse_maxspeed", help="Parse the maxspeed string field with the improved logic", required=False, action="store_true")
    parser.add_argument(
        "-kae", "--keep_all_edges", help="Disable simplification to keep all OSM segments as individual edges in the graph", required=False, action="store_true"
    )
    parser.add_argument("-f", "--force_overwrite", help="Force overwriting existing files", required=False, action="store_true")
    parser.add_argument("-hf", "--heatmap_filter", help="Use the heatmap to filter roads with too low volume.", required=False, action="store_true")
    parser.add_argument(
        "-cn",
        "--counter_nodes",
        type=Path,
        help="Use the points in this .geojson files as counters to be merged with the road graph.",
        required=False,
    )
    parser.add_argument("-cns", "--counter_split", help="Assign counters to all neighbouring nodes in a junction.", required=False, action="store_true")
    parser.add_argument(
        "-of",
        "--osm_file",
        type=Path,
        help="Use this .osm file instead of download.",
        required=False,
    )
    parser.add_argument(
        "-cbb",
        "--custom_bounding_box",
        type=str,
        help="Custom bounding box",
        required=False,
    )
    return parser


def main(argv):
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"), format="[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
    )
    parser = create_parser()
    try:
        params = parser.parse_args(argv)
        params = vars(params)
        data_folder = Path(params["data_folder"])
        city = params["city"]
        custom_filter = params["custom_filter"]
        parse_maxspeed = params["parse_maxspeed"]
        overwrite = params["force_overwrite"]
        heatmap_filter = params["heatmap_filter"]
        counter_nodes = params["counter_nodes"]
        counter_split = params["counter_split"]
        osm_file = params["osm_file"]
        simplify = not params["keep_all_edges"]
        custom_bounding_box = params["custom_bounding_box"]
        if custom_bounding_box is not None:
            custom_bounding_box = ast.literal_eval(custom_bounding_box)
            assert isinstance(custom_bounding_box, tuple), f"Bounding box must be given as tuple, found {type(custom_bounding_box)} in {custom_bounding_box}"
            assert len(custom_bounding_box) == 4, f"Bounding box must contain 4 values, found {len(custom_bounding_box)} in {custom_bounding_box}"
            assert all(
                [isinstance(b, numbers.Number) for b in custom_bounding_box]
            ), f"Bounding box must contain only number, found {[type(b) for b in custom_bounding_box]} in {custom_bounding_box}"
    except Exception as e:
        logging.exception(f"Could not parse args.", exc_info=e)
        parser.print_help()
        exit(1)
    process_road_graph(
        data_folder,
        city=city,
        custom_filter=custom_filter,
        overwrite=overwrite,
        heatmap_filter=heatmap_filter,
        parse_maxspeed=parse_maxspeed,
        simplify=simplify,
        counter_nodes=counter_nodes,
        counter_split=counter_split,
        osm_file=osm_file,
        bbox=custom_bounding_box,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
