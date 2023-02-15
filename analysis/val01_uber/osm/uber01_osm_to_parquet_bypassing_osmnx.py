import argparse
import ast
import datetime
import logging
import os
import sys
import timeit
from pathlib import Path

import geopandas as gpd
import humanize
import xmltodict
from shapely.geometry import LineString
from shapely.geometry import Point

# TODO write to geopandas!


# https://stackoverflow.com/questions/14452145/how-to-measure-time-taken-between-lines-of-code-in-python
class CodeTimer:
    def __init__(self, name=None):
        self.name = name if name else "Code block"

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = int((timeit.default_timer() - self.start) * 1000.0)
        delta = datetime.timedelta(milliseconds=self.took)
        logging.info(self.name + " took " + humanize.precisedelta(delta))


def osm_to_parquet(data_folder: Path, place: str, output_refs: bool = True):
    logging.info(f"/ start osm_to_parquet {data_folder} {place}")
    nodes = []
    file_path = data_folder / f"{place}.osm"
    nodes = []
    nodes_d = {}
    ways = []

    def handle_item(path, item, log_interval=100000):
        # TODO strange, would have expected callback to be called with item_depth=1...
        # print(f"path {path[:-1]}: found {path[-1]}")
        tag, content = path[-1]

        if tag == "node":
            nodes.append(content)
            content["lon"] = float(content["lon"])
            content["lat"] = float(content["lat"])
            content["geometry"] = Point(content["lon"], content["lat"])
            nodes_d[content["id"]] = content
            if len(nodes) % log_interval == 0:
                print(f"\r{len(nodes)} nodes, {len(ways)} ways", end="")
                # print(path)
                # print(item)
        elif tag == "way":

            # print(path)
            # print(item)
            if "tag" not in item:
                item["tag"] = []
            if isinstance(item["tag"], dict):
                item["tag"] = [item["tag"]]
            d = {r["@k"]: r["@v"] for r in item["tag"]}

            content["highway"] = d.get("highway", "")
            content["maxspeed"] = d.get("maxspeed", "")
            content["name"] = d.get("name", "")

            refs = [r["@ref"] for r in item["nd"]]
            coordinates = [(nodes_d[node_id]["lon"], nodes_d[node_id]["lat"]) for node_id in refs if node_id in nodes_d]
            if output_refs:
                content["refs"] = f"[{','.join(refs)}]"
                ast.literal_eval(content["refs"])

            # some ways at the border are filtered out
            full_geometry = len(coordinates) == len(refs)
            content["full_geometry"] = full_geometry
            valid_geometry = len(coordinates) >= 2
            content["valid_geometry"] = full_geometry
            if valid_geometry:
                content["geometry"] = LineString(coordinates)
            else:
                content["geometry"] = LineString([(-1, -1), (-1, -1)])

            ways.append(content)
            if len(ways) % log_interval == 0:
                print(f"\r{len(nodes)} nodes, {len(ways)} ways", end="")
                # print(path)
                # print(item)
        return True

    with CodeTimer(f"Parsing  {file_path}"):
        with file_path.open("rb") as f:
            xmltodict.parse(f, item_depth=2, encoding="utf-8", item_callback=handle_item)
            print("")

    gdf_nodes = gpd.GeoDataFrame.from_records(nodes)
    gdf_nodes = gdf_nodes.rename(columns={"id": "osmid", "lat": "y", "lon": "x"})
    gdf_nodes["osmid"] = gdf_nodes["osmid"].astype("int")
    gdf_nodes["x"] = gdf_nodes["x"].astype("float")
    gdf_nodes["y"] = gdf_nodes["y"].astype("float")
    print(gdf_nodes)
    nodes_suffix = "_nodes"
    nodes_gparquet_file = file_path.with_name(file_path.with_suffix("").name + nodes_suffix).with_suffix(".parquet")
    with CodeTimer(f"Writing {nodes_gparquet_file}"):
        gdf_nodes.to_parquet(nodes_gparquet_file)
    nodes_gpkg_file = file_path.with_name(file_path.with_suffix("").name + nodes_suffix).with_suffix(".gpkg")
    with CodeTimer(f"Writing {nodes_gpkg_file}"):
        gdf_nodes.to_file(nodes_gpkg_file, driver="GPKG", layer="nodes")
    del gdf_nodes
    gdf_ways = gpd.GeoDataFrame.from_records(ways)
    gdf_ways = gdf_ways.rename(columns={"id": "osmid"})
    gdf_ways["osmid"] = gdf_ways["osmid"].astype("int")
    print(gdf_ways)
    ways_suff = "_ways"
    ways_gparquet_file = file_path.with_name(file_path.with_suffix("").name + ways_suff).with_suffix(".parquet")
    with CodeTimer(f"Writing {ways_gparquet_file}"):
        gdf_ways.to_parquet(ways_gparquet_file)
    ways_gpkg_file = file_path.with_name(file_path.with_suffix("").name + ways_suff).with_suffix(".gpkg")
    with CodeTimer(f"Writing {ways_gpkg_file}"):
        gdf_ways.to_file(ways_gpkg_file, driver="GPKG", layer="edges")
    logging.info(f"\\ end osm_to_parquet {data_folder} {place}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="This extracts nodes and ways from OSM to parquet.")
    parser.add_argument(
        "-d",
        "--data_folder",
        type=Path,
        help="Folder containing T4c data folder structure",
        required=True,
    )
    parser.add_argument(
        "--place",
        type=str,
        help="City to be processed",
        required=True,
        default=None,
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
        data_folder = params["data_folder"]
        place = params["place"]
    except Exception as e:
        logging.exception(f"Could not parse args.", exc_info=e)
        parser.print_help()
        exit(1)
    osm_to_parquet(data_folder=data_folder, place=place)


if __name__ == "__main__":
    main(sys.argv[1:])
