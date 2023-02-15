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
import ast
import datetime
import re
from collections import defaultdict
from pathlib import Path
from typing import Tuple
from typing import Union

import geojson
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

T4C_BBOXES = {
    "antwerp": {"bounds": [5100100, 5143700, 415300, 464800], "rotate": True},
    "bangkok": {"bounds": [1355400, 1404900, 10030800, 10074400]},
    "barcelona": {"bounds": [4125300, 4174800, 192500, 236100]},
    "berlin": {"bounds": [5235900, 5285400, 1318900, 1362500]},
    "chicago": {"bounds": [4160100, 4209600, -8794500, -8750900]},
    "istanbul": {"bounds": [4081000, 4130500, 2879400, 2923000]},
    "london": {"bounds": [5120500, 5170000, -36900, 6700]},
    "madrid": {"bounds": [4017700, 4067200, -392700, -349100]},
    "melbourne": {"bounds": [-3810600, -3761100, 14475700, 14519300]},
    "moscow": {"bounds": [5550600, 5594200, 3735800, 3785300], "rotate": True},
    "newyork": {"bounds": [4054400, 4103900, -7415800, -7372200]},
    "vienna": {"bounds": [4795300, 4844800, 1617300, 1660900]},
    "warsaw": {"bounds": [5200100, 5249600, 2081700, 2125300]},
    "zurich": {"bounds": [4708300, 4757800, 834500, 878100]},
}

NUM_ROWS = 495
NUM_COLUMNS = 436


def load_h5_file(file_path: Union[str, Path]) -> np.ndarray:
    with h5py.File(str(file_path) if isinstance(file_path, Path) else file_path, "r") as fr:
        data = fr.get("array")
        return np.array(data)


def get_bin_bounds(city):
    return T4C_BBOXES[city]["bounds"]


def is_rotate(city):
    return T4C_BBOXES[city].get("rotate", False)


def get_latlon_bounds(city, verbose=False) -> Tuple[Tuple[float, float, float, float], bool]:
    # Grid cells have steps of size 100. LatLon conversion is floored after the 3rd
    # decimal digit (see degree_to_grid() below). Hence grid to degree needs division
    # through 100 * 1000 = 1e5
    bbox = tuple([c / 1e5 for c in get_bin_bounds(city)])
    if verbose:
        print(f"Bounds for {city}: {bbox}")
    lat_min, lat_max, lon_min, lon_max = bbox
    rotate = is_rotate(city)
    assert np.isclose(lat_max - lat_min, (NUM_ROWS if not rotate else NUM_COLUMNS) * 0.001), lat_max - lat_min
    assert np.isclose(lon_max - lon_min, (NUM_COLUMNS if not rotate else NUM_ROWS) * 0.001), lon_max - lon_min
    return bbox, rotate


d = {
    # "2020": [
    #     "berlin",
    #     "istanbul",
    #     "moscow",
    # ],
    "2021": [
        "antwerp",
        "bangkok",
        "barcelona",
        "berlin",
        "chicago",
        "istanbul",
        "melbourne",
        "moscow",
        # "newyork",
        # "vienna",
    ],
    "2022": ["london", "madrid", "melbourne"],
}


def create_bounding_boxes(RELEASE_FOLDER):
    features = []
    for city in T4C_BBOXES:
        (lat_min, lat_max, lon_min, lon_max), _ = get_latlon_bounds(city)
        poly = geojson.Polygon([[(lon_min, lat_min), (lon_max, lat_min), (lon_max, lat_max), (lon_min, lat_max), (lon_min, lat_min)]])
        grid_bounds = get_bin_bounds(city)
        features.append(
            geojson.Feature(
                geometry=poly,
                properties={
                    "grid_bounds": grid_bounds,
                    "grid_factor": int(1e5),
                    "grid_cell_size": 100,
                    "grid_width": int((grid_bounds[3] - grid_bounds[2]) / 100),
                    "grid_height": int((grid_bounds[1] - grid_bounds[0]) / 100),
                },
            )
        )
    fc = geojson.FeatureCollection(features)
    with open(f"{RELEASE_FOLDER}/bounding_boxes.geojson", "w+", encoding="utf-8") as bbf:
        geojson.dump(fc, bbf)
    print(geojson.dumps(fc, sort_keys=True))
    return f"Wrote bounding boxes to {RELEASE_FOLDER}/bounding_boxes.geojson"


if __name__ == "__main__":
    num_agg = 20

    tables = defaultdict(lambda: [])
    year_style = True
    BASEDIR = Path("/iarai/public/t4c/data_pipeline/release20221026_residential_unclassified")
    # BASEDIR = Path("/iarai/public/t4c/data_pipeline/release20221021_residential_unclassified_do_not_use")
    print(create_bounding_boxes(BASEDIR))

    # year_style = False
    tables["markdown"].append(
        "|{city.title()} ({competition_year.title()}) | {len(dates)}  | {', '.join(ranges)}  | {lat_min, lat_max, lon_min, lon_max} | {len(df_nodes)} | {len(df_edges)} | {total_length_meters:.1f} | {mean_length_meters:.1f} |  {ratio_covered_edges:.2F} | {mapped_ratio:.2f} | {vol_sum:.3e}| {coverage:.2f} | {mean_volume:.2f}  |"
    )
    tables["markdown"].append(
        "| :------------------------------------------|:--------------|:---------------------|:-------------------------------------|:----------------|:----------------|:--------------------------|:-------------------------|:---------------------------|:-------------------|:-------------|:---------------|:-------------------|"
    )

    tables["latex"].append("\\begin{tabular}{llp{3cm}llll}")
    tables["latex"].append("\\toprule")
    tables["latex"].append("city (t4c year) & days & date ranges & 8--18 coverage & mapped ratio \\\\")
    tables["latex"].append("\\midrule")

    tables["latex_full"].append("\\begin{tabular}{llp{3cm}llll}")
    tables["latex_full"].append("\\toprule")
    tables["latex_full"].append(
        "city (t4c year) & days & date ranges & lat\_min, lat\_max, lon\_min, lon\_max & nodes & edges & total segment length [m] & mean segment length [m] &  ratio covered edges & mapped ratio  & daily fcd data & 8--18 coverage & mean segment volume  \\\\"
    )
    tables["latex_full"].append("\\midrule")

    for competition_year, cities in d.items():

        year_basedir = BASEDIR
        if year_style:
            year_basedir = BASEDIR / competition_year
        for city in cities:
            print((competition_year, city))

            road_graph_dir = year_basedir / "road_graph" / city

            df_edges = pq.read_table(road_graph_dir / "road_graph_freeflow.parquet").to_pandas()
            df_nodes = pq.read_table(road_graph_dir / "road_graph_nodes.parquet").to_pandas()

            # print(df_edges.columns)
            intersecting_cells = [ast.literal_eval(ic) for ic in df_edges["intersecting_cells"]]
            # print(type(intersecting_cells[0]))
            ics = {c[:3] for ic in intersecting_cells for c in ic}
            # print(ics)
            # print(len(ics) / (495 * 436 * 4))

            mask = np.zeros((495, 436, 4))
            neg_mask = np.full((495, 436, 4), fill_value=1)
            for r, c, h in ics:
                mask[r, c, h] = 1
                neg_mask[r, c, h] = 0

            mapped_ratio = 0

            dates = []
            resolved_movie_dir = (year_basedir / "movie_15min" / city).resolve()
            vol_sum = 0
            # TODO random choice for movie files
            movie_files = list(resolved_movie_dir.rglob("**/*8ch_15min.h5"))
            assert len(movie_files) >= num_agg, (len(movie_files), num_agg)
            for file in movie_files:
                date = re.search(r"([0-9]{4}-[0-9]{2}-[0-9]{2})", str(file)).group(1)
                date = datetime.datetime.strptime(date, "%Y-%m-%d")
                dates.append(date)
                if len(dates) <= num_agg:
                    data = load_h5_file(file).astype(np.float64)
                    vols = data[..., [0, 2, 4, 6]]
                    total_vol_date = np.sum(vols)
                    vol_sum += total_vol_date
                    mapped_ratio += np.sum(vols * mask) / total_vol_date
            vol_sum /= num_agg
            mapped_ratio /= num_agg

            dates = list(sorted(dates))
            ranges = []
            for year_ in [2018, 2019, 2020, 2021]:
                year_dates = [date for date in dates if date.year == year_]
                if len(year_dates) > 0:
                    start_date = year_dates[0].strftime("%Y-%m")
                    end_date = year_dates[-1].strftime("%Y-%m")
                    if start_date == end_date:
                        ranges.append(start_date)
                    else:
                        ranges.append(f"{start_date}--{end_date}")  # ({(year_dates[-1]-year_dates[0]).days+1})")

            (lat_min, lat_max, lon_min, lon_max), _ = get_latlon_bounds(city)

            # print(df_edges[(df_edges["highway"]=="residential")&((df_edges["speed_kph"]-df_edges["free_flow_kph"])<-80)])

            speed_classes_files = list((year_basedir / "speed_classes" / city).rglob("speed_classes_*.parquet"))
            assert len(speed_classes_files) >= num_agg, (len(speed_classes_files), num_agg)
            speed_classes_files = [speed_classes_files[index] for index in np.random.choice(len(speed_classes_files), size=num_agg, replace=False)]
            # print(speed_classes_files)
            df_speeds = pd.concat([pd.read_parquet(f) for f in speed_classes_files])

            coverage = df_speeds[(df_speeds["t"] >= 8 * 4) & (df_speeds["t"] < 18 * 4)].groupby(["day", "t"]).agg(count=("median_speed_kph", "count"))[
                "count"
            ].mean() / len(df_edges)
            mean_volume = df_speeds[(df_speeds["t"] >= 8 * 4) & (df_speeds["t"] < 18 * 4)]["volume"].mean()
            ratio_covered_edges = len(df_speeds.groupby(["u", "v", "gkey"])) / len(df_edges)
            total_length_meters = df_edges["length_meters"].sum()
            mean_length_meters = total_length_meters / len(df_edges)

            tables["markdown"].append(
                f"|{city.title()} ({competition_year.title()}) | {len(dates)}  | {', '.join(ranges)}  | {lat_min, lat_max, lon_min, lon_max} | {len(df_nodes)} | {len(df_edges)} | {total_length_meters:.1f} | {mean_length_meters:.1f} |  {ratio_covered_edges:.2F} | {mapped_ratio:.2f} | {vol_sum:.3e}| {coverage:.2f} | {mean_volume:.2f}  |"
            )
            tables["latex"].append(
                f" {city.title()} ({competition_year.title()}) & {len(dates)}  & {', '.join(ranges)}  & {coverage:.2f} & {mapped_ratio:.2f}\\\\"
            )

            tables["latex_full"].append(
                f"{city.title()} ({competition_year.title()}) & {len(dates)}  & {', '.join(ranges)}  & {lat_min, lat_max, lon_min, lon_max} & {len(df_nodes)} & {len(df_edges)} & {total_length_meters:.1f} & {mean_length_meters:.1f} &  {ratio_covered_edges:.2F} & {mapped_ratio:.2f} & {vol_sum:.3e}& {coverage:.2f} & {mean_volume:.2f} \\\\"
            )

    tables["latex"].append("\\bottomrule")
    tables["latex"].append("\\end{tabular}")
    tables["latex_full"].append("\\bottomrule")
    tables["latex_full"].append("\\end{tabular}")
    for style, lines in tables.items():
        print(style)
        for line in lines:
            print(line)
