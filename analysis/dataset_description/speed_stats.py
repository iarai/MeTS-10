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
import argparse
import ast
import logging
import os
import sys
from ast import literal_eval
from collections import defaultdict
from pathlib import Path
from typing import List
from typing import Tuple
from typing import Union

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

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

# https://wiki.openstreetmap.org/wiki/Key:highway
highway_compact_list = [
    ["motorway"],
    ["trunk", "motorway_link"],
    ["primary", "trunk_link"],
    ["secondary", "primary_link"],
    ["tertiary", "secondary_link"],
    ["unclassified", "residential", "tertiary_link"],
]
highway_flat_list = []
# https://github.com/gravitystorm/openstreetmap-carto/blob/master/road-colors.yaml
# https://github.com/gravitystorm/openstreetmap-carto/blob/a2077c0c8d40fb7f6d308b4ca4e8941d4c8b699a/style/road-colors-generated.mss
# @motorway-fill: #e892a2;
# @trunk-fill: #f9b29c;
# @primary-fill: #fcd6a4;
# @secondary-fill: #f7fabf;

highway_compact_map = {}

highway_ordering = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "cycleway",
    "road",
    "construction",
]


def osm_color_palette():
    for c in ["#e892a2", "#e892a2", "#f9b29c", "#f9b29c", "#fcd6a4", "#fcd6a4", "#f7fabf", "#f7fabf"] + ["white"] * 99:
        yield c


def update_highway_compact_map():
    global highway_flat_list
    global highway_compact_list
    for hws in highway_compact_list:
        for hw in hws:
            highway_compact_map[hw] = "/".join(hws)
    highway_flat_list = [hw for hws in highway_compact_list for hw in hws]
    assert len(highway_compact_map) == len(highway_flat_list)


update_highway_compact_map()


def load_h5_file(file_path: Union[str, Path]) -> np.ndarray:
    with h5py.File(str(file_path) if isinstance(file_path, Path) else file_path, "r") as fr:
        data = fr.get("array")
        return np.array(data)


def q01(x):
    return x.quantile(0.01)


def q99(x):
    return x.quantile(0.99)


def _stats(
    df: pd.DataFrame, cols: List[str], d: List[(Tuple[str, str])], per_highway: bool = False, skip_sum: bool = False, skip_count: bool = False, suffix: str = ""
):
    for col in cols:
        # print(f"| | {col} |  [{df[col].min():,.1f},{df[col].max():,.1f}], {df[col].mean():,.1f}+-{df[col].std():,.1f} / {df[col].median():,.1f}, {len(df[col])} |")
        sum_ = f"{df[col].sum():4.1e}" if not skip_sum else ""
        count_ = f"{len(df[col]):,}" if not skip_count else ""
        d.append(
            (
                f"{col} {suffix}",
                f"{df[col].mean():,.1f} & {df[col].std():,.1f} & {df[col].median():,.1f} & {q01(df[col]):,.1f} & {q99(df[col]):,.1f} &  {count_} & {sum_}".replace(
                    ",", "'"
                ),
            )
        )

        if per_highway:
            df = df.copy()
            df["sort_key"] = [highway_ordering.index(hw) for hw in df["highway"]]
            df = df.sort_values("sort_key")
            for rec in (
                df.groupby("highway")
                .agg(
                    mean=(col, "mean"),
                    std=(col, "std"),
                    min=(col, q01),
                    max=(col, q99),
                    median=(col, "median"),
                    count=(col, "count"),
                    sum=(col, "sum"),
                    sort_key=("sort_key", "first"),
                )
                .reset_index()
                .sort_values("sort_key")
                .to_dict("records")
            ):
                rec["count_per_edge"] = "todo"
                sum_ = f"{rec['sum']:4.1e}" if not skip_sum else ""
                count_ = f"{rec['count']:,}" if not skip_count else ""
                d.append(
                    (
                        f"\\hspace{{10pt}}  {rec['highway']}",
                        f"{rec['mean']:,.1f} & {rec['std']:,.1f} & {rec['median']:,.1f} & {rec['min']:,.1f} & {rec['max']:,.1f} &  {count_}  & {sum_}",
                    )
                )


def write_density(data_folder: Path, df_speed_classes: pd.DataFrame, city: str, year_ext: str, output_folder: Path, num_days: int, to_hour=18, from_hour=8):
    gdf_edges = gpd.read_parquet(data_folder / "road_graph" / city / "road_graph_edges.parquet")

    df_densities_8_18 = (
        df_speed_classes[(df_speed_classes["t"] >= from_hour * 4) & (df_speed_classes["t"] < to_hour * 4)]
        .groupby(by=["u", "v", "gkey"])
        .agg(density=("median_speed_kph", "count"))
    )
    df_densities_8_18["density"] = df_densities_8_18["density"] / (num_days * (to_hour - from_hour) * 4)

    gdf_densities = gdf_edges.merge(
        df_densities_8_18,
        on=["u", "v", "gkey"],
    )
    gdf_densities.to_parquet(output_folder / f"density_{from_hour}_{to_hour}_{city}{year_ext}.parquet")
    gdf_densities.to_file(output_folder / f"density_{from_hour}_{to_hour}_{city}{year_ext}.gpkg", driver="GPKG", layer="edges")


def plot_daylines(
    df_speed_classes: pd.DataFrame,
    df_freeflow: pd.DataFrame,
    city: str,
    year_ext: str,
    output_folder: Path,
    apply_bounding_box: bool,
    num_agg_days: int,
    palette="bright",
):
    hw_count = {
        rec["highway_compact"]: rec["count"]
        for rec in df_freeflow.groupby("highway_compact").agg(count=("u", "count"), highway_compact=("highway_compact", "first")).to_dict("records")
    }
    df_speed_classes["coverage"] = [1 / (hw_count[hw] * num_agg_days) for hw in df_speed_classes["highway_compact"]]

    df_speed_classes["sort_key"] = [highway_ordering.index(hw) for hw in df_speed_classes["highway"]]
    df_speed_classes = df_speed_classes.sort_values("sort_key")

    for attr, attr_human, ylim, estimator in [
        ("median_speed_kph", "MeTS-10 median speed [km/h]", [0, 120], "mean"),
        ("mean_speed_kph", "MeTS-10 mean speed [km/h]", [0, 120], "mean"),
        ("std_speed_kph", "MeTS-10 speed standard deviation [km/h]", [0, 120], "mean"),
        ("volume", "MeTS-10 volume [probes]", [0, 120], "mean"),
        ("coverage", "MeTS-10 density [-]", [0, 1], "sum"),
    ]:
        fig, ax = plt.subplots(1, tight_layout=True, figsize=(15, 8), sharey=True)
        ax.set_xticks([h * 4 for h in range(25)])
        ax.set_xticklabels([f"{h}:00" for h in range(25)])
        # https://stackoverflow.com/questions/46125182/is-seaborn-confidence-interval-computed-correctly
        sns.lineplot(data=df_speed_classes, x="t", y=attr, hue="highway_compact", ax=ax, estimator=estimator, errorbar=("pi", 80), palette=palette)
        ax.set_ylim(ylim)
        ax.set_xlim([0, 96])
        ax.set(xlabel="time of day", ylabel=attr_human)
        # https://stackoverflow.com/questions/46266700/how-to-add-legend-below-subplots-in-matplotlib
        plt.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.15))
        output_folder.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_folder / f"speed_stats_{attr}_{city}{year_ext}_by_highway{'_in_bb' if apply_bounding_box else ''}.pdf")
        plt.close(fig)


def stats(
    data_folder: Path,
    city: str,
    output_folder: Path,
    year_ext: str,
    num_agg: int = 20,
    run_cc=False,
    apply_bounding_box: bool = False,
    suffix="",
    skip_movie_volumes=True,
):
    d = []

    df_edges = pd.read_parquet(data_folder / "road_graph" / city / "road_graph_edges.parquet")
    df_freeflow = pd.read_parquet(data_folder / "road_graph" / city / "road_graph_freeflow.parquet")
    df_edges["highway"] = [ast.literal_eval(hw) if hw.startswith("[") else hw for hw in df_edges["highway"]]
    df_edges["highway"] = [hw[0] if isinstance(hw, list) else hw for hw in df_edges["highway"]]
    df_edges["sort_key"] = [highway_ordering.index(hw) for hw in df_edges["highway"]]
    df_edges = df_edges.sort_values("sort_key")
    # print(df_edges)
    # print(df_edges.columns)
    df_nodes = pd.read_parquet(data_folder / "road_graph" / city / "road_graph_nodes.parquet")
    if apply_bounding_box:

        def in_bb(x, y):
            return (x_min <= x <= x_max) and (y_min <= y <= y_max)

        # e.g.London: 51.20500, 51.70000, -0.36900, 0.06700
        y_min, y_max, x_min, x_max = [c / 1e5 for c in T4C_BBOXES[city]["bounds"]]

        df_edges = df_edges.merge(df_nodes, left_on="u", right_on="node_id", suffixes=["", "_u"]).merge(
            df_nodes, left_on="v", right_on="node_id", suffixes=["", "_v"]
        )
        df_edges.rename(columns={"x": "x_u", "y": "y_u"}, inplace=True)
        df_edges["in_bb"] = [
            (in_bb(x_u, y_u) or in_bb(x_v, y_v)) for x_u, y_u, x_v, y_v in zip(df_edges["x_u"], df_edges["y_u"], df_edges["x_v"], df_edges["y_v"])
        ]
        df_edges = df_edges[df_edges["in_bb"] == True]
        df_freeflow = df_freeflow.merge(df_nodes, left_on="u", right_on="node_id", suffixes=["", "_u"]).merge(
            df_nodes, left_on="v", right_on="node_id", suffixes=["", "_v"]
        )
        df_freeflow.rename(columns={"x": "x_u", "y": "y_u"}, inplace=True)
        df_freeflow["in_bb"] = [
            (in_bb(x_u, y_u) or in_bb(x_v, y_v)) for x_u, y_u, x_v, y_v in zip(df_freeflow["x_u"], df_freeflow["y_u"], df_freeflow["x_v"], df_freeflow["y_v"])
        ]
        df_freeflow = df_freeflow[df_freeflow["in_bb"] == True]

        nodes_in_bb = list(df_edges["u"]) + list(df_edges["v"])
        df_nodes["in_bb"] = [node_id in nodes_in_bb for node_id in df_nodes["node_id"]]
        df_nodes = df_nodes[df_nodes["in_bb"] == True]
    (lat_min, lat_max, lon_min, lon_max) = [c / 1e5 for c in T4C_BBOXES[city]["bounds"]]
    d.append((f"bounding box {suffix}", f" &  &  &  &  &  {lon_min}--{lon_max} / {lat_min}--{lat_max} &"))
    d.append((f"num_edges {suffix}", f" &  &  &  &  &  {len(df_edges):,} &"))

    for rec in df_edges.groupby("highway").agg(count=("u", "count"), sort_key=("sort_key", "first")).reset_index().sort_values("sort_key").to_dict("records"):
        d.append((f"\\hspace{{10pt}}  {rec['highway']}", f" &  &  &  &  &  {rec['count']} &"))

    d.append((f"num_nodes {suffix}", f" &  &  &  &  &  {len(df_nodes)} & "))

    df_freeflow["intersecting_cells"] = [literal_eval(intersecting_cells) for intersecting_cells in df_freeflow["intersecting_cells"]]
    df_freeflow["num_intersecting_cells"] = [len(intersecting_cells) for intersecting_cells in df_freeflow["intersecting_cells"]]

    df_freeflow["highway"] = [ast.literal_eval(hw) if hw.startswith("[") else hw for hw in df_freeflow["highway"]]
    df_freeflow["highway"] = [hw[0] if isinstance(hw, list) else hw for hw in df_freeflow["highway"]]
    for hw in df_freeflow["highway"]:
        if hw not in highway_compact_map:
            highway_compact_list[-1].append(hw)
            update_highway_compact_map()
    df_freeflow["highway_compact"] = [highway_compact_map[hw] for hw in df_freeflow["highway"]]

    # print(df_intersecting_cells)

    cell_to_edges = defaultdict(lambda: set())
    for u, v, gkey, intersecting_cells in zip(df_freeflow["u"], df_freeflow["v"], df_freeflow["gkey"], df_freeflow["intersecting_cells"]):
        for cell in intersecting_cells:
            # print((u,v))
            # print(cell)
            cell_to_edges[cell].add((u, v, gkey))
    cell_to_num_edges = {cell: len(edges) for cell, edges in cell_to_edges.items()}
    # print(cell_to_num_edges)
    df_cell_to_edges = pd.DataFrame(data={"cell": cell_to_num_edges.keys(), "edges": cell_to_edges.values()})
    df_cell_to_edges["num_edges_per_cell"] = [len(edges) for edges in cell_to_edges.values()]

    _stats(df_cell_to_edges, ["num_edges_per_cell"], d=d, skip_sum=True, suffix=suffix)

    _stats(df_freeflow, cols=["num_intersecting_cells"], d=d, skip_sum=True, suffix=suffix)

    neighbors = defaultdict(lambda: set())
    for u, v in zip(df_edges["u"], df_edges["v"]):
        neighbors[u].add(v)
        neighbors[v].add(u)
    df_node_degree = pd.DataFrame(data={"node": neighbors.keys(), "neighbors": neighbors.values()})
    df_node_degree["node_degree"] = [len(neighbors) for neighbors in df_node_degree["neighbors"]]
    _stats(df_node_degree, ["node_degree"], d=d, skip_sum=True, suffix=suffix)

    _stats(df_edges, ["length_meters"], d=d, per_highway=True, suffix=suffix)
    _stats(df_edges, ["speed_kph"], d=d, per_highway=True, skip_sum=True, suffix=suffix)
    df_freeflow["free_flow_kph-speed_kph"] = df_freeflow["free_flow_kph"] - df_freeflow["speed_kph"]

    # check we can safely merge with  free flow and have all datat
    assert len(df_freeflow) == len(df_edges), (len(df_freeflow), len(df_edges))

    # free flow undefined is recorded as -1
    # print(len(df_freeflow[df_freeflow["free_flow_kph"] < 0]))

    _stats(df_freeflow[df_freeflow["free_flow_kph"] >= 0], ["free_flow_kph", "free_flow_kph-speed_kph"], d=d, per_highway=True, skip_sum=True, suffix=suffix)
    speed_classes_files = sorted((data_folder / "speed_classes" / city).rglob("speed_classes*.parquet"))
    speed_classes_files = [speed_classes_files[index] for index in np.random.choice(len(speed_classes_files), size=num_agg, replace=False)]
    num_speed_classes_files = len(speed_classes_files)
    # print(speed_classes_files)
    df_speed_classes = pd.concat([pd.read_parquet(f) for f in speed_classes_files])
    # print(df_speed_classes.columns)
    # print(df_freeflow.columns)
    df_speed_classes = df_speed_classes.merge(df_freeflow, on=["u", "v", "gkey"])

    plot_daylines(
        df_speed_classes=df_speed_classes,
        df_freeflow=df_freeflow,
        city=city,
        year_ext=year_ext,
        output_folder=output_folder,
        apply_bounding_box=apply_bounding_box,
        num_agg_days=num_speed_classes_files,
    )
    if apply_bounding_box:
        return d, num_speed_classes_files
    write_density(
        data_folder=data_folder, df_speed_classes=df_speed_classes, city=city, year_ext=year_ext, num_days=num_speed_classes_files, output_folder=output_folder
    )

    if run_cc:
        cc_files = sorted((data_folder / "train" / city / "labels").rglob("cc_labels*.parquet"))
        cc_files_sampled = np.random.choice(len(cc_files), size=num_agg, replace=False)

        # TODO aggregation over all/more files
        for f in cc_files:
            df_cc = pd.read_parquet(f)
            # print(df_cc)
            for cc in range(1, 4):
                d[f"cc{cc}"] = f"{len(df_cc[df_cc['cc'] == cc]) / len(df_cc) * 100:,.1f}% ({len(df_cc[df_cc['cc'] == cc])}/{len(df_cc)})"
            _stats(df_cc.groupby(["u", "v", "gkey"]).agg(cc_count=("t", "count")), ["cc_count"], d)
            break
        # print(files)

    if not skip_movie_volumes:
        # TODO should we take the same days?
        movie_15min_files = sorted((data_folder / "movie_15min" / city).rglob("*.h5"))
        if len(movie_15min_files) > 0:
            movie_15_min_files_sampled = np.random.choice(len(movie_15min_files), size=num_agg, replace=False)
            vol_sums_24h = np.zeros(len(movie_15_min_files_sampled))
            vol_sums_8x3h = np.zeros((len(movie_15_min_files_sampled), 8))
            for i, index in enumerate(movie_15_min_files_sampled):
                f = movie_15min_files[index]
                movie_data = load_h5_file(f)
                vol_channels = [0, 2, 4, 6]
                vol_sums_24h[i] = np.sum(movie_data[..., vol_channels].astype(np.float64))

                for k in range(8):
                    vol_sums_8x3h[i, k] = np.sum(movie_data[k * 3 : k * 3 + 3, ..., vol_channels].astype(np.float64))

            # d.append(("daily movie volume", f"{np.mean(vol_sums) :.3e}"))
            for item, sub in [(vol_sums_24h, "00:00--24:00")] + [(vol_sums_8x3h[:, k], f"{k * 3:02d}:00--{k * 3 + 3:02}:00") for k in range(8)]:
                d.append(
                    (
                        f"movie volume {sub}",
                        f" {np.mean(item):4.1e} & {np.std(item):4.1e}  & {np.median(item):4.1e} & {np.min(item):4.1e} & {np.max(item):4.1e} &  {len(item):,} &",
                    )
                )
        else:
            d.append(("movie volume", f"n.a."))
    return d, num_speed_classes_files


def _latex_figure(fig_path: str, caption: Tuple[str, str], nopagebreak_before=False):
    lines = []
    lines.append("\\mbox{}")
    if nopagebreak_before:
        lines.append("\\nopagebreak{}")
    lines.append("\\begin{figure}[H]")
    lines.append("\\centering")
    lines.append(f"\\includegraphics[width=0.85\\textwidth]{{{fig_path}}}")
    lines.append(f"\\caption[{caption[0].strip()}]{{{caption[1].strip()}}}")
    lines.append(f"\\label{{{fig_path}}}")
    lines.append("\\end{figure}")
    return lines


def dp_stats(
    data_folder: Path,
    city: str,
    num_agg: int,
    output_folder: Path,
    year_ext: str,
    beyond_bounding_box: bool = False,
    full_figures: bool = False,
):
    default_stats_ = {}
    bbb_stats_ = {}

    # https://matplotlib.org/stable/tutorials/introductory/customizing.html#matplotlibrc-sample
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["legend.title_fontsize"] = 18
    plt.rcParams["legend.fontsize"] = 14
    plt.rcParams["xtick.major.size"] = 12
    plt.rcParams["ytick.major.size"] = 12

    if beyond_bounding_box:
        default_suffix = " (full historic road graph)"
        default_stats_[city] = stats(
            data_folder=data_folder, output_folder=output_folder, city=city, num_agg=num_agg, apply_bounding_box=False, suffix=default_suffix, year_ext=year_ext
        )

        bbb_suffix = " (MeTS-10 extent (bounding box))"
        bbb_stats_[city] = stats(
            data_folder=data_folder, output_folder=output_folder, city=city, num_agg=num_agg, apply_bounding_box=True, suffix=bbb_suffix, year_ext=year_ext
        )
    else:
        default_suffix = ""
        default_stats_[city] = stats(
            data_folder=data_folder, output_folder=output_folder, city=city, num_agg=num_agg, apply_bounding_box=False, year_ext=year_ext
        )

    latex_lines = []

    def _add_latex_lines(lines):
        for l in lines:
            print(l)
            latex_lines.append(l)
            latex_lines.append("\n")

    def _add_latex_line(line):
        print(line)
        latex_lines.append(line)
        latex_lines.append("\n")

    _add_latex_line(f"% {data_folder}")

    ext = ""
    if not data_folder.name.startswith("release"):
        ext = f" ({data_folder.name})"

    _add_latex_line(f"\\subsection{{Key Figures {city.title()}{ext}}}")
    print(f"% {num_agg} sampled files")
    _add_latex_line(f"\\subsubsection{{Road graph map {city.title()}{ext}}}")
    _add_latex_lines(
        _latex_figure(
            f"figures/{output_folder.name}/road_graph_{city}{year_ext}.jpg",
            caption=(f"Road graph {city.title()}", f"Road graph {city.title()}, OSM color scheme{ext}."),
            nopagebreak_before=True,
        )
    )

    static_config = [(default_stats_, default_suffix)]
    if beyond_bounding_box:
        static_config += [(bbb_stats_, bbb_suffix)]
    for stats_, suffix_ in static_config:
        _add_latex_line(f"\\subsubsection{{Static data  {city.title()} {ext} {suffix_}}}".strip())
        _add_latex_line("\\mbox{}\\nopagebreak")
        _add_latex_line("\\begin{small}")
        _add_latex_line("\\begin{longtable}{p{4cm}rrrrrrrr}")
        _add_latex_line("\\toprule")
        _add_latex_line("Attribute      & {mean} &{std} & {median}  & {q01} & {q99} & {data points} & {sum}  \\\\")
        _add_latex_line("\\midrule")
        _add_latex_line(f"% {data_folder} / {city}")
        # print(f" {city}  &                &                                                 \\\\")
        stats__city_, num_speed_classes_files = stats_[city]
        for i, (k, v) in enumerate(stats__city_):
            k = k.replace("_", "\\_").replace("%", "\\%")
            v = v.replace("_", "\\_").replace("%", "\\%").replace(",", "'")
            # city_ = f"\\multirow{{{len(stats_[city])}}}{{*}}{{{city}}}" if i == 0 else ""
            _add_latex_line(f" {k}               & {v}                                                \\\\")
        _add_latex_line("\\bottomrule")
        # print("\\end{longtable}")
        _add_latex_line(
            """
        \\caption[Key figures """
            f"{city.title()} {suffix_}"
            """]{Key figures """
            f"{city.title()}"
            f""" for the generated data from {num_speed_classes_files} randomly sampled days"""
            f"{suffix_}"
            """.
        \\textbf{num\\_edges} number of edges in the street network graph;
        \\textbf{num\\_nodes} number of nodes in the street network graph;
        \\textbf{num\\_edges\\_per\\_cell} number of edges a cell (row,col,heading) has in its intersecting cells;
        \\textbf{num\\_intersecting\\_cells} number of cells (row,col,heading) in an edge's intersecting cells;
        \\textbf{node\\_degree} number of (unique) neighbor nodes per node;
        \\textbf{length\\_meters} free flow speed derived from data;
        \\textbf{speed\\_kph} signalled speed;
        \\textbf{free\\_flow\\_kph} free flow speed derived from data;
        \\textbf{free\\_flow\\_kph-speed\\_kph} difference
        }
    \label{tab:key_figures:"""
            f"{data_folder}:{city.title()}:{suffix_}"
            """}
    \end{longtable}
    \end{small}
    """
        )

    _add_latex_line(f"\\subsubsection{{Segment density map  {city.title()}{ext}}}")
    _add_latex_lines(
        _latex_figure(
            f"figures/{output_folder.name}/density_8_18_{city}{year_ext}.jpg",
            caption=(
                f"Segment-wise density 8am--6pm {city.title()}",
                f"Segment-wise density 8am--6pm {city.title()} from {num_agg} randomly sampled days.",
            ),
            nopagebreak_before=True,
        )
    )
    _add_latex_line(f"\\clearpage")

    def gen_dayline_caption(item: str, city: str, suffix: str, error_hull: bool = True):
        second_sentence = (
            f"Data from {num_agg} randomly sampled days."
            if not error_hull
            else f"The error hull is the 80\\% data interval [10.0--90.0 percentiles] from {num_agg} randomly sampled days."
        )
        return (
            f"""
        Daily {item} profile {city.title()} {suffix}
        """,
            f"""
            Daily {item} profile for different road types for {city.title()} {suffix}. {second_sentence}
        """,
        )

    _add_latex_line(f"\\subsubsection{{Daily density profile  {city.title()} {ext} {default_suffix}}} ")
    _add_latex_lines(
        _latex_figure(
            f"figures/{output_folder.name}/speed_stats_coverage_{city}{year_ext}_by_highway.pdf",
            caption=gen_dayline_caption(f"density", city=city.title(), suffix=default_suffix, error_hull=False),
            nopagebreak_before=True,
        )
    )
    _add_latex_line(f"\\subsubsection{{Daily speed profile  {city.title()} {ext} {default_suffix}}}")
    _add_latex_lines(
        _latex_figure(
            f"figures/{output_folder.name}/speed_stats_median_speed_kph_{city}{year_ext}_by_highway.pdf",
            caption=gen_dayline_caption(f"median 15 min speeds of all intersecting cells", city=city.title(), suffix=default_suffix),
            nopagebreak_before=True,
        )
    )
    if full_figures:
        _add_latex_lines(
            _latex_figure(
                f"figures/{output_folder.name}/speed_stats_mean_speed_kph_{city}{year_ext}_by_highway.pdf",
                caption=gen_dayline_caption(f"mean 15 min speeds of all intersecting cells", city=city.title(), suffix=default_suffix),
            )
        )
        _add_latex_lines(
            _latex_figure(
                f"figures/{output_folder.name}/speed_stats_std_speed_kph_{city}{year_ext}_by_highway.pdf",
                caption=gen_dayline_caption(f"std 15 min speeds of all intersecting cells", city=city.title(), suffix=default_suffix),
            )
        )

    if beyond_bounding_box:
        _add_latex_line(f"\\subsubsection{{Daily density profile  {city.title()} {ext} {bbb_suffix}}}")
        _add_latex_lines(
            _latex_figure(
                f"figures/{output_folder.name}/speed_stats_coverage_{city}{year_ext}_by_highway_in_bb.pdf",
                caption=gen_dayline_caption(f"density", city=city.title(), suffix=bbb_suffix, error_hull=False),
                nopagebreak_before=True,
            )
        )
        _add_latex_line(f"\\subsubsection{{Daily speed profile  {city.title()} {ext} {bbb_suffix}}}")
        _add_latex_lines(
            _latex_figure(
                f"figures/{output_folder.name}/speed_stats_median_speed_kph_{city}{year_ext}_by_highway_in_bb.pdf",
                caption=gen_dayline_caption(f"median 15 min speeds of all intersecting cells", city=city.title(), suffix=bbb_suffix),
                nopagebreak_before=True,
            )
        )
        if full_figures:
            _add_latex_lines(
                _latex_figure(
                    f"figures/{output_folder.name}/speed_stats_mean_speed_kph_{city}{year_ext}_by_highway_in_bb.pdf",
                    caption=gen_dayline_caption(f"mean 15 min speeds of all intersecting cells", city=city.title(), suffix=bbb_suffix),
                )
            )
            _add_latex_lines(
                _latex_figure(
                    f"figures/{output_folder.name}/speed_stats_std_speed_kph_{city}{year_ext}_by_highway_in_bb.pdf",
                    caption=gen_dayline_caption(f"std 15 min speeds of all intersecting cells", city=city.title(), suffix=bbb_suffix),
                )
            )
    _add_latex_line(f"\\clearpage")
    return latex_lines


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="This script creates cell mappings for road segments.")
    parser.add_argument(
        "-d",
        "--data_folder",
        type=str,
        help="Folder containing T4c data folder structure",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        help="Folder to save the figures to. The last component will be used for the LaTeX includes.",
        required=True,
    )
    parser.add_argument("--city", type=str, help="Competition", required=False, default=["london", "madrid", "melbourne"], nargs="+")
    parser.add_argument("--num_agg", type=int, help="Competition", required=False, default=20)
    parser.add_argument(
        "-bbb",
        "--beyond_bounding_box",
        help="Add static information for full road graph and for road graph truncated to bounding box.",
        required=False,
        action="store_true",
    )
    # TODO write into separate files!
    parser.add_argument("-ff", "--full_figures", help="Add also less important figures for the generated documentation.", required=False, action="store_true")

    return parser


def main(argv):
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"), format="[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
    )
    pd.set_option("display.max_columns", None)
    parser = create_parser()
    try:

        params = parser.parse_args(argv)
        params = vars(params)
        data_folder = Path(params["data_folder"])
        output_folder = Path(params["output_folder"])
        cities = params["city"]
        num_agg = params["num_agg"]
        beyond_bounding_box = params["beyond_bounding_box"]
        full_figures = params["full_figures"]
        print(f"% {params}")
        for city in cities:
            try:
                year = data_folder.name
                year = int(year)
                year_ext = f"_{year}"
            except ValueError:
                year_ext = ""
            latex_lines = dp_stats(
                data_folder=data_folder,
                city=city,
                year_ext=year_ext,
                num_agg=num_agg,
                output_folder=output_folder,
                beyond_bounding_box=beyond_bounding_box,
                full_figures=full_figures,
            )
            with (output_folder / f"speed_stats_{city}{year_ext}.tex").open("w") as f:
                f.writelines(latex_lines)

    except Exception as e:
        logging.exception(f"Could not parse args.", exc_info=e)
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
