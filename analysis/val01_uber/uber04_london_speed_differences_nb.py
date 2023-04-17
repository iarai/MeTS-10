# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
# +
import ast
from collections import defaultdict
from pathlib import Path

import folium
import humanize
import numpy as np
import pandas
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# U for Uber, T for T4c, O for OpenStreetMap
START_DAY = 21
END_DAY = 27  # incl.
UBASEPATH = Path("/iarai/public/t4c/uber")
TBASEPATH = Path("/iarai/public/t4c/data_pipeline/release20221028_historic_uber")
OBASEPATH = Path("/iarai/public/t4c/osm")

CITY = "london"
YEAR = 2019
MONTH = 10

DAYTIME_START_HOUR = 8
DAYTIME_END_HOUR = 18
DAYTIME_HOURS = DAYTIME_END_HOUR - DAYTIME_START_HOUR
# -

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

plt.rcParams["axes.labelsize"] = 32
plt.rcParams["legend.title_fontsize"] = 28
plt.rcParams["legend.fontsize"] = 24

# simplified_filter = lambda hw: hw not in ['residential', 'living_street', 'service', 'road']
simplified_filter = lambda hw: hw in [
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
]


def osm_color_palette():
    for c in ["#e892a2", "#e892a2", "#f9b29c", "#f9b29c", "#fcd6a4", "#fcd6a4", "#f7fabf", "#f7fabf"] + ["white"] * 99:
        yield c


# ## Load Uber segment speeds daytime and within bounding box

uspeeds_df = pandas.read_parquet(UBASEPATH / f"movement-speeds-hourly-{CITY}-{YEAR}-{MONTH}.parquet")
uspeeds_df = uspeeds_df[(uspeeds_df["day"] >= START_DAY) & (uspeeds_df["day"] <= END_DAY)]
uspeeds_df["speed_kph_mean"] = uspeeds_df["speed_mph_mean"] * 1.60934
uspeeds_df = uspeeds_df.rename(columns={"osm_start_node_id": "u", "osm_end_node_id": "v", "osm_way_id": "osmid"})
uspeeds_df["gkey"] = [f"{u}_{v}_{osmid}" for u, v, osmid in zip(uspeeds_df["u"], uspeeds_df["v"], uspeeds_df["osmid"])]
uspeeds_df

# ## Load our segment speeds and aggregate by hour

tspeeds_l = []
for i in range(START_DAY, END_DAY + 1):
    f = TBASEPATH / "speed_classes" / CITY / f"speed_classes_{YEAR}-{MONTH}-{i}.parquet"
    print(f)
    df = pd.read_parquet(f)
    # ['u', 'v', 'gkey', 'osmid', 'day', 't', 'volume_class', 'volume','median_speed_kph', 'mean_speed_kph', 'std_speed_kph', 'free_flow_kph'],
    #     print(df.columns)
    df["date"] = df["day"]
    df["year"] = YEAR
    df["month"] = MONTH
    df["day"] = i
    df["hour"] = df["t"] // 4
    df = (
        df[["u", "v", "gkey", "year", "month", "day", "hour", "volume", "volume_class", "median_speed_kph", "free_flow_kph", "std_speed_kph"]]
        .groupby(by=["u", "v", "gkey", "year", "month", "day", "hour"])
        .agg(
            volume_class=("volume_class", "mean"),
            volume=("volume", "mean"),
            std_speed_kph=("std_speed_kph", "mean"),
            median_speed_kph=("median_speed_kph", "mean"),
            free_flow_kph=("free_flow_kph", "first"),
        )
        .reset_index()
    )
    tspeeds_l.append(df)
tspeeds_df = pandas.concat(tspeeds_l)
# tspeeds_df = tspeeds_df.rename(columns={'osmid': 'osm_way_id'})
tspeeds_df

u_ids = set(zip(uspeeds_df["u"], uspeeds_df["v"], uspeeds_df["gkey"]))
u_ids

t_ids = set(zip(tspeeds_df["u"], tspeeds_df["v"], tspeeds_df["gkey"]))
t_ids

# +
# does not hold with the restricted time frame...
# assert t_ids.issubset(u_ids)
# -

# ## Restrict to daytime only

tspeeds_df = tspeeds_df[(tspeeds_df["hour"] >= DAYTIME_START_HOUR) & (tspeeds_df["hour"] < DAYTIME_END_HOUR)]
tspeeds_df

uspeeds_df = uspeeds_df[(uspeeds_df["hour"] >= DAYTIME_START_HOUR) & (uspeeds_df["hour"] < DAYTIME_END_HOUR)]
uspeeds_df

# ## Merge with road graph in bounding box

df_edges = pd.read_parquet(TBASEPATH / "road_graph" / CITY / "road_graph_edges.parquet")
df_edges

df_nodes = pd.read_parquet(TBASEPATH / "road_graph" / CITY / "road_graph_nodes.parquet")
df_nodes

df_edges = df_edges.merge(df_nodes, left_on="u", right_on="node_id", suffixes=["", "_u"]).merge(df_nodes, left_on="v", right_on="node_id", suffixes=["", "_v"])
df_edges.rename(columns={"x": "x_u", "y": "y_u"}, inplace=True)
del df_edges["geometry_u"]
del df_edges["geometry_v"]
df_edges


def in_bb(x, y):
    return (x_min <= x <= x_max) and (y_min <= y <= y_max)


# London
y_min, y_max, x_min, x_max = 51.20500, 51.70000, -0.36900, 0.06700

df_edges["in_bb"] = [(in_bb(x_u, y_u) or in_bb(x_v, y_v)) for x_u, y_u, x_v, y_v in zip(df_edges["x_u"], df_edges["y_u"], df_edges["x_v"], df_edges["y_v"])]

df_edges = df_edges[df_edges["in_bb"] == True]

ut_merged = uspeeds_df.merge(tspeeds_df, on=["u", "v", "gkey", "year", "month", "day", "hour"]).merge(df_edges, on=["u", "v", "gkey"])
ut_merged

# ## Matching rate

uspeeds_with_road_graph = uspeeds_df.merge(df_edges, on=["u", "v", "gkey"])
uspeeds_with_road_graph

humanize.metric(len(ut_merged))

len(ut_merged) / len(tspeeds_df)

humanize.metric(len(tspeeds_df))

# road_graph is only within bb
len(uspeeds_with_road_graph) / len(uspeeds_df)

humanize.metric(len(uspeeds_df))

len(ut_merged) / len(uspeeds_with_road_graph[uspeeds_with_road_graph["in_bb"] == True])

len(tspeeds_df)

len(uspeeds_with_road_graph)

# ### Speed Differences

ut_merged.columns

ut_merged["speed_diff"] = ut_merged["median_speed_kph"] - ut_merged["speed_kph_mean"]
ut_merged["sort_key"] = [highway_ordering.index(hw) for hw in ut_merged["highway"]]
ut_merged = ut_merged.sort_values("sort_key")
ut_merged

fig, ax = plt.subplots(1, figsize=(20, 10), tight_layout=True)
sns.boxplot(
    ut_merged[[simplified_filter(hw) for hw in ut_merged["highway"]]],
    x="highway",
    y="std_speed_kph",
    notch=True,
    sym="",
    palette=osm_color_palette(),
    medianprops={"color": "coral"},
    ax=ax,
)
ax.tick_params(axis="x", which="major", labelsize=32, rotation=45)
ax.tick_params(axis="y", which="major", labelsize=32)
ax.grid(axis="y")
ax.set(xlabel="road type (highway)", ylabel="MeTS-10 speed standard deviation [km/h]               ")
#  t4c: higher stds in link classes than in the corresponding road class they link -> plausibily, they get probes from the higher class running parallel.
plt.savefig(f"{CITY.title()}_t4c_std_speed_kph.pdf")

fig, ax = plt.subplots(1, figsize=(20, 10), tight_layout=True)
sns.boxplot(
    ut_merged[[simplified_filter(hw) for hw in ut_merged["highway"]]],
    x="highway",
    y="volume",
    notch=True,
    sym="",
    palette=osm_color_palette(),
    medianprops={"color": "coral"},
    ax=ax,
)
ax.tick_params(axis="x", which="major", labelsize=32, rotation=45)
ax.tick_params(axis="y", which="major", labelsize=32)
ax.grid(axis="y")
ax.set(xlabel="road type (highway)", ylabel="volume MeTS-10 [probes]    ")
# t4c: higher volumes on links than on the class below -> plausibily, they get probes from the higher class running parallel.
plt.savefig(f"{CITY.title()}_t4c_volume.pdf")

# +
fig, ax = plt.subplots(1, figsize=(20, 10), tight_layout=True)
sns.boxplot(
    ut_merged[[simplified_filter(hw) for hw in ut_merged["highway"]]],
    x="highway",
    y="speed_mph_stddev",
    notch=True,
    sym="",
    palette=osm_color_palette(),
    medianprops={"color": "coral"},
    ax=ax,
)
ax.tick_params(axis="x", which="major", labelsize=32, rotation=45)
ax.tick_params(axis="y", which="major", labelsize=32)
ax.grid(axis="y")
ax.set(xlabel="road type (highway)", ylabel="Uber speed standard deviation [mph]        ")

# Uber: link class have higher std than the class they link to (apart from motorway) -> acceleration/deceleration on links plausible
plt.savefig(f"{CITY.title()}_Uber_speed_mph_stddev.pdf")
# -

fig, ax = plt.subplots(1, figsize=(20, 10), tight_layout=True)
# sns.boxplot(ut_merged, x='highway', y='speed_kph_mean', ax=ax, palette=osm_color_palette())
sns.boxplot(
    ut_merged[[simplified_filter(hw) for hw in ut_merged["highway"]]],
    x="highway",
    y="speed_kph_mean",
    notch=True,
    sym="",
    palette=osm_color_palette(),
    medianprops={"color": "coral"},
    ax=ax,
)
ax.tick_params(axis="x", which="major", labelsize=32, rotation=45)
ax.tick_params(axis="y", which="major", labelsize=32)
ax.grid(axis="y")
ax.set(xlabel="road type (highway)", ylabel="Uber speed [km/h]    ")
plt.savefig(f"{CITY.title()}_Uber_speed_kph_mean.pdf")

fig, ax = plt.subplots(1, figsize=(20, 10), tight_layout=True)
# sns.boxplot(ut_merged, x='highway', y='speed_kph_mean', ax=ax, palette=osm_color_palette())
sns.boxplot(
    ut_merged[[simplified_filter(hw) for hw in ut_merged["highway"]]],
    x="highway",
    y="median_speed_kph",
    notch=True,
    sym="",
    palette=osm_color_palette(),
    medianprops={"color": "coral"},
    ax=ax,
)
ax.tick_params(axis="x", which="major", labelsize=32, rotation=45)
ax.tick_params(axis="y", which="major", labelsize=32)
ax.grid(axis="y")
ax.set(xlabel="road type (highway)", ylabel="MeTS-10 speed [km/h]    ")
plt.savefig(f"{CITY.title()}_Uber_median_speed_kph.pdf")

# +
fig, ax = plt.subplots(1, figsize=(20, 10), tight_layout=True)
sns.boxplot(
    ut_merged[[simplified_filter(hw) for hw in ut_merged["highway"]]],
    x="highway",
    y="speed_diff",
    notch=True,
    sym="",
    palette=osm_color_palette(),
    medianprops={"color": "coral"},
    ax=ax,
)

ax.tick_params(axis="x", which="major", labelsize=32, rotation=45)
ax.tick_params(axis="y", which="major", labelsize=32)
ax.grid(axis="y")
ax.set(xlabel="road type (highway)", ylabel="speed diff (MeTS-10 - Uber) [km/h]              ")
plt.savefig(f"{CITY.title()}_Uber_speed_diff.pdf")


# -


# ## APE -- absolute percentage error

ut_merged["ape"] = [abs(d) for d in ut_merged["speed_diff"]]
ut_merged["ape"] = ut_merged["ape"] / ut_merged["speed_kph_mean"] * 100

ut_merged["ape"].mean()

ut_merged[[simplified_filter_wo_links(hw) for hw in ut_merged["highway"]]]["ape"].mean()

ut_merged[[simplified_filter(hw) for hw in ut_merged["highway"]]]["ape"].median()

# +
fig, ax = plt.subplots(1, figsize=(20, 12), tight_layout=True)
sns.boxplot(
    ut_merged[[simplified_filter(hw) for hw in ut_merged["highway"]]],
    x="highway",
    y="ape",
    notch=True,
    sym="",
    palette=osm_color_palette(),
    medianprops={"color": "coral"},
    showmeans=True,
    meanprops={"marker": "x", "markeredgecolor": "blue", "markersize": "20"},
    ax=ax,
)

ax.tick_params(axis="x", which="major", labelsize=32, rotation=45)
ax.tick_params(axis="y", which="major", labelsize=32)
ax.set_yticks([10 * t for t in range(12)], minor=True)
ax.tick_params(axis="y", which="minor", labelsize=25)
ax.grid(axis="y", which="both")
ax.set(xlabel="road type (highway)", ylabel="absolute percentage error (MeTS-10 vs. Uber) [%]                   ")
plt.savefig(f"{CITY.title()}_Uber_ape.pdf")


# -

# ## APE by size


def get_size(l):
    if l <= 100:
        return "S (<=100m)"
    elif l <= 500:
        return "M (>100m,<=500m)"
    else:
        return "L (>500m)"


ut_merged["size"] = [get_size(l) for l in ut_merged["length_meters"]]

simplified_filter_wo_links = lambda hw: hw in [
    "motorway",
    #     "motorway_link",
    "trunk",
    #     "trunk_link",
    "primary",
    #     "primary_link",
    "secondary",
    #     "secondary_link",
    "tertiary",
    #     "tertiary_link",
]


ut_merged[[simplified_filter(hw) for hw in ut_merged["highway"]]].groupby(["highway", "size"]).mean()["ape"]

# +
# fig, ax = plt.subplots(1, figsize=(20, 12), tight_layout=True)
# sns.barplot(data=ut_merged[[simplified_filter(hw) for hw in ut_merged["highway"]]].groupby(["highway","size"]).mean().reset_index().sort_values("sort_key"),
#            x="highway", y="ape", hue="size", ax=ax)

fig, ax = plt.subplots(1, figsize=(20, 12), tight_layout=True)
sns.boxplot(
    data=ut_merged[[simplified_filter(hw) for hw in ut_merged["highway"]]],
    x="highway",
    y="ape",
    notch=True,
    sym="",
    hue="size",
    hue_order=["S (<=100m)", "M (>100m,<=500m)", "L (>500m)"],
    # palette=osm_color_palette(),
    medianprops={"color": "coral"},
    showmeans=True,
    meanprops={"marker": "x", "markeredgecolor": "blue", "markersize": "20"},
    ax=ax,
)
ax.tick_params(axis="x", which="major", labelsize=32, rotation=45)
ax.tick_params(axis="y", which="major", labelsize=32)
ax.grid(axis="y")
ax.set_ylim([0, 100])
ax.set(xlabel="road type (highway)", ylabel="absolute percentage error (MeTS-10 vs. Uber) [%]                 ")
plt.savefig(f"{CITY.title()}_Uber_ape_size.pdf")
# -

ut_merged[[simplified_filter(hw) for hw in ut_merged["highway"]]].groupby(["size"]).mean()["ape"]

ut_merged[[simplified_filter(hw) for hw in ut_merged["highway"]]].groupby(["size"]).median()["ape"]


# ## complex road segments

df_cell_mapping = pd.read_parquet(TBASEPATH / "road_graph" / CITY / "road_graph_intersecting_cells.parquet")
df_cell_mapping["intersecting_cells"] = [ast.literal_eval(c) for c in df_cell_mapping["intersecting_cells"]]
df_cell_mapping.columns

# cell mapping contains the whole road graph, not only box... therefore, the zeros!
sns.histplot([len(ics) for ics in df_cell_mapping["intersecting_cells"]], discrete=True)

hashy = defaultdict(lambda: set())
for u, v, gkey, ic in zip(df_cell_mapping["u"], df_cell_mapping["v"], df_cell_mapping["gkey"], df_cell_mapping["intersecting_cells"]):
    #     print((u,v,gkey,ic))
    for c in ic:
        hashy[c[:3]].add((u, v, gkey))
#         print(c[:3])

num_segments = []
for cell, segments in hashy.items():
    #     print(cell)
    #     print(segments)
    #     print(len(segments))
    num_segments.append(len(segments))
#     raise
sns.histplot(num_segments, discrete=True)

df_cell_mapping["complex"] = [
    np.max([len(hashy[c[:3]]) for c in intersecting_cells], initial=0) for intersecting_cells in df_cell_mapping["intersecting_cells"]
]
df_cell_mapping

df_cell_mapping["is_complex"] = df_cell_mapping["complex"] >= 5

sns.histplot(df_cell_mapping, x="complex", discrete=True)

ut_merged = ut_merged.merge(df_cell_mapping, on=["u", "v", "gkey"], suffixes=["", "_cell_mapping"])
ut_merged.columns

sns.histplot(ut_merged.groupby(["u", "v", "gkey"]).first().reset_index(), x="complex", discrete=True)

ut_merged.groupby(["u", "v", "gkey"]).first().reset_index()

road_counts = (
    ut_merged.groupby(["u", "v", "gkey"])
    .first()
    .reset_index()
    .groupby("highway")
    .agg(count=("gkey", "count"), sort_key=("sort_key", "first"))
    .reset_index()
    .sort_values("sort_key")
)
road_counts

road_counts_complex_non_complex = (
    ut_merged.groupby(["u", "v", "gkey"])
    .first()
    .reset_index()
    .groupby(["highway", "is_complex"])
    .agg(count=("gkey", "count"), sort_key=("sort_key", "first"))
    .reset_index()
    .sort_values("sort_key")
)
road_counts_complex_non_complex

fig, ax = plt.subplots(1, figsize=(20, 12), tight_layout=True)
sns.barplot(
    data=road_counts[[simplified_filter(hw) for hw in road_counts["highway"]]],
    x="highway",
    y="count",
    palette=osm_color_palette(),
    linewidth=1,
    edgecolor=".5",
    ax=ax,
)
ax.tick_params(axis="x", which="major", labelsize=32, rotation=90)
ax.tick_params(axis="y", which="major", labelsize=32)
ax.set_yticks([10 * t for t in range(12)], minor=True)
ax.tick_params(axis="y", which="minor", labelsize=25)
ax.grid(axis="y", which="both")
ax.set(xlabel="road type (highway)", ylabel="number of segments [-]")
ax.bar_label(container=ax.containers[0], labels=[int(f) for f in ax.containers[0].datavalues], size=20)
plt.savefig(f"{CITY.title()}_Uber_segment_counts.pdf")

fig, ax = plt.subplots(1, figsize=(20, 12), tight_layout=True)
sns.barplot(
    data=road_counts_complex_non_complex[[simplified_filter(hw) for hw in road_counts_complex_non_complex["highway"]]],
    x="highway",
    y="count",
    hue="is_complex",
    #     palette=osm_color_palette(),
    linewidth=1,
    edgecolor=".5",
    ax=ax,
)
ax.tick_params(axis="x", which="major", labelsize=32, rotation=90)
ax.tick_params(axis="y", which="major", labelsize=32)
ax.set_yticks([10 * t for t in range(12)], minor=True)
ax.tick_params(axis="y", which="minor", labelsize=25)
ax.grid(axis="y", which="both")
ax.set(xlabel="road type (highway)", ylabel="number of segments [-]")
ax.bar_label(container=ax.containers[0], labels=[int(f) for f in ax.containers[0].datavalues], size=20)
ax.bar_label(container=ax.containers[1], labels=[int(f) for f in ax.containers[1].datavalues], size=20)
plt.savefig(f"{CITY.title()}_Uber_segment_counts_complex_non_complex.pdf")

# ## APE complex vs. non-complex

fig, ax = plt.subplots(1, figsize=(20, 12), tight_layout=True)
sns.boxplot(
    ut_merged[(~ut_merged["is_complex"]) & [simplified_filter(hw) for hw in ut_merged["highway"]]],
    x="highway",
    y="ape",
    notch=True,
    sym="",
    hue="size",
    hue_order=["S (<=100m)", "M (>100m,<=500m)", "L (>500m)"],
    # palette=osm_color_palette(),
    medianprops={"color": "coral"},
    # flierprops={"marker": "x"},
    # boxprops={"facecolor": (.4, .6, .8, .5)},
    showmeans=True,
    meanprops={"marker": "x", "markeredgecolor": "blue", "markersize": "20"},
    ax=ax,
)
ax.tick_params(axis="x", which="major", labelsize=32, rotation=45)
ax.tick_params(axis="y", which="major", labelsize=32)
ax.set_yticks([10 * t for t in range(12)], minor=True)
ax.tick_params(axis="y", which="minor", labelsize=25)
ax.grid(axis="y", which="both")
ax.set(xlabel="road type (highway)", ylabel="absolute percentage error (MeTS-10 vs. Uber) [%]                   ")
ax.set_ylim([0, 100])
plt.savefig(f"{CITY.title()}_Uber_ape_non_complex.pdf")

# +
fig, ax = plt.subplots(1, figsize=(20, 12), tight_layout=True)
sns.boxplot(
    ut_merged[(ut_merged["is_complex"]) & [simplified_filter(hw) for hw in ut_merged["highway"]]],
    x="highway",
    y="ape",
    notch=True,
    sym="",
    hue="size",
    hue_order=["S (<=100m)", "M (>100m,<=500m)", "L (>500m)"],
    # palette=osm_color_palette(),
    medianprops={"color": "coral"},
    showmeans=True,
    meanprops={"marker": "x", "markeredgecolor": "blue", "markersize": "20"},
    ax=ax,
)

ax.tick_params(axis="x", which="major", labelsize=32, rotation=45)
ax.tick_params(axis="y", which="major", labelsize=32)
ax.set_yticks([10 * t for t in range(12)], minor=True)
ax.tick_params(axis="y", which="minor", labelsize=25)
ax.grid(axis="y", which="both")
ax.set(xlabel="road type (highway)", ylabel="absolute percentage error (MeTS-10 vs. Uber) [%]                   ")
ax.set_ylim([0, 100])
plt.savefig(f"{CITY.title()}_Uber_ape_complex.pdf")
# -

ut_merged[~ut_merged["is_complex"]].groupby(["highway", "size"]).mean()["ape"]

ut_merged[ut_merged["is_complex"]].groupby(["highway", "size"]).mean()["ape"]

raise

df_edges[df_edges["u"] == 3240063179]

tspeeds_df.columns


def plot_dayline(u, v, day):
    fig, ax = plt.subplots(1, figsize=(10, 5), tight_layout=True, sharex=True, sharey=True)
    ax2 = ax.twinx()

    t_data = tspeeds_df[(tspeeds_df["u"] == u) & (tspeeds_df["v"] == v) & (tspeeds_df["day"] == day)].sort_values("hour")
    sns.lineplot(t_data, x="hour", y="median_speed_kph", ax=ax, color="orange")
    sns.lineplot(t_data, x="hour", y="volume", ax=ax, color="red")
    ax.errorbar(t_data["hour"], t_data["median_speed_kph"], yerr=t_data["std_speed_kph"], capsize=2, capthick=0.5, color="orange", linewidth=0.5, linestyle=":")

    u_data = uspeeds_df[(uspeeds_df["u"] == u) & (uspeeds_df["v"] == v) & (uspeeds_df["day"] == day)].sort_values("hour")
    sns.lineplot(u_data, x="hour", y="speed_kph_mean", ax=ax2, color="blue")
    ax.errorbar(u_data["hour"], u_data["speed_kph_mean"], yerr=u_data["speed_mph_stddev"], capsize=2, capthick=0.5, color="blue", linewidth=0.5, linestyle=":")
    rec = df_edges[(df_edges["u"] == u) & (df_edges["v"] == v)].iloc[0]
    [min([u_data["speed_kph_mean"].min(), t_data["median_speed_kph"].min()]), max([u_data["speed_kph_mean"].max(), t_data["median_speed_kph"].max()])]
    ax.set_ylim([0, 120])
    ax2.set_ylim([0, 120])
    #     print(rec)
    plt.title(f"{rec['name']}, {rec['highway']} {rec['length_meters']:.2f}m")


# +
#         # 3240063179	3153656030		['King William Street', 'London Bridge']
#         ("London bridge, northbound", 3240063179, 3153656030),
#         # 2180693488	1868411754	A3	Elephant and Castle
#         ("Elephant and castle, northbound", 2180693488, 1868411754),
#         #	1178910690	197630	0		A40	['Marylebone Flyover', 'Westway']
#         ("Marylebone flyover, eastbound", 2180693488, 1868411754),
#         # 195975	28419372		M25		motorway
#         ("M25 near Waltham, eastbound", 195975, 28419372),
#         # 	208885668	257550997	M25		motorway
#         ("M25 near Potters Bar, westbound", 208885668, 257550997)
# -

# ### "London bridge, northbound"

# +
# u = 3240063179, v = 1249708436,
# -

u = 1249708436
v = 25161387

plot_dayline(u=u, v=v, day=21)

plot_dayline(u=u, v=v, day=22)

plot_dayline(u=u, v=v, day=23)

plot_dayline(u=u, v=v, day=24)

# ###         ("Marylebone flyover, eastbound", 2180693488, 1868411754),

u = 2180693488
v = 3890206825

plot_dayline(u=u, v=v, day=21)

plot_dayline(u=u, v=v, day=22)

plot_dayline(u=u, v=v, day=23)

plot_dayline(u=u, v=v, day=24)

plot_dayline(u=u, v=v, day=25)

# Saturday
plot_dayline(u=u, v=v, day=26)

plot_dayline(u=u, v=v, day=27)

# ### ("M25 near Waltham, eastbound", 195975, 28419372),

u = 195975
v = 195965

plot_dayline(u=u, v=v, day=21)

plot_dayline(u=u, v=v, day=22)

plot_dayline(u=u, v=v, day=23)

plot_dayline(u=u, v=v, day=24)

# ###  ("M25 near Potters Bar, westbound", 208885668, 257550997)

u = 208885668
v = 195875

plot_dayline(u=u, v=v, day=21)

plot_dayline(u=u, v=v, day=22)

plot_dayline(u=u, v=v, day=23)

# ## KDE

fig, ax = plt.subplots(1, tight_layout=True, figsize=(10, 10))
sns.kdeplot(
    data=ut_merged[[simplified_filter(hw) and "link" not in hw for hw in ut_merged["highway"]]], x="median_speed_kph", y="speed_kph_mean", hue="highway", ax=ax
)
ax.plot([0, 130], [0, 130], ls="--", c=".3")
ax.set_xlim([0, 130])
ax.set_ylim([0, 130])
ax.grid(axis="both")
ax.set(xlabel="MeTS-10 speed [km/h]", ylabel="Uber speed [km/h]")
plt.savefig(f"{CITY.title()}_Uber_kde_highway_non_links.pdf")

# +
# fig, ax = plt.subplots(1, tight_layout=True, figsize=(10, 10))
# sns.kdeplot(
#     data=ut_merged[[simplified_filter(hw) and "link" in hw for hw in ut_merged["highway"]]], x="median_speed_kph", y="speed_kph_mean", hue="highway", ax=ax
# )
# ax.plot([0, 130], [0, 130], ls="--", c=".3")
# ax.set_xlim([0, 130])
# ax.set_ylim([0, 130])
# ax.grid(axis="both")
# ax.set(xlabel='MeTS-10 speed [km/h]',
#        ylabel='Uber speed [km/h]')
# plt.savefig(f"{CITY.title()}_Uber_kde_highway_links.pdf")

# +
fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(20, 10))
ax0, ax1 = axs
sns.scatterplot(
    data=ut_merged[[simplified_filter(hw) and "link" not in hw for hw in ut_merged["highway"]] & (ut_merged["day"] == END_DAY)],
    x="median_speed_kph",
    y="speed_kph_mean",
    hue="highway",
    ax=ax0,
)
sns.scatterplot(
    data=ut_merged[[simplified_filter(hw) and "link" in hw for hw in ut_merged["highway"]] & (ut_merged["day"] == END_DAY)],
    x="median_speed_kph",
    y="speed_kph_mean",
    hue="highway",
    ax=ax1,
)

for ax in axs:
    ax.plot([0, 130], [0, 130], ls="--", c=".3")
    ax.set_xlim([0, 130])
    ax.set_ylim([0, 130])
    ax.set(xlabel="MeTS-10 speed [km/h]", ylabel="Uber speed [km/h]")

plt.savefig(f"{CITY.title()}_Uber_scatter_highway.png")
# -

fig, ax = plt.subplots(1, tight_layout=True, figsize=(10, 10))
sns.histplot(data=ut_merged[[simplified_filter(hw) for hw in ut_merged["highway"]]], x="speed_diff", hue="highway", element="step", stat="density")
# ax.set_yscale('log')
ax.grid()
plt.savefig(f"{CITY.title()}_Uber_histogram_highway.pdf")

fig, ax = plt.subplots(1, tight_layout=True, figsize=(10, 10))
sns.histplot(
    data=ut_merged[[simplified_filter(hw) for hw in ut_merged["highway"]]],
    x="speed_diff"
    #              , hue="highway"
    ,
    element="step",
    stat="density",
)
# ax.set_yscale('log')
ax.grid()
plt.savefig(f"{CITY.title()}_Uber_histogram.pdf")

# ## Inspect extreme cases (speed_diff > 10)

df_cell_mapping = pd.read_parquet(TBASEPATH / "road_graph" / CITY / "road_graph_intersecting_cells.parquet")
df_cell_mapping

# +
cols = [
    "year",
    "month",
    "day",
    "hour",
    "speed_diff",
    "std_speed_kph",
    "median_speed_kph",
    "speed_kph_mean",
    "free_flow_kph",
    "speed_kph",
    "length_meters",
    "x_u",
    "y_u",
    "x_v",
    "y_v",
]

extreme_edges = (
    ut_merged[ut_merged["speed_diff"] > 10]
    .groupby(["u", "v", "gkey"])
    .agg(**{col: (col, "mean") for col in cols}, count=("year", "count"), name=("name", "first"))
    .reset_index()[cols + ["name", "count", "u", "v", "gkey"]]
    .sort_values("speed_diff", ascending=False)
)
extreme_edges = extreme_edges[extreme_edges["count"] > 1]
extreme_edges


# +
def show_marker_on_map(lat, lon, lat2, lon2):
    pt = (lat, lon)
    pt2 = (lat2, lon2)
    bb = [(lat - 0.001, lon - 0.001), (lat + 0.001, lon + 0.001)]
    f = folium.Figure(width=930, height=300)
    m = folium.Map().add_to(f)
    folium.Marker(pt).add_to(m)
    folium.Marker(pt2).add_to(m)
    # folium.PolyLine(line, weight=5, opacity=1).add_to(m)
    m.fit_bounds(bb)
    return m


def visualize_extreme_case(i, extreme_edges=extreme_edges):
    item = extreme_edges.iloc[i]
    print(item)
    print("intersecting_cells")
    ic = df_cell_mapping[(df_cell_mapping["gkey"] == item["gkey"])]
    assert len(ic) == 1
    ic = ic.iloc[0]
    for ic in ast.literal_eval(ic["intersecting_cells"]):
        print(f"  {ic}")
    fig, ax = plt.subplots(1, tight_layout=True, figsize=(20, 10))
    data = ut_merged[(ut_merged["u"] == item["u"]) & (ut_merged["v"] == item["v"]) & (ut_merged["gkey"] == item["gkey"])]
    ax.plot(range(len(data)), data["median_speed_kph"], label="median_speed_kph", color="red")
    ax.plot(range(len(data)), data["speed_kph_mean"], label="speed_kph_mean", color="orange")
    ax.plot(range(len(data)), data["std_speed_kph"], label="std_speed_kph", color="green")
    ax.plot(range(len(data)), data["speed_mph_stddev"], label="speed_mph_stddev", color="yellow")
    ax.axhline(item["free_flow_kph"], label="free_flow_kph", color="black")
    ax.axhline(item["speed_kph"], label="speed_kph", color="grey", linestyle="dashed")
    ax.bar(range(len(data)), data["volume"], color="blue", label="volume")
    ax.legend()
    return show_marker_on_map(item["y_u"], item["x_u"], item["y_v"], item["x_v"])


# -

visualize_extreme_case(0)

visualize_extreme_case(55)

visualize_extreme_case(105)

visualize_extreme_case(3)

visualize_extreme_case(4)

visualize_extreme_case(5)

visualize_extreme_case(15)

# ## Inspect extreme cases (speed_diff < -10 and median_speed_kph < 15)


# +
cols = [
    "year",
    "month",
    "day",
    "hour",
    "speed_diff",
    "std_speed_kph",
    "median_speed_kph",
    "speed_kph_mean",
    "free_flow_kph",
    "speed_kph",
    "length_meters",
    "x_u",
    "y_u",
    "x_v",
    "y_v",
]

extreme_edges_low = (
    ut_merged[(ut_merged["speed_diff"] < -10) & (ut_merged["median_speed_kph"] < 15)]
    .groupby(["u", "v", "gkey"])
    .agg(**{col: (col, "mean") for col in cols}, count=("year", "count"), name=("name", "first"), highway=("highway", "first"))
    .reset_index()[cols + ["name", "count", "u", "v", "gkey", "highway"]]
    .sort_values("speed_diff", ascending=True)
)
extreme_edges_low = extreme_edges_low[extreme_edges_low["count"] > 1]
extreme_edges_low
# -

visualize_extreme_case(0, extreme_edges_low)

visualize_extreme_case(1, extreme_edges_low)

visualize_extreme_case(2, extreme_edges_low)

visualize_extreme_case(55, extreme_edges_low)

visualize_extreme_case(157, extreme_edges_low)

visualize_extreme_case(2000, extreme_edges_low)

visualize_extreme_case(5000, extreme_edges_low)

visualize_extreme_case(9999, extreme_edges_low)

visualize_extreme_case(12000, extreme_edges_low)

extreme_edges_low.groupby("highway").count()

extreme_edges_low.groupby("highway").median()

extreme_edges_low.groupby("highway").max()

extreme_edges_low[extreme_edges_low["length_meters"] > 150]
