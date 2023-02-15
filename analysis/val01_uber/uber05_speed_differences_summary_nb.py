# -*- coding: utf-8 -*-
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
from pathlib import Path

import humanize
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

plt.rcParams["axes.labelsize"] = 24

simplified_filter = lambda hw: hw not in ["residential", "living_street", "service", "road"]


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
    df["year"] = 2019
    df["month"] = 10
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
ax.tick_params(axis="x", which="major", labelsize=24, rotation=45)
ax.tick_params(axis="y", which="major", labelsize=24)
#  t4c: higher stds in link classes than in the corresponding road class they link -> plausibily, they get probes from the higher class running parallel.
plt.savefig(f"{CITY.title()}_t4c_std_speed_kph.png")

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
ax.tick_params(axis="x", which="major", labelsize=24, rotation=45)
ax.tick_params(axis="y", which="major", labelsize=24)
# t4c: higher volumes on links than on the class below -> plausibily, they get probes from the higher class running parallel.
plt.savefig(f"{CITY.title()}_t4c_volume.png")

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
ax.tick_params(axis="x", which="major", labelsize=24, rotation=45)
ax.tick_params(axis="y", which="major", labelsize=24)
# Uber: link class have higher std than the class they link to (apart from motorway) -> acceleration/deceleration on links plausible
plt.savefig(f"{CITY.title()}_Uber_speed_mph_stddev.png")

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
ax.tick_params(axis="x", which="major", labelsize=24, rotation=45)
ax.tick_params(axis="y", which="major", labelsize=24)
plt.savefig(f"{CITY.title()}_Uber_speed_kph_mean.png")

# +
fig, ax = plt.subplots(1, figsize=(20, 10), tight_layout=True)
plt.rcParams["axes.labelsize"] = 24
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

ax.tick_params(axis="x", which="major", labelsize=24, rotation=45)
ax.tick_params(axis="y", which="major", labelsize=24)
ax.grid(axis="y")
plt.savefig(f"{CITY.title()}_Uber_speed_diff.png")
# -


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
