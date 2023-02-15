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

import geopandas
import geopandas as gpd
import pandas
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# TODO whole overlapping period?

# U for Uber, T for T4c, O for OpenStreetMap
UBASEPATH = Path("/iarai/public/t4c/uber")
TBASEPATH = Path("/iarai/public/t4c/data_pipeline/release20221028_historic_uber")
OBASEPATH = Path("/iarai/public/t4c/osm")
CITY = "berlin"
DAYTIME_START_HOUR = 8
DAYTIME_END_HOUR = 18
DAYTIME_HOURS = DAYTIME_END_HOUR - DAYTIME_START_HOUR
# -

gen_gpkg = True

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


# ## Load Uber segment speeds

uspeeds_df = pandas.read_parquet(UBASEPATH / "movement-speeds-hourly-berlin-2019-2.parquet")
# uspeeds_df['speed_kph_mean'] = uspeeds_df['speed_mph_mean'] * 1.60934
uspeeds_df = uspeeds_df.rename(columns={"osm_start_node_id": "u", "osm_end_node_id": "v", "osm_way_id": "osmid"})
uspeeds_df["gkey"] = [f"{u}_{v}_{osmid}" for u, v, osmid in zip(uspeeds_df["u"], uspeeds_df["v"], uspeeds_df["osmid"])]
uspeeds_df

# ## Load our segment speeds

tspeeds_l = []
for i in range(1, 8):
    f = TBASEPATH / "speed_classes" / CITY / f"speed_classes_2019-02-{i:02d}.parquet"
    print(f)
    df = pd.read_parquet(f)
    df["date"] = df["day"]
    df["year"] = 2019
    df["month"] = 2
    df["day"] = i
    df["hour"] = df["t"] // 4
    df = (
        df[["u", "v", "gkey", "year", "month", "day", "hour", "volume_class", "volume", "median_speed_kph", "free_flow_kph"]]
        .groupby(by=["u", "v", "gkey", "year", "month", "day", "hour"])
        .mean()
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

assert t_ids.issubset(u_ids)

# +
# for u,v in u_uv - t_uv:
#     print(f"https://www.openstreetmap.org/node/{u}")
#     print(f"https://www.openstreetmap.org/node/{v}")
#     break
# -

# ## Restrict to daytime only

tspeeds_df = tspeeds_df[(tspeeds_df["hour"] >= DAYTIME_START_HOUR) & (tspeeds_df["hour"] < DAYTIME_END_HOUR)]
tspeeds_df

uspeeds_df = uspeeds_df[(uspeeds_df["hour"] >= DAYTIME_START_HOUR) & (uspeeds_df["hour"] < DAYTIME_END_HOUR)]
uspeeds_df

# ## Counts/densityfor both

ucounts = uspeeds_df.groupby(["u", "v", "gkey"]).agg(count=("speed_kph_mean", "count"), speed_kph_mean=("speed_kph_mean", "first")).reset_index()
ucounts

tcounts = (
    tspeeds_df.groupby(["u", "v", "gkey"])
    .agg(count=("median_speed_kph", "count"), median_speed_kph=("median_speed_kph", "first"), mean_volume=("volume", "mean"))
    .reset_index()[["u", "v", "gkey", "count", "median_speed_kph", "mean_volume"]]
)
tcounts


num_slots_t_speeds = len(set(zip(tspeeds_df["year"], tspeeds_df["month"], tspeeds_df["day"], tspeeds_df["hour"])))
num_slots_t_speeds

num_slots_t_speeds / DAYTIME_HOURS

assert num_slots_t_speeds % DAYTIME_HOURS == 0

tcounts["density"] = tcounts["count"] / num_slots_t_speeds
tcounts

len(tcounts)

(tcounts["count"] == 0).sum()

num_slots_u_speeds = len(set(zip(uspeeds_df["year"], uspeeds_df["month"], uspeeds_df["day"], uspeeds_df["hour"])))
num_slots_u_speeds

num_slots_u_speeds / DAYTIME_HOURS

assert num_slots_u_speeds % DAYTIME_HOURS == 0

ucounts["density"] = ucounts["count"] / num_slots_u_speeds
ucounts


len(ucounts)

(ucounts["count"] == 0).sum()

# ## Merge with historic OSM data for UBER

gdf_nodes = gpd.read_parquet(TBASEPATH / "road_graph" / CITY / "road_graph_nodes.parquet")
gdf_nodes

gdf_edges = gpd.read_parquet(TBASEPATH / "road_graph" / CITY / "road_graph_edges.parquet")
gdf_edges

gdf_edges = gdf_edges.merge(gdf_nodes, left_on="u", right_on="node_id", suffixes=["", "_u"]).merge(
    gdf_nodes, left_on="v", right_on="node_id", suffixes=["", "_v"]
)
gdf_edges.rename(columns={"x": "x_u", "y": "y_u"}, inplace=True)
del gdf_edges["geometry_u"]
del gdf_edges["geometry_v"]
gdf_edges

# Berlin
y_min, y_max, x_min, x_max = 52.35900, 52.85400, 13.18900, 13.62500


def in_bb(x, y):
    return (x_min <= x <= x_max) and (y_min <= y <= y_max)


gdf_edges["in_bb"] = [
    (in_bb(x_u, y_u) or in_bb(x_v, y_v)) for x_u, y_u, x_v, y_v in zip(gdf_edges["x_u"], gdf_edges["y_u"], gdf_edges["x_v"], gdf_edges["y_v"])
]

gdf_edges.groupby("in_bb").agg(count=("gkey", "count"))


gdf_ucounts_merged = gdf_edges.merge(ucounts, on=["u", "v", "gkey"], how="left")
gdf_ucounts_merged[["count", "density"]] = gdf_ucounts_merged[["count", "density"]].fillna(0.0)
gdf_ucounts_merged

assert type(gdf_ucounts_merged) == geopandas.geodataframe.GeoDataFrame

gdf_tcounts_merged = gdf_edges.merge(tcounts, on=["u", "v", "gkey"], how="left")
gdf_tcounts_merged[["count", "density"]] = gdf_tcounts_merged[["count", "density"]].fillna(0.0)
gdf_tcounts_merged

# ### Write counts/density to parquet/gpkg

if gen_gpkg:
    gdf_ucounts_merged.to_parquet(f"{CITY}_ucounts.parquet")
    gdf_ucounts_merged.to_file(f"{CITY}_ucounts.gpkg", driver="GPKG", layer="edges")

if gen_gpkg:
    gdf_tcounts_merged.to_parquet(f"{CITY}_tcounts.parquet")
    gdf_tcounts_merged.to_file(f"{CITY}_tcounts.gpkg", driver="GPKG", layer="edges")

# ## Density differences

gdf_ut_counts_merged = gdf_tcounts_merged.merge(gdf_ucounts_merged, on=["u", "v", "gkey"], suffixes=["", "_u"], how="outer")
# workaround (if we use suffixes =["_t", "_u"]), we get a plain pandas frame...
gdf_ut_counts_merged.rename(columns={"density": "density_t", "count": "count_t"}, inplace=True)
gdf_ut_counts_merged[["count_t", "density_t"]] = gdf_ut_counts_merged[["count_t", "density_t"]].fillna(0.0)
gdf_ut_counts_merged["density_diff"] = gdf_ut_counts_merged["density_t"] - gdf_ut_counts_merged["density_u"]
del gdf_ut_counts_merged["geometry_u"]
gdf_ut_counts_merged

gdf_ut_counts_merged["in_bb"] = [
    (in_bb(x_u, y_u) or in_bb(x_v, y_v))
    for x_u, y_u, x_v, y_v in zip(gdf_ut_counts_merged["x_u"], gdf_ut_counts_merged["y_u"], gdf_ut_counts_merged["x_v"], gdf_ut_counts_merged["y_v"])
]

gdf_ut_counts_merged["sort_key"] = [highway_ordering.index(hw) for hw in gdf_ut_counts_merged["highway"]]
gdf_ut_counts_merged = gdf_ut_counts_merged.sort_values("sort_key")

assert gdf_ut_counts_merged["density_u"].isnull().sum() == 0
assert gdf_ut_counts_merged["density_t"].isnull().sum() == 0
assert type(gdf_ut_counts_merged) == geopandas.geodataframe.GeoDataFrame

if gen_gpkg:
    gdf_ut_counts_merged.to_parquet(f"{CITY}_utcounts.parquet")
    gdf_ut_counts_merged.to_file(f"{CITY}_utcounts.gpkg", driver="GPKG", layer="edges")

gdf_ut_counts_merged_by_hw_in_bb = (
    gdf_ut_counts_merged[gdf_ut_counts_merged["in_bb"] == True]
    .groupby("highway")
    .agg(count=("density_diff", "count"), mean_density_diff=("density_diff", "mean"))
    .reset_index()
)
gdf_ut_counts_merged_by_hw_in_bb["sort_key"] = [highway_ordering.index(hw) for hw in gdf_ut_counts_merged_by_hw_in_bb["highway"]]
gdf_ut_counts_merged_by_hw_in_bb = gdf_ut_counts_merged_by_hw_in_bb.sort_values("sort_key")
gdf_ut_counts_merged_by_hw_in_bb

# +
plt.rcParams["axes.labelsize"] = 32
fig, ax = plt.subplots(1, figsize=(20, 10), tight_layout=True)

sns.boxplot(
    gdf_ut_counts_merged[(gdf_ut_counts_merged["in_bb"] == True) & ([simplified_filter(hw) for hw in gdf_ut_counts_merged["highway"]])],
    x="highway",
    y="density_diff",
    notch=True,
    sym="",
    palette=osm_color_palette(),
    medianprops={"color": "coral"},
    ax=ax,
)
# plt.rcParams["axes.labelsize"] = 24
ax.tick_params(axis="x", which="major", labelsize=24, rotation=45)
ax.tick_params(axis="y", which="major", labelsize=24)
ax.tick_params(axis="x", which="major", labelsize=32, rotation=45)
ax.tick_params(axis="y", which="major", labelsize=32)
ax.grid(axis="y")
ax.set(xlabel="road type (highway)", ylabel="density diff (MeTS-10 - Uber)    ")
plt.savefig(f"{CITY}_Uber_density_diff_barplot.pdf")

# +
# TODO do we need full area at all?
# -

# ## Density histograms and stats


# instead of un/pivoting.....
ucounts["src"] = "u"
tcounts["src"] = "t"
counts_all = pd.concat([ucounts.rename(columns={"speed_kph_mean": "speed"}), tcounts.rename(columns={"median_speed_kph": "speed"})])
counts_all = counts_all.merge(gdf_edges, on=["u", "v", "gkey"])
counts_all["sort_key"] = [highway_ordering.index(hw) for hw in counts_all["highway"]]
counts_all = counts_all.sort_values("sort_key")
counts_all["matching"] = [(u, v, gkey) in t_ids for u, v, gkey in zip(counts_all["u"], counts_all["v"], counts_all["gkey"])]


# ## Density histogram for both


def plot_and_stats(df, density_attr, speed_attr):
    fig, axs = plt.subplots(2, figsize=(10, 5), tight_layout=True)
    axs[0].hist(df[density_attr], bins=10)
    axs[0].set_xlabel(density_attr)
    axs[1].hist(df[speed_attr], bins=48)
    axs[1].set_xlabel(speed_attr)
    df_gr = df.groupby("highway").agg(
        {
            density_attr: ["count", "mean", "median", "min", "max", "std"],
            speed_attr: ["mean", "median", "min", "max", "std"],
            "speed_kph": ["mean", "median", "min", "max", "std"],
            "length_meters": ["mean", "median", "min", "max", "std"],
        }
    )
    df_gr["sort_key"] = [highway_ordering.index(hw) for hw in df_gr.reset_index()["highway"]]
    display(df_gr.sort_values("sort_key"))


# ### density over full area


fig, ax = plt.subplots(1, figsize=(20, 10), tight_layout=True)
sns.boxplot(counts_all, x="highway", y="density", hue="src", ax=ax)
# TODO this looks fishy: motorway over full areay t should have 0 for everything ourside of the box - outer join?!

# #### full area Uber

plot_and_stats(gdf_ut_counts_merged, "density_u", "speed_kph_mean")

# #### density full area t4c

plot_and_stats(gdf_ut_counts_merged, "density_t", "median_speed_kph")

# ### density within bounding box

fig, ax = plt.subplots(1, figsize=(20, 10), tight_layout=True)
sns.boxplot(counts_all[counts_all["in_bb"] == True], x="highway", y="density", hue="src", ax=ax)

# #### density bounding box Uber

plot_and_stats(gdf_ut_counts_merged[gdf_ut_counts_merged["in_bb"] == True], "density_u", "speed_kph_mean")

# #### density bounding box t4c

plot_and_stats(gdf_ut_counts_merged[gdf_ut_counts_merged["in_bb"] == True], "density_t", "median_speed_kph")


# ### Density and speeds for matched edges

# +
# more fishy...
# -

counts_all[(counts_all["matching"] == True) & ((counts_all["src"] == "t"))]

counts_all[(counts_all["matching"] == True) & ((counts_all["src"] == "u"))]

fig, ax = plt.subplots(1, figsize=(20, 10), tight_layout=True)
sns.boxplot(counts_all[counts_all["matching"] == True], x="highway", y="density", hue="src", ax=ax)

fig, ax = plt.subplots(1, figsize=(20, 10), tight_layout=True)
sns.boxplot(counts_all[counts_all["matching"] == True], x="highway", y="speed", hue="src", ax=ax)

# TODO check after daytime only
assert len(counts_all[counts_all["src"] == "t"]) == len(counts_all[(counts_all["src"] == "u") & (counts_all["matching"] == True)])

plot_and_stats(counts_all[(counts_all["src"] == "u") & (counts_all["matching"] == True)], "density", "speed")

plot_and_stats(counts_all[counts_all["src"] == "t"], "density", "speed")

# +
# TODO plot distribution of differences
