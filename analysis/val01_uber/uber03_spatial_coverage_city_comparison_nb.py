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

import folium
import pandas
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# TODO whole overlapping period?

# U for Uber, T for T4c, O for OpenStreetMap
UBASEPATH = Path("/iarai/public/t4c/uber")
TBASEPATH = Path("/iarai/public/t4c/data_pipeline/release20221028_historic_uber")
OBASEPATH = Path("/iarai/public/t4c/osm")
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

plt.rcParams["axes.labelsize"] = 18

simplified_filter = lambda hw: hw not in ["residential", "living_street", "service", "road"]


def osm_color_palette():
    for c in ["#e892a2", "#e892a2", "#f9b29c", "#f9b29c", "#fcd6a4", "#fcd6a4", "#f7fabf", "#f7fabf"] + ["white"] * 99:
        yield c


df_barcelona = pd.read_parquet(f"barcelona_utcounts.parquet")
df_barcelona

df_berlin = pd.read_parquet(f"berlin_utcounts.parquet")
df_berlin

df_london = pd.read_parquet(f"london_utcounts.parquet")
df_london

df_london.columns

nl = "\\\\ \\arrayrulecolor{Grey0!60!RoyalBlue3}\\midrule[0.05pt]"

print(f"num edges & {len(df_barcelona)} & {len(df_berlin)} & {len(df_london)} \\\\")
print(nl)
for lb, ub in [(-1, -0.6), (-0.6, -0.2), (-0.2, 0.2), (0.2, 0.6), (0.6, 1.0)]:
    s = f"edge coverage ({lb},{ub}]"
    for city, df in [("barcelona", df_barcelona), ("berlin", df_berlin), ("london", df_london)]:
        l = len(df[(df["density_diff"] > lb) & (df["density_diff"] <= ub)])
        s += f" & {l} ({l/len(df)*100:.2f}\%)"
    print(s)
    print(nl)
print("\\\\ \\arrayrulecolor{black} \midrule")
print(
    f"num edges bb & {len(df_barcelona[df_barcelona['in_bb']==True])} & {len(df_berlin[df_berlin['in_bb']==True])} & {len(df_london[df_london['in_bb']==True])}"
)
print(nl)
for lb, ub in [(-1, -0.6), (-0.6, -0.2), (-0.2, 0.2), (0.2, 0.6), (0.6, 1.0)]:
    s = f"edge coverage bb ({lb},{ub}]"
    for city, df in [("barcelona", df_barcelona), ("berlin", df_berlin), ("london", df_london)]:
        df = df[df["in_bb"] == True]
        l = len(df[(df["density_diff"] > lb) & (df["density_diff"] <= ub)])
        s += f" & {l} ({l/len(df)*100:.2f}\%)"
    print(s)
    print(nl)

df_edges_barcelona = pd.read_parquet("/iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/road_graph/london/road_graph_edges.parquet")
df_edges_london = pd.read_parquet("/iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/road_graph/london/road_graph_edges.parquet")
df_edges_berlin = pd.read_parquet("/iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/road_graph/berlin/road_graph_edges.parquet")
len(df_edges_london)
print(f"num edges mcswts & {len(df_edges_barcelona)} & {len(df_edges_berlin)} & {len(df_edges_london)}")
print(nl)


# +
df_barcelona["city"] = "Barcelona"
df_berlin["city"] = "Berlin"
df_london["city"] = "London"

df_all = pd.concat([df_barcelona, df_berlin, df_london])
# -

df_london.groupby("highway").count()

# +
plt.rcParams["axes.labelsize"] = 24
fig, ax = plt.subplots(1, tight_layout=True, figsize=(10, 6), sharex=True)

data_split_t = df_all.copy()
data_split_u = df_all.copy()
data_split_u["density"] = data_split_u["density_u"]
data_split_u["src"] = "Uber"
data_split_t["density"] = data_split_t["density_t"]
data_split_t["src"] = "MeTS-10"
data_split = pd.concat([data_split_t, data_split_u])

sns.violinplot(
    data=data_split,
    y="density",
    x="city",
    hue="src",
    split=True,
    inner="quart",
    linewidth=1,
)

ax.set_ylim([0.0, 1.0])
ax.set(xlabel="")
ax.tick_params(axis="x", which="major", labelsize=24, rotation=0)
ax.tick_params(axis="y", which="major", labelsize=24)
ax.grid()
plt.legend(fontsize=18)
plt.savefig(f"uber03_spatial_coverage_city_comparison_no_bb_density_u_density_t.pdf")

# +
fig, ax = plt.subplots(1, tight_layout=True, figsize=(10, 6), sharex=True)

data_in_bb = df_all[df_all["in_bb"] == True]
data_in_bb_split_t = data_in_bb.copy()
data_in_bb_split_u = data_in_bb.copy()
data_in_bb_split_u["density"] = data_in_bb_split_u["density_u"]
data_in_bb_split_u["src"] = "Uber"
data_in_bb_split_t["density"] = data_in_bb_split_t["density_t"]
data_in_bb_split_t["src"] = "MeTS-10"
data_in_bb_split = pd.concat([data_in_bb_split_t, data_in_bb_split_u])
print(data_in_bb_split.columns)

sns.violinplot(
    data=data_in_bb_split,
    y="density",
    x="city",
    hue="src",
    split=True,
    inner="quart",
    linewidth=1,
)

ax.set_ylim([0.0, 1.0])
ax.set(xlabel="")
ax.tick_params(axis="x", which="major", labelsize=24, rotation=0)
ax.tick_params(axis="y", which="major", labelsize=24)
ax.grid()
plt.legend(fontsize=18)
plt.savefig(f"uber03_spatial_coverage_city_comparison_bb_density_u_density_t.pdf")
# -

fig, ax = plt.subplots(1, tight_layout=True, figsize=(10, 6))
# sns.histplot(data=df_all[df_all["in_bb"]==True],x="density_diff", hue="city", ax=ax, element="step",
#    stat="density", common_norm=True, )
# ax.set_yscale('log')
sns.violinplot(
    data=df_all[df_all["in_bb"] == True],
    y="density_diff",
    x="city",
    ax=ax,
    # element="step",    stat="density", common_norm=True,
)
ax.grid()
ax.set_ylim([-1.0, 1.0])
ax.tick_params(axis="x", which="major", labelsize=18, rotation=0)
ax.tick_params(axis="y", which="major", labelsize=18)
ax.set(ylabel="density diff (MeTS-10 - Uber)", xlabel="")
# plt.savefig(f"uber03_spatial_coverage_city_comparison_bb.pdf")

fig, ax = plt.subplots(1, tight_layout=True, figsize=(10, 6))
# sns.histplot(data=df_all[df_all["in_bb"]==True],x="density_diff", hue="city", ax=ax, element="step",
#    stat="density", common_norm=True, )
# ax.set_yscale('log')
sns.violinplot(
    data=df_all,
    y="density_diff",
    x="city",
    ax=ax,
    # element="step",    stat="density", common_norm=True,
)
ax.grid()
ax.set_ylim([-1.0, 1.0])
ax.tick_params(axis="x", which="major", labelsize=18, rotation=0)
ax.tick_params(axis="y", which="major", labelsize=18)
ax.set(ylabel="density diff (MeTS-10 - Uber)", xlabel="")
# plt.savefig(f"uber03_spatial_coverage_city_comparison_no_bb.pdf")

fig, ax = plt.subplots(1, tight_layout=True, figsize=(20, 10))
sns.kdeplot(
    data=df_all[df_all["in_bb"] == True],
    x="density_t",
    y="density_u",
    ax=ax,
    hue="city",
    common_norm=False
    # element="step",    stat="density", common_norm=True,
)
ax.grid()
# plt.savefig(f"uber03_spatial_coverage_city_comparison_bb.pdf")

fig, ax = plt.subplots(1, tight_layout=True, figsize=(20, 10))
sns.kdeplot(
    data=df_barcelona[df_barcelona["in_bb"] == True],
    x="density_t",
    y="density_u",
    ax=ax,
    # element="step",    stat="density", common_norm=True,
)
ax.grid()
# plt.savefig(f"uber03_spatial_coverage_city_comparison_bb.pdf")

fig, ax = plt.subplots(1, tight_layout=True, figsize=(20, 10))
sns.kdeplot(data=df_london[df_london["in_bb"] == True], x="density_t", y="density_u", ax=ax, fill=True)
ax.grid()
# plt.savefig(f"uber03_spatial_coverage_city_comparison_bb.pdf")

fig, ax = plt.subplots(1, tight_layout=True, figsize=(20, 10))
sns.kdeplot(
    data=df_berlin[df_berlin["in_bb"] == True],
    x="density_t",
    y="density_u",
    ax=ax,
    # element="step",    stat="density", common_norm=True,
)
ax.grid()
# plt.savefig(f"uber03_spatial_coverage_city_comparison_bb.pdf")

# ## Closer Look at London...

fig, ax = plt.subplots(1, tight_layout=True, figsize=(20, 10))
sns.histplot(
    data=df_london[df_london["in_bb"] == True],
    x="density_t",
    ax=ax,
    # element="step",    stat="density", common_norm=True,
)
ax.grid()
# ax.set_ylim([-1.0,1.0])
# plt.savefig(f"uber03_spatial_coverage_city_comparison_london.pdf")

fig, ax = plt.subplots(1, tight_layout=True, figsize=(20, 10))
sns.histplot(
    data=df_london[df_london["in_bb"] == True],
    x="density_u",
    ax=ax,
    # element="step",    stat="density", common_norm=True,
)
ax.grid()
# ax.set_ylim([-1.0,1.0])
# plt.savefig(f"uber03_spatial_coverage_city_comparison_london.pdf")

fig, ax = plt.subplots(1, tight_layout=True, figsize=(20, 10))
sns.violinplot(
    data=df_london[df_london["in_bb"] == True],
    y="density_diff",
    x="highway",
    ax=ax,
    # element="step",    stat="density", common_norm=True,
)
ax.grid()
ax.set_ylim([-1.0, 1.0])
plt.savefig(f"uber03_spatial_coverage_city_comparison_london.pdf")

df_london_outliers = df_london[(df_london["density_diff"] < -0.25) & (df_london["in_bb"] == True)]
df_london_outliers

df_london_outliers.groupby("highway").mean()[["length_meters"]]

df_london_outliers.groupby("highway").median()[["length_meters"]]

sns.histplot(df_london[df_london["highway"] == "primary"][["length_meters"]])

df_london[df_london["highway"] == "primary"][["length_meters"]].mean()

df_london[df_london["highway"] == "primary"][["length_meters"]].median()

sns.histplot(df_london_outliers[df_london_outliers["highway"] == "primary"][["length_meters"]])

df_london_outliers[df_london_outliers["highway"] == "primary"][["length_meters"]].mean()

df_london_outliers[df_london_outliers["highway"] == "primary"][["length_meters"]].median()

df_london_outliers.groupby("highway").count()


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


def visualize_extreme_case(i):
    item = df_london_outliers[df_london_outliers["highway"] == "primary"].iloc[i]
    print(item)
    return show_marker_on_map(item["y_u"], item["x_u"], item["y_v"], item["x_v"])


# -

visualize_extreme_case(0)

visualize_extreme_case(1)

visualize_extreme_case(5)

visualize_extreme_case(55)
