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
import pandas
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt

UBASEPATH = Path('/iarai/public/t4c/uber')
TBASEPATH = Path('/iarai/public/t4c/data_pipeline/release20221003_keep_all_edges/')
# TBASEPATH = Path('/iarai/public/t4c/data_pipeline/release20220930/')

# -

# uspeeds_df = pandas.read_csv(UBASEPATH / 'movement-speeds-hourly-madrid-2020-3.csv')
# uspeeds_df = pandas.read_csv(UBASEPATH / 'movement-speeds-hourly-london-2019-10.csv')
uspeeds_df = pandas.read_parquet(UBASEPATH / 'movement-speeds-hourly-london-2019-10.parquet')
uspeeds_df['speed_kph_mean'] = uspeeds_df['speed_mph_mean'] * 1.60934
uspeeds_df = uspeeds_df.rename(columns={'osm_start_node_id': 'u', 'osm_end_node_id': 'v'})
uspeeds_df

# +
# uspeeds_df = uspeeds_df[[
#     'osm_start_node_id', 'osm_end_node_id', 'osm_way_id', 'year', 'month', 'day', 'hour',
#     'speed_mph_mean', 'speed_mph_stddev']]
# uspeeds_df.to_parquet(UBASEPATH / 'movement-speeds-hourly-london-2019-10.parquet', compression='snappy')
# uspeeds_df
# -

uedges_df = uspeeds_df[['u', 'v', 'osm_way_id']].groupby(
    by=['u', 'v']).count().reset_index()
uedges_df = uedges_df.rename(columns={'osm_way_id': 'cnt'})
uedges_df

uedges_df['cnt'].hist()

# tedges_df = pandas.read_parquet(TBASEPATH / '2022' / 'road_graph' / 'madrid' / 'road_graph_edges.parquet')
tedges_df = pandas.read_parquet(TBASEPATH / '2022' / 'road_graph' / 'london' / 'road_graph_edges.parquet')
tedges_df

edges_merged_df = uedges_df.merge(tedges_df, on=['u', 'v'])
edges_merged_df

edges_merged_df['cnt'].hist(bins=800, figsize=(20,10))

uspeeds_day_df = uspeeds_df[(uspeeds_df['year'] == 2019) & (uspeeds_df['month'] == 10) & (uspeeds_df['day'] >= 21) & (uspeeds_df['day'] <= 27)]
uspeeds_day_df

uspeeds_day_df['hour'].hist()

tspeeds_l = []
for i in range(21,28):
    f = TBASEPATH / '2022' / 'speed_classes' / 'london' / f'speed_classes_2019-10-{i}.parquet'
    df = pandas.read_parquet(f)
    df["date"]=df["day"]
    df["day"]=i
    tspeeds_l.append(df)

tspeeds_df = pandas.concat(tspeeds_l)
tspeeds_df

# +
#tspeeds_df = pandas.read_parquet(TBASEPATH / '2022' / 'speed_classes' / 'london' / 'speed_classes_2019-10-07.parquet')
#tspeeds_df
# -

tspeeds_df['hour'] = tspeeds_df['t'] // 4
tspeeds_df

tspeeds_df['hour'].hist()

tspeeds_60min_df = tspeeds_df[['u', 'v', 'day', 'hour', 'volume_class', 'median_speed_kph', 'free_flow_kph']].groupby(
    by=['u', 'v', 'day','hour']).mean().reset_index()
tspeeds_60min_df

tspeeds_merged_df = uspeeds_day_df.merge(tspeeds_60min_df, on=['u', 'v', 'day','hour']).reset_index()
tspeeds_merged_df

sns.barplot(data=uspeeds_day_df, x="hour", y="speed_kph_mean")

sns.barplot(data=tspeeds_60min_df, x="hour", y="median_speed_kph")

sns.histplot(data=uspeeds_day_df, x="hour", bins=24)

sns.histplot(data=tspeeds_60min_df, x="hour", bins=24)

tspeeds_merged_df["factor"]=tspeeds_merged_df["speed_mph_stddev"]/tspeeds_merged_df["speed_mph_mean"]
tspeeds_merged_df

# +
fig, ax = plt.subplots(1, tight_layout=True, figsize=(10,10))
ax.plot([0, 130], [0, 130], ls="--", c=".3")
ax.set_xlim([0,130])
ax.set_ylim([0,130])
#tspeeds_merged_df[(tspeeds_merged_df["day"]==27)&(tspeeds_merged_df["median_speed_kph"]>60)&(tspeeds_merged_df["speed_kph_mean"]<50)].plot(kind='scatter', x='median_speed_kph', y='speed_kph_mean', figsize=(10,10), ax=ax)
tspeeds_merged_df[(tspeeds_merged_df["day"]==27)].plot(kind='scatter', x='median_speed_kph', y='speed_kph_mean', figsize=(10,10), ax=ax)

#sns.kdeplot(
#    data=tspeeds_merged_df[(tspeeds_merged_df["day"]==27)], x='median_speed_kph', y='speed_kph_mean', fill=True, ax=ax
#)
# -

import geopandas
#edges_df = geopandas.read_parquet(data_folder / "road_graph" / city / "road_graph_freeflow.parquet")
edges_gdf = geopandas.read_parquet(TBASEPATH / '2022' / 'road_graph' / 'london' / 'road_graph_edges.parquet')

bottom_right = edges_gdf.merge(tspeeds_merged_df[(tspeeds_merged_df["day"]==27)&(tspeeds_merged_df["median_speed_kph"]>60)&(tspeeds_merged_df["speed_kph_mean"]<50)], on=["u", "v"])
bottom_right

bottom_right["hour"].hist()

bottom_right.groupby("highway").agg(count=("u", "count")).sort_values("count", ascending=False)

# +
#bottom_right.to_file("bla.gkpg", driver="GPKG", layer="edges_ff")
# -

tspeeds_merged_df[(tspeeds_merged_df["day"]==27)&(tspeeds_merged_df["median_speed_kph"]>60)&(tspeeds_merged_df["speed_kph_mean"]<50)].groupby(["u", "v"]).count()["year"].hist()

tspeeds_merged_df["median_speed_kph"].mean()

tspeeds_merged_df["median_speed_kph"].median()

tspeeds_merged_df["median_speed_kph"].std()

tspeeds_merged_df["median_speed_kph"].hist(bins=120)

tspeeds_merged_df["speed_kph_mean"].mean()

tspeeds_merged_df["speed_kph_mean"].median()

tspeeds_merged_df["speed_kph_mean"].std()

fig, ax = plt.subplots(1, tight_layout=True, figsize=(10,10))
ax.set_xlim([0,125])
sns.histplot(tspeeds_merged_df, x="median_speed_kph", bins=120, ax=ax, cumulative=True, fill=False)
sns.histplot(tspeeds_merged_df, x="speed_kph_mean", bins=120, ax=ax, cumulative=True, fill=False)
ax.legend()

fig, ax = plt.subplots(1, tight_layout=True, figsize=(10,10))
ax.plot([0, 130], [0, 130], ls="--", c=".3")
ax.set_xlim([0,130])
ax.set_ylim([0,130])
#tspeeds_merged_df[(tspeeds_merged_df["day"]==27)&(tspeeds_merged_df["factor"]>.8)].plot(kind='scatter', x='median_speed_kph', y='speed_kph_mean', figsize=(10,10), ax=ax)
sns.kdeplot(
    data=tspeeds_merged_df[(tspeeds_merged_df["day"]==27)&(tspeeds_merged_df["factor"]>.8)], x='median_speed_kph', y='speed_kph_mean', fill=True, ax=ax
)

# +
import seaborn as sns
from matplotlib import pyplot as plt

#fig, ax = plt.subplots(1, tight_layout=True, figsize=(20,20))
#ax.set_xlim([0,130])
#ax.set_ylim([0,130])
#sns.kdeplot(
#    data=tspeeds_merged_df, x='median_speed_kph', y='speed_kph_mean', fill=True, ax=axs
#)

# -

day_counts = tspeeds_merged_df.groupby(["u", "v"]).agg(daycount=("hour", "count"), speed_kph_mean=("speed_kph_mean","mean"))
day_counts

day_counts.sort_values("speed_kph_mean", ascending=False)

day_counts[(day_counts["daycount"]>20)&(day_counts["daycount"]<100)].sort_values("speed_kph_mean", ascending=False)

tspeeds_merged_df[(tspeeds_merged_df["v"]==u)]

# +
# M25: 
#      https://www.openstreetmap.org/node/206224677#map=16/51.2582/-0.2024
#      https://www.openstreetmap.org/node/1639045045#map=15/51.2641/-0.1821
u=206224677
v=1639045045

# #??
#u=6573256890
#v=771703067

#u=6563498350
#v=6563498349

u=292158
v=292160
edge_df = tspeeds_merged_df[(tspeeds_merged_df["u"]==u)&(tspeeds_merged_df["v"]==v)].reset_index()
edge_df = edge_df.sort_values(["day","hour"])
edge_df
# -

fig, ax = plt.subplots(1, tight_layout=True, figsize=(10,10))
edge_df.plot(kind='scatter', x='median_speed_kph', y='speed_kph_mean', ax=ax)
ax.plot([0, 130], [0, 130], ls="--", c=".3")
ax.set_xlim([0,130])
ax.set_ylim([0,130])

edge_df["median_speed_kph"].mean()

edge_df["median_speed_kph"].std()

edge_df["speed_kph_mean"].mean()

edge_df["speed_kph_mean"].std()

fig, ax = plt.subplots(1, tight_layout=True, figsize=(10,10))
plt.plot(edge_df["day"]*24+edge_df["hour"], edge_df["median_speed_kph"], marker='o', label="median_speed_kph")
plt.plot(edge_df["day"]*24+edge_df["hour"], edge_df["speed_kph_mean"], color="orange", marker='o', label="speed_kph_mean")
plt.legend()
