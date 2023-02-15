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
#     display_name: Python [conda env:t4c] *
#     language: python
#     name: conda-env-t4c-py
# ---

# +
from IPython.display import display
from IPython.core.display import HTML
# %load_ext autoreload
# %load_ext time
# %autoreload 2
# %autosave 60
# %matplotlib inline

display(HTML("<style>.container { width:80% !important; }</style>"))

# +
import pandas as pd
import numpy as np
import folium
import geojson
import geopandas
import contextily as cx
import osmnx as ox
import networkx as nx
import seaborn as sns
import geopy.distance
from pathlib import Path
from pyproj import CRS, Transformer
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

from exploration_utils import plot_counter, plot_counters, plot_locations, inspect_diff_area, plot_scatters
from exploration_utils import plot_speed_dayline, plot_flow_dayline, plot_kde_scatter
from exploration_utils import MOVIE_BBOXES


DATA_PATH = Path('/private/data/mets10/loop_counters')
RELEASE_PATH = Path('/private/data/mets10/release20221026_residential_unclassified')
# -

speeds_merged_berlin = pd.read_parquet(DATA_PATH / 'berlin' / 'speeds_merged_clean_berlin.parquet')
print(len(speeds_merged_berlin))
speeds_merged_london = pd.read_parquet(DATA_PATH / 'london' / 'speeds_merged_clean_london.parquet')
print(len(speeds_merged_london))
speeds_merged_madrid = pd.read_parquet(DATA_PATH / 'madrid' / 'speeds_merged_clean_madrid.parquet')
print(len(speeds_merged_madrid))

# +
# _ = inspect_diff_area(speeds_merged_london, 0, 120, 0, 120, bbox_city='london')
# -
fig, axs = plt.subplots(1, 3, tight_layout=True, figsize=(24,8))
plot_locations(speeds_merged_berlin, bbox_city='berlin', ax=axs[0])
plot_locations(speeds_merged_london, bbox_city='london', ax=axs[1])
plot_locations(speeds_merged_madrid, bbox_city='madrid', ax=axs[2])
plt.show()

# +
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")

def plot_locations_map(df, city, ax=None):
    fig = None
    if not ax:
        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(8,8))
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    bbox = tuple([c / 1e5 for c in MOVIE_BBOXES[city.lower()]['bounds']])
    south, north, west, east = bbox
    lon_min, lat_min, lon_max, lat_max = transformer.transform_bounds(south, west, north, east)
    
    df = geopandas.GeoDataFrame(df.copy(), geometry=geopandas.points_from_xy(df.lon, df.lat))
    df = df.set_crs('EPSG:4326')
    df = df.to_crs(epsg=3857)
    if 'cnt' in df.columns:
        df.plot(column='cnt', markersize=30, cmap=plt.get_cmap("jet"), colorbar=True, alpha=0.4, ax=ax)
        plt.legend()
    else:
        df.plot(color="red", markersize=30, ax=ax)
    
    # Enlarge the lat bbox values to a span of 88426, this copes with the projection
    # size differences between the different latitudes. For visualization only.
    lat_mid = lat_min + ((lat_max - lat_min) / 2)
    lat_min = lat_mid - (88426 / 2)
    lat_max = lat_mid + (88426 / 2)
    
    ax.set_xlim([lon_min,lon_max])
    ax.set_ylim([lat_min,lat_max])
        
    cx.add_basemap(ax, source=cx.providers.Stamen.TonerLite, attribution_size=18)
    
    ax.title.set_text(city)
    ax.title.set_size(26)

    if fig:
        plt.show()


# plot_locations_map(speeds_merged_berlin, bbox_city='berlin')
# plot_locations_map(speeds_merged_london, bbox_city='london')
# plot_locations_map(speeds_merged_madrid, bbox_city='madrid')

fig, axs = plt.subplots(1, 3, tight_layout=False, figsize=(18,12))
fig.tight_layout(pad=0.0, w_pad=-1.5)
# plot_locations_map(speeds_merged_berlin, city='Berlin', ax=axs[0])
# plot_locations_map(speeds_merged_london, city='London', ax=axs[1])
# plot_locations_map(speeds_merged_madrid, city='Madrid', ax=axs[2])
plot_locations_map(speeds_merged_berlin[['lat', 'lon']].groupby(['lat', 'lon']).max().reset_index(), city='Berlin', ax=axs[0])
plot_locations_map(speeds_merged_london[['lat', 'lon']].groupby(['lat', 'lon']).max().reset_index(), city='London', ax=axs[1])
plot_locations_map(speeds_merged_madrid[['lat', 'lon']].groupby(['lat', 'lon']).max().reset_index(), city='Madrid', ax=axs[2])
plt.savefig('sxs_city_counter_map.pdf')

# attribution
# Map tiles by Stamen Design, CC BY 3.0 -- Map data (C) OpenStreetMap contributors
# -

fig, axs = plt.subplots(1, 3, tight_layout=False, figsize=(9,6))
fig.tight_layout(pad=0.0, w_pad=-3.0)
plot_locations_map(speeds_merged_berlin[['lat', 'lon']].groupby(['lat', 'lon']).max().reset_index(), city='Berlin', ax=axs[0])
plot_locations_map(speeds_merged_london[['lat', 'lon']].groupby(['lat', 'lon']).max().reset_index(), city='London', ax=axs[1])
plot_locations_map(speeds_merged_madrid[['lat', 'lon']].groupby(['lat', 'lon']).max().reset_index(), city='Madrid', ax=axs[2])
plt.savefig('sxs_city_counter_map.pdf')  


# +
def print_bbox_meters(city):
    bbox = tuple([c / 1e5 for c in MOVIE_BBOXES[city]['bounds']])
    south, north, west, east = bbox
    print(f'{city} ({east - west}, {north - south})')
#     print((north, west))
#     print((north, east))
    print(f'width: {geopy.distance.geodesic((north, west), (north, east)).km} km')
    print(f'height: {geopy.distance.geodesic((north, west), (south, west)).km} km\n')
    
# print_bbox_meters('berlin')
# print_bbox_meters('london')
# print_bbox_meters('madrid')


# -

fig, axs = plt.subplots(1, 3, tight_layout=True, figsize=(24,8))
plot_kde_scatter(speeds_merged_berlin, ax=axs[0], city='Berlin')
plot_kde_scatter(speeds_merged_london, ax=axs[1], city='London')
plot_kde_scatter(speeds_merged_madrid, ax=axs[2], city='Madrid')
plt.show()

fig, axs = plt.subplots(1, 3, tight_layout=True, figsize=(21,8))
plot_kde_scatter(speeds_merged_berlin, ax=axs[0], labelsize=25, titlesize=32, city='Berlin')
plot_kde_scatter(speeds_merged_london, ax=axs[1], labelsize=25, titlesize=32, city='London')
plot_kde_scatter(speeds_merged_madrid, ax=axs[2], labelsize=25, titlesize=32, city='Madrid')
axs[1].get_yaxis().set_visible(False)
axs[2].get_yaxis().set_visible(False)
plt.savefig("sxs_snsspeed.png")
plt.savefig("sxs_snsspeed.pdf")

speeds_merged_berlin['diff_perc'] = (speeds_merged_berlin['diff'] / speeds_merged_berlin['free_flow_kph']) * 100
speeds_merged_berlin['diff_perc'].clip(upper=100, inplace=True)  
speeds_merged_berlin[['diff', 'diff_perc']].hist(figsize=(14,5))
speeds_merged_london['diff_perc'] = (speeds_merged_london['diff'] / speeds_merged_london['free_flow_kph']) * 100
speeds_merged_london['diff_perc'].clip(upper=100, inplace=True)  
speeds_merged_london[['diff', 'diff_perc']].hist(figsize=(14,5))
speeds_merged_madrid['diff_perc'] = (speeds_merged_madrid['diff'] / speeds_merged_madrid['free_flow_kph']) * 100
speeds_merged_madrid['diff_perc'].clip(upper=100, inplace=True)  
speeds_merged_madrid[['diff', 'diff_perc']].hist(figsize=(14,5))

edges_berlin = geopandas.read_parquet(RELEASE_PATH / '2021' / 'road_graph' / 'berlin' / 'road_graph_edges.parquet')
speeds_merged_berlin = speeds_merged_berlin.merge(edges_berlin[['u', 'v', 'gkey', 'highway']], on=['u', 'v', 'gkey'])
print(len(speeds_merged_berlin))
# edges_london = geopandas.read_parquet(RELEASE_PATH / '2022' / 'road_graph' / 'london' / 'road_graph_edges.parquet')
# speeds_merged_london = speeds_merged_london.merge(edges_london[['u', 'v', 'gkey', 'highway']], on=['u', 'v', 'gkey'])
# print(len(speeds_merged_london))
edges_madrid = geopandas.read_parquet(RELEASE_PATH / '2022' / 'road_graph' / 'madrid' / 'road_graph_edges.parquet')
speeds_merged_madrid = speeds_merged_madrid.merge(edges_madrid[['u', 'v', 'gkey', 'highway']], on=['u', 'v', 'gkey'])
print(len(speeds_merged_madrid))

# speeds_merged_berlin['diff_rat'] = ((speeds_merged_berlin['speed_probes'] - speeds_merged_berlin['speed_counter']) / 120).astype(float)
speeds_merged_berlin.head()

# speeds_merged_london['diff_rat'] = ((speeds_merged_london['speed_probes'] - speeds_merged_london['speed_counter']) / 120).astype(float)
speeds_merged_london.head()

# speeds_merged_madrid['diff_rat'] = ((speeds_merged_madrid['speed_probes'] - speeds_merged_madrid['speed_counter']) / 120).astype(float)
speeds_merged_madrid.head()


# +
def osm_color_palette():
    for c in ["#e892a2", "#e892a2", "#f9b29c", "#f9b29c", "#fcd6a4", "#fcd6a4", "#f7fabf", "#f7fabf"] + ["white"] * 99:
        yield c

highway_ordering = [
    'motorway', 'motorway_link', 
    'trunk','trunk_link', 
    'primary','primary_link',
    'secondary', 'secondary_link', 
    'tertiary','tertiary_link',
    'unclassified',  'residential',
    'living_street', 'service',
    'cycleway', 'road', 'construction'
]
        
def diff_boxplot(data, ax=None, labelsize=24, city=''):
    data = data[~data['highway'].str.contains('_link')].copy()
    data.loc[data["highway"].str.contains('motorway'), "highway"] = "motorway"
    data.loc[data["highway"].str.contains('trunk'), "highway"] = "trunk"
    data.loc[data["highway"].str.contains('primary'), "highway"] = "primary"
    data["sort_key"] = [highway_ordering.index(hw) for hw in data["highway"]]
    data = data.sort_values("sort_key")
    data["diff_km"] = data['speed_probes'] - data['speed_counter']
    if not ax:
        fig, ax = plt.subplots(1, figsize=(20, 10), tight_layout=True)
    sns.boxplot(data=data, 
                x="highway",
                y="diff_km",  
                notch=True, 
                sym='',
                order=['motorway', 'trunk', 'primary', 'secondary', 'tertiary'],
                palette=osm_color_palette(),
                medianprops={"color": "coral"},
                ax=ax)
    ax.set(ylim=(-56, 56))
    ax.set_xlabel('road type (highway)', fontsize=labelsize)
    ax.set_ylabel('Speed diff (MeTS-10 - Detector) [km/h]', fontsize=labelsize)
    ax.tick_params(axis='x', which='major', labelsize=labelsize, rotation=45)
    ax.tick_params(axis='y', which='major', labelsize=labelsize)
    ax.grid(axis='y')
    ax.title.set_text(city)
    ax.title.set_size(labelsize+4)


diff_boxplot(speeds_merged_berlin, city='Berlin')
# plt.savefig("London_counter_density_diff_barplot.png")
# speeds_merged_berlin['diff_rat'].min()
# -

diff_boxplot(speeds_merged_london)

diff_boxplot(speeds_merged_madrid)

fig, axs = plt.subplots(1, 3, figsize=(20, 10), tight_layout=True)
diff_boxplot(speeds_merged_berlin, axs[0], labelsize=28, city='Berlin')
diff_boxplot(speeds_merged_london, axs[1], labelsize=28, city='London')
diff_boxplot(speeds_merged_madrid, axs[2], labelsize=28, city='Madrid')
axs[1].get_yaxis().set_ticklabels([])
axs[2].get_yaxis().set_ticklabels([])
axs[1].set_ylabel('')
axs[2].set_ylabel('')
plt.savefig("counter_diff_boxplot.png")
plt.savefig("counter_diff_boxplot.pdf")

speeds_merged_berlin['diff_km'] = speeds_merged_berlin['speed_probes'] - speeds_merged_berlin['speed_counter']
diff_g0_berlin = speeds_merged_berlin[speeds_merged_berlin['diff_km'] > 0]
diff_g0_berlin

diff_g0_berlin['highway'].hist()

diff_g0mwt_berlin = diff_g0_berlin[(diff_g0_berlin['highway'].str.contains('motorway')) |
                                   (diff_g0_berlin['highway'].str.contains('trunk'))]
diff_g0mwt_berlin

# +
for i in diff_g0mwt_berlin[['id', 'name', 'lat', 'lon']].groupby(['id', 'name', 'lat', 'lon']).count().reset_index().iterrows():
    print(f"{i[1][0]}: {i[1][2]}, {i[1][3]}")

diff_g0mwt_berlin['id'].unique()

# +
hwt_ids = [
    100101010000167,  # Spanische Allebrücke, 52.43386833, 13.19257786 --> change from speed limit 100 to 80!
    100101010000369, 100101010000470, # Spanische Allebrücke, 52.43381256, 13.1927467 --> change from speed limit 100 to 80!
    100101010004312, 100101010004413, 100101010004514, # Prenzlauer Promenade/Pankow, 52.57461381, 13.42952123
    100101010040785, 100101010040987]  # Prenzlauer Promenade/Pasewalker Strasse, 52.58194883, 13.42993792

diff_g0mwt_berlin[['id', 'v_kfz_det_hr', 'v_pkw_det_hr', 'speed_probes', 'diff_km']].groupby('id').agg(
    {'v_kfz_det_hr': ['mean', 'std'],
     'v_pkw_det_hr': ['mean', 'std'],
     'speed_probes': ['mean', 'std'],
     'diff_km': ['mean', 'std', 'count']})

# +
# Prenzlauer Promenade, , 52.58194883, 13.42993792
# -

# # Counter situation plots
#
# ![image.png](attachment:image.png)

speeds_merged_london_all = pd.read_parquet('speeds_merged_london.parquet')
edges_london = geopandas.read_parquet(DATA_PATH / 'road_graph' / 'london' / 'road_graph_freeflow.parquet')


# +
def plot_counter_marker(df, edf, id, ax=None, w=3):
    fig = None
    if not ax:
        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(9,2))
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    df = df[['id', 'lat', 'lon', 'u', 'v', 'gkey']].groupby(['id', 'lat', 'lon']).max().reset_index()
    df = df[df['id'] == id]
    assert len(df) == 1
    df = geopandas.GeoDataFrame(df.copy(), geometry=geopandas.points_from_xy(df.lon, df.lat))
    df = df.set_crs('EPSG:4326')
    
    edf = edf.merge(df[['id', 'u', 'v', 'gkey']], on=['u', 'v', 'gkey'])
    edf = edf[edf['id'] == id]
    edf = edf.set_crs('EPSG:4326')
    assert len(edf) == 1
    
    df = pd.concat([edf, df])
    df = df.to_crs(epsg=3857)
    
    df.plot(color=['blue', 'red'], marker='$\odot$', markersize=700, linewidth=4, ax=ax)
    
    lat = df.geometry.values[1].y
    lon = df.geometry.values[1].x
    s = 200
    ax.set_xlim([lon-w*s,lon+w*s])
    ax.set_ylim([lat-s,lat+s])
    
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, attribution_size=12, reset_extent=True)
    

plot_counter_marker(speeds_merged_london_all, edges_london, '6137')
# -

plot_counter_marker(speeds_merged_london_all, edges_london, '1737')

# London easy situation
plot_speed_dayline(speeds_merged_london_all, '6137', figsize=(9,3), labels={'speed_counter': 'Detector', 'speed_probes': 'MeTS-10'})

fig, axs = plt.subplots(2, 1, figsize=(9, 7), tight_layout=True)
plot_counter_marker(speeds_merged_london_all, edges_london, '6137', ax=axs[0])
plot_speed_dayline(speeds_merged_london_all, '6137', labels={'speed_counter': 'Detector', 'speed_probes': 'MeTS-10'},
                   labelsize=14, ax=axs[1])
plt.savefig("counter_situation_easy.pdf")

fig, axs = plt.subplots(2, 1, figsize=(9, 4.5), tight_layout=True)
plot_counter_marker(speeds_merged_london_all, edges_london, '6137', ax=axs[0], w=5)
plot_speed_dayline(speeds_merged_london_all, '6137', labels={'speed_counter': 'Detector', 'speed_probes': 'MeTS-10'},
                   labelsize=14, ax=axs[1])
plt.savefig("counter_situation_easy_small.pdf")

# London crowded situation
plot_speed_dayline(speeds_merged_london_all, '1737', figsize=(9,3), labels={'speed_counter': 'Detector', 'speed_probes': 'MeTS-10'})

fig, axs = plt.subplots(2, 1, figsize=(9, 7), tight_layout=True)
plot_counter_marker(speeds_merged_london_all, edges_london, '1737', ax=axs[0])
plot_speed_dayline(speeds_merged_london_all, '1737', labels={'speed_counter': 'Detector', 'speed_probes': 'MeTS-10'}, 
                   labelsize=14, ax=axs[1])
plt.savefig("counter_situation_flaky.pdf")

fig, axs = plt.subplots(2, 1, figsize=(9, 4.5), tight_layout=True)
plot_counter_marker(speeds_merged_london_all, edges_london, '1737', ax=axs[0], w=5)
plot_speed_dayline(speeds_merged_london_all, '1737', labels={'speed_counter': 'Detector', 'speed_probes': 'MeTS-10'}, 
                   labelsize=14, ax=axs[1])
plt.savefig("counter_situation_flaky_small.pdf")
