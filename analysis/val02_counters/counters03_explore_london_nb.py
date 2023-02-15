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
import osmnx as ox
import networkx as nx
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

from exploration_utils import plot_counter, plot_counters, plot_locations, inspect_diff_area, plot_scatters
from exploration_utils import plot_speed_dayline, plot_flow_dayline, compute_highway_stats

RELEASE_PATH = Path('/private/data/mets10/release20221026_residential_unclassified/2022')
COUNTER_PATH = Path('/private/data/mets10/loop_counters')

lat_min, lat_max, lon_min, lon_max = (51.205, 51.7, -0.369, 0.067)
# -

# ### Load road graph

edges_df = geopandas.read_parquet(RELEASE_PATH / 'road_graph' / 'london' / 'road_graph_edges.parquet')
edges_df

# ### Load loop counter locations matched

counters_assigned_df = geopandas.read_parquet(RELEASE_PATH / 'road_graph' / 'london' / 'counters_matched.parquet')
print(len(counters_assigned_df))
counters_assigned_df = counters_assigned_df[counters_assigned_df['way'] != -1]
counters_assigned_df

compute_highway_stats(counters_assigned_df, edges_df)

counters_assigned_df = counters_assigned_df[(counters_assigned_df['way'] != -1) & (counters_assigned_df['way'] != '-1')]
counters_assigned_df


# +
def way_ambiguity(ways, way_dists):
    way_dists = eval(way_dists)
    if len(way_dists) < 2:
#         print(f'{ways}: {way_dists}')
        return 10
    p = 0
    if way_dists[0] == way_dists[1] and len(way_dists) > 2:
        p = 1  # TODO use check for oneway here
    amb = (way_dists[p+1] - way_dists[p])
    if amb == 0:
        return 0
    amb =  way_dists[p] / amb
    if amb > 10:
        amb = 10
    return amb


counters_assigned_df['way_ambiguity'] = [way_ambiguity(w, wd) for w, wd in
    zip(counters_assigned_df['ways'], counters_assigned_df['way_dists'])]
counters_assigned_df
# -

counters_assigned_df['way_ambiguity'].hist()

# ### Load loop counter speed values (WEBTRIS)

counter_speed_df = pd.read_parquet(COUNTER_PATH / 'london' / 'speed' / 'webtris_london_201907-202001.parquet')
counter_speed_df = counter_speed_df[
        (counter_speed_df['lat'] >= lat_min) & (counter_speed_df['lat'] <= lat_max) &
        (counter_speed_df['lon'] >= lon_min) & (counter_speed_df['lon'] <= lon_max)]
all_webtris_df = counter_speed_df
counter_speed_df

plot_speed_dayline(all_webtris_df[(all_webtris_df['day'] == '2020-01-22')], '6137', plot_volume=True)
plot_counter(counters_assigned_df, edges_df, '6137')

plot_speed_dayline(all_webtris_df[(all_webtris_df['day'] == '2020-01-29')], '6137', plot_volume=True)
plot_counter(counters_assigned_df, edges_df, '6137')

# +
# plot_speed_dayline(all_webtris_df[(all_webtris_df['day'] == '2019-01-01')], '6137', plot_volume=True)
# plot_counter(counters_assigned_df, edges_df, '6137')

# +
# plot_speed_dayline(all_webtris_df[(all_webtris_df['day'] == '2019-07-04')], '6137', plot_volume=True)
# plot_counter(counters_assigned_df, edges_df, '6137')

# +
# plot_speed_dayline(all_webtris_df[(all_webtris_df['day'] == '2020-01-29')], '1737', plot_volume=True)
# plot_counter(counters_assigned_df, edges_df, '1737')
# -

# Select a single day and merge with the locations
counter_speeds_oneday = counter_speed_df[counter_speed_df['day'] == '2020-01-29'].merge(
    counters_assigned_df[['id', 'way', 'u', 'v']], on=['id'])
counter_speeds_oneday


# +
def is_valid_counter(s):
    h = np.histogram(s, bins=2)
    return h[-1][-1]

invalid_counters = counter_speeds_oneday[['id', 'speed_counter']].groupby(by=['id']).agg(list).reset_index()
invalid_counters['shist'] = [is_valid_counter(s) for s in invalid_counters['speed_counter']]
invalid_counters = invalid_counters[invalid_counters['shist'] < 1]
invalid_counters
# -

counter_speeds_oneday = counter_speeds_oneday[~counter_speeds_oneday['id'].isin(invalid_counters['id'])]
counter_speeds_oneday

# ### Load T4c speed values

speed_files = sorted(list((RELEASE_PATH / 'speed_classes' / 'london').glob('*.parquet')))
speed_files[-3:]

# speed_df = pd.read_parquet(speed_files[0])
speed_df = pd.read_parquet(RELEASE_PATH / 'speed_classes' / 'london' / 'speed_classes_2020-01-29.parquet')
speed_df = speed_df.rename(columns={'median_speed_kph': 'speed_probes'})
speed_df

# # Compare Speed Values

speeds_merged = counter_speeds_oneday.merge(speed_df, on=['u', 'v', 't'])
speeds_merged

speeds_merged = speeds_merged[speeds_merged['volume_y'] > 0]
speeds_merged['diff'] = (speeds_merged['speed_counter'] - speeds_merged['speed_probes']).abs()
speeds_merged

len(speeds_merged['id'].unique())

counters_speed_df = speeds_merged[['id', 'u', 'v', 'way']].groupby(by=['id', 'u', 'v', 'way']).count().reset_index()
counters_speed_df['way'] = counters_speed_df['way'].astype(str)
counters_stats_df = counters_speed_df.merge(edges_df, left_on=['u', 'v', 'way'], right_on=['u', 'v', 'osmid'])
counters_stats_df = counters_stats_df[['id', 'highway']]
counters_stats_df['hwc'] = counters_stats_df['highway'].str.replace('_link', '')
counters_stats_df['hwc'] = counters_stats_df['hwc'].str.replace('.*trunk.*', 'trunk')
counters_stats_df['hwc'] = counters_stats_df['hwc'].str.replace('.*primary.*', 'primary')
highway_stats = counters_stats_df[['hwc', 'highway']].groupby(by=['hwc']).count().reset_index()
highway_stats = highway_stats.rename(columns={'highway': 'count'})
highway_stats = highway_stats.rename(columns={'hwc': 'highway'})
highway_stats['share'] = (highway_stats['count'] / len(counters_stats_df) * 100).round()
highway_stats = highway_stats.set_index('highway')
highway_stats.style.format({'share': "{:.0f} %"})

speeds_merged['diff'].hist()

plot_scatters(speeds_merged)

speeds_merged_day = speeds_merged[(speeds_merged['t'] > 23) & (speeds_merged['t'] < 92)]
speeds_merged_day['diff'].hist()

plot_scatters(speeds_merged_day)

_ = inspect_diff_area(speeds_merged, 0, 120, 0, 120)

_ = inspect_diff_area(speeds_merged, 0, 120, 0, 120, bbox_city='london')

speeds_merged.to_parquet(COUNTER_PATH / 'london' / 'speeds_merged_london.parquet', compression="snappy")

# hw_df = edges_df[['u', 'v', 'gkey', 'highway']]#.groupby(by=['u', 'v', 'gkey', 'highway']).count().reset_index()
speeds_merged_nolink = speeds_merged.merge(edges_df[['u', 'v', 'gkey', 'highway']], on=['u', 'v', 'gkey'])
print(len(speeds_merged_nolink))
speeds_merged_nolink = speeds_merged_nolink[(~speeds_merged_nolink['highway'].str.contains('_link'))]
speeds_merged_nolink['diff'].hist()

plot_scatters(speeds_merged_nolink)

speeds_merged_nolink.to_parquet(COUNTER_PATH / 'london' / 'speeds_merged_clean_london.parquet', compression="snappy")

top_left = inspect_diff_area(speeds_merged_nolink, 0, 40, 80, 120)
top_left.head(5)

top_mid = inspect_diff_area(speeds_merged_nolink, 40, 75, 100, 120)
top_mid.head(5)

speeds_merged[speeds_merged['diff'] > 50]

plot_speed_dayline(speeds_merged, '97')
plot_counter(counters_assigned_df, edges_df, '97')

plot_speed_dayline(speeds_merged, '10510')
plot_counter(counters_assigned_df, edges_df, '10510')

plot_speed_dayline(speeds_merged, '1737')
plot_counter(counters_assigned_df, edges_df, '1737')

# +
# plot_speed_dayline(speeds_merged, '6137', plot_volume=True)
# plot_counter(counters_assigned_df, edges_df, '6137')
