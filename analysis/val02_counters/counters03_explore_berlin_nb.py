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
import osmnx as ox
import networkx as nx
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

from exploration_utils import plot_counter, plot_counters, plot_locations, inspect_diff_area, plot_scatters
from exploration_utils import plot_speed_dayline, plot_flow_dayline, compute_highway_stats

RELEASE_PATH = Path('/private/data/mets10/release20221026_residential_unclassified/2021')
COUNTER_PATH = Path('/private/data/mets10/loop_counters')
# -

# ### Load road graph

edges_df = geopandas.read_parquet(RELEASE_PATH / 'road_graph' / 'berlin' / 'road_graph_edges.parquet')
edges_df

# ### Load loop counter locations matched

counters_assigned_df = geopandas.read_parquet(RELEASE_PATH / 'road_graph' / 'berlin' / 'counters_matched.parquet')
print(len(counters_assigned_df))
counters_assigned_df = counters_assigned_df[counters_assigned_df['way'] != -1]
counters_assigned_df

compute_highway_stats(counters_assigned_df, edges_df)

# ### Load loop counter speed values (2019-06-04 is a Tuesday)

loop_counter_speed_files = sorted(list((COUNTER_PATH / 'berlin' / 'speed').glob('*.parquet')))
loop_counter_speed_files[:3]

counter_speed_df = pd.read_parquet(COUNTER_PATH / 'berlin' / 'speed' / 'counters_2019-06.parquet')
counter_speed_df

counter_speed_df['day'] = [tb[:10] for tb in counter_speed_df['time_bin']]
counter_speed_df['h'] = counter_speed_df['stunde']
counter_speed_df['speed_counter'] = counter_speed_df['v_kfz_det_hr'].astype(float)
counter_speed_df['volume'] = counter_speed_df['q_kfz_det_hr'].astype(int)
counter_speed_df

counter_speed_df[counter_speed_df['day'] == '2019-06-05']

# Select a single day and merge with the locations
counter_speeds_20190605 = counter_speed_df[counter_speed_df['day'] == '2019-06-05'].merge(
    counters_assigned_df[['id', 'way', 'u', 'v']], on=['id'])
counter_speeds_20190605


# +
def is_valid_counter(s):
    h = np.histogram(s, bins=2)
    return h[-1][-1]

invalid_counters = counter_speeds_20190605[['id', 'speed_counter']].groupby(by=['id']).agg(list).reset_index()
invalid_counters['shist'] = [is_valid_counter(s) for s in invalid_counters['speed_counter']]
invalid_counters = invalid_counters[invalid_counters['shist'] < 1]
invalid_counters
# -

counter_speeds_20190605 = counter_speeds_20190605[~counter_speeds_20190605['id'].isin(invalid_counters['id'])]
counter_speeds_20190605

# ### Load T4c speed values

speed_files = sorted(list((RELEASE_PATH / 'speed_classes' / 'berlin').glob('*.parquet')))
speed_files[:4]

speed_df = pd.read_parquet(RELEASE_PATH / 'speed_classes' / 'berlin' / 'speed_classes_2019-06-05.parquet')
speed_df

# TODO aggregate hours
speed_df['h'] = [int(t/4) for t in speed_df['t']]
speed_df

speed_df = speed_df.groupby(by=['u', 'v', 'gkey', 'day', 'h']).mean().reset_index()
speed_df = speed_df.drop(columns=['t'])
speed_df = speed_df.rename(columns={'median_speed_kph': 'speed_probes'})
speed_df

# # Compare Speed Values

speeds_merged = counter_speeds_20190605.merge(speed_df, on=['u', 'v', 'h', 'day'])
speeds_merged

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

speeds_merged = speeds_merged[speeds_merged['volume_x'] > 0]
speeds_merged['diff'] = (speeds_merged['speed_counter'] - speeds_merged['speed_probes']).abs()
speeds_merged

speeds_merged['diff'].hist(range=[0,120])

plot_scatters(speeds_merged)

speeds_merged_day = speeds_merged[(speeds_merged['h'] > 6) & (speeds_merged['h'] < 23)]
speeds_merged_day['diff'].hist(range=[0,120])

plot_scatters(speeds_merged_day)

speeds_merged[speeds_merged['diff'] > 50]

plot_speed_dayline(speeds_merged, 100101010000268, x_field="h")

plot_counter(counters_assigned_df, edges_df, 100101010000268)

plot_speed_dayline(speeds_merged, 100101010033513, x_field="h")

plot_counter(counters_assigned_df, edges_df, 100101010033513)

plot_speed_dayline(speeds_merged, 100101010039371, x_field="h")

plot_counter(counters_assigned_df, edges_df, 100101010039371)

_ = inspect_diff_area(speeds_merged, 0, 120, 0, 120)

_ = inspect_diff_area(speeds_merged, 0, 120, 0, 120, bbox_city='berlin')

inspect_diff_area(speeds_merged, 20, 120, 0, 5)

# +
speeds_merged_filtered = speeds_merged[~speeds_merged['id'].isin(
    [100101010000268, 100101010021183, 100101010023409, 100101010023712, 100101010024015, 100101010027247, 100101010027550, 100101010027651, 100101010030681, 100101010033311, 100101010033412, 100101010033614, 100101010040886, 100101010054731, 100101010063421, 100101010068269, 100101010068370, 100101010079181, 100101010082215]
)]

plot_scatters(speeds_merged_filtered)
# -

speeds_merged.to_parquet(COUNTER_PATH / 'berlin' / 'speeds_merged_berlin.parquet', compression="snappy")

speeds_merged_filtered.to_parquet(
    COUNTER_PATH / 'berlin' / 'speeds_merged_clean_berlin.parquet', compression="snappy")

speeds_merged_filtered[['id', 'u', 'v', 'gkey']]

speeds_merged_filtered_day = speeds_merged_filtered[
    (speeds_merged_filtered['h'] > 6) & (speeds_merged_filtered['h'] < 23)]
speeds_merged_filtered_day['diff'].hist(range=[0,120])
plot_scatters(speeds_merged_filtered_day)

day_top_left = inspect_diff_area(speeds_merged_filtered_day, 0, 70, 80, 120)
day_top_left.head(10)

plot_counter(counters_assigned_df, edges_df, 100101010039371)
# next to the wholesale market?

day_bottom_left = inspect_diff_area(speeds_merged_filtered_day, 30, 70, 0, 15)
day_bottom_left.head(10)

plot_counter(counters_assigned_df, edges_df, 100101010061300)
# next to the wholesale market?

# Now on the day it looks already quite good.
#
# Let's do some stats: how many diffs are +-10%

good_speeds_merged = speeds_merged_filtered_day
# good_speeds_merged = speeds_merged_filtered_day[speeds_merged_filtered_day['id'].isin(good_ids)]
print(len(good_speeds_merged))
good_speeds_merged['pdiff'] = good_speeds_merged['diff'] / good_speeds_merged['free_flow_kph'] * 100
print(len(good_speeds_merged[good_speeds_merged['pdiff'] < 10]))
print(len(good_speeds_merged[good_speeds_merged['pdiff'] < 15]))
print(len(good_speeds_merged[good_speeds_merged['pdiff'] < 20]))
print(len(good_speeds_merged[good_speeds_merged['pdiff'] < 25]))
good_speeds_merged['pdiff'].hist()

# +
# Try with speedlimit filter
# -

edges_speed_limit = edges_df[['u', 'v', 'gkey', 'speed_kph']]
edges_speed_limit = edges_speed_limit.rename(columns={'speed_kph': 'speed_limit_kph'})
edges_speed_limit

speeds_merged_speed_limit = speeds_merged.merge(edges_speed_limit, on=['u', 'v', 'gkey'])
print(len(speeds_merged_speed_limit))
speeds_merged_speed_limit = speeds_merged_speed_limit[
    speeds_merged_speed_limit['speed_counter'] < (speeds_merged_speed_limit['speed_limit_kph'] * 1.2)]
speeds_merged_speed_limit

plot_scatters(speeds_merged_speed_limit)

len(speeds_merged_speed_limit['id'].unique())


# +
def plot_counter_daylines(id):
    speeds_merged[speeds_merged['id'] == id].plot(
        x="h", y=["speed_counter", "speed_probes", "v_pkw_det_hr", "v_lkw_det_hr",], figsize=(16,6))
    speeds_merged[speeds_merged['id'] == id].plot(
        x="h", y=["volume", "volume_class", "q_pkw_det_hr", "q_lkw_det_hr"], figsize=(16,6))

plot_counter_daylines(100101010024015)
# -

plot_counter_daylines(100101010023712)

# +
hwt_ids = [
    100101010000167, 100101010000369, 100101010000470, # Spanische Allebrücke
    100101010004312, 100101010004413, 100101010004514, # Prenzlauer Promenade
    100101010040785, 100101010040987]  # Pasewalker Strasse

plot_counter(counters_assigned_df, edges_df, hwt_ids[0])
# -

plot_counter(counters_assigned_df, edges_df, hwt_ids[1])

plot_counter(counters_assigned_df, edges_df, hwt_ids[2])

plot_counter(counters_assigned_df, edges_df, hwt_ids[3])

plot_counter(counters_assigned_df, edges_df, hwt_ids[4])

plot_counter(counters_assigned_df, edges_df, hwt_ids[5])

plot_counter(counters_assigned_df, edges_df, hwt_ids[6])

plot_counter(counters_assigned_df, edges_df, hwt_ids[7])

edges_df[(edges_df['speed_kph'] > 80)]['highway'].unique()

e2 = pd.read_parquet('/Users/neun/data/t4c/data_pipeline/release20220930/2021/road_graph/berlin/road_graph_edges.parquet')
e2

e2[(e2['speed_kph'] > 80)]['highway'].unique()
