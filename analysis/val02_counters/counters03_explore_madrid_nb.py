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
import h5py
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

DAY = '2021-06-01'
# -

# ### Load road graph

edges_df = geopandas.read_parquet(RELEASE_PATH / 'road_graph' / 'madrid' / 'road_graph_edges.parquet')
edges_df

# ### Load loop counter locations matched

counters_assigned_df = geopandas.read_parquet(RELEASE_PATH / 'road_graph' / 'madrid' / 'counters_matched.parquet')
print(len(counters_assigned_df))
counters_assigned_df = counters_assigned_df[counters_assigned_df['way'] != -1]
counters_assigned_df

compute_highway_stats(counters_assigned_df, edges_df)


# ### Load loop counter values

# +
def t_from_time_bin(tb, do_floor=True):
    h, m = tb[10:].split(':')
    t = int(h) * 4 + int(float(m) / 15)
    if not do_floor:
        return t
    # T4c speed is floored, counters seem ceiled --> subtract 1
    t -= 1
    if t < 0:
        t = 95
    return t

def day_from_time_bin(tb, do_floor=True):
    if do_floor and t_from_time_bin(tb) == 95:
        d = datetime.strptime(tb, '%Y-%m-%d %H:%M') - timedelta(days=1)
        return d.strftime('%Y-%m-%d')
    return tb[:10]

def convert_counter_times(df, do_floor=True):
    df['t'] = [t_from_time_bin(tb, do_floor) for tb in df['time_bin']]
    df['day'] = [day_from_time_bin(tb, do_floor) for tb in df['time_bin']]
    return df

def convert_all_counters(output_path, m30_only=True):
    dfs = []
    loop_counter_files = sorted(list((DATA_PATH / 'loop_counters' / 'madrid' / 'all').glob('*.parquet')))
    for f in loop_counter_files:
        print(f)
        df = pd.read_parquet(f)
        if m30_only:
            df = df[df['type'] == 'M30']
        dfs.append(df)
    df = pd.concat(dfs)
    df = convert_counter_times(df, do_floor=False)
    df.to_parquet(output_path, compression="snappy")
    return df

# all_m30_df = convert_all_counters(DATA_PATH / 'loop_counters' / 'madrid' / 'm30_madrid_202106-202112.parquet')
# all_m30_df = pd.read_parquet(DATA_PATH / 'loop_counters' / 'madrid' / 'm30_madrid_202106-202112.parquet')
# all_m30_df


# -

counter_df = pd.read_parquet(COUNTER_PATH / 'madrid' / 'all' / f'counters_{DAY[:7]}.parquet')
counter_df = counter_df.rename(columns={'speed_avg': 'speed_counter'})
counter_df = convert_counter_times(counter_df)
counter_df

# ### Load T4c speed values

speed_df = pd.read_parquet(RELEASE_PATH / 'speed_classes' / 'madrid' / f'speed_classes_{DAY}.parquet')
speed_df = speed_df.rename(columns={'median_speed_kph': 'speed_probes'})
speed_df = speed_df.rename(columns={'volume': 'volume_probes'})
speed_df

# ## Compare Speed Values

counter_speed_df = counter_df[counter_df['type'] == 'M30']
counter_speed_df

counter_speeds_oneday = counter_speed_df[counter_speed_df['day'] == DAY].merge(
    counters_assigned_df[['id', 'way', 'u', 'v']], on=['id'])
counter_speeds_oneday

speeds_merged = counter_speeds_oneday.merge(speed_df, on=['u', 'v', 't', 'day'])
speeds_merged

counters_speed_df = speeds_merged[['id', 'u', 'v', 'way']].groupby(by=['id', 'u', 'v', 'way']).count().reset_index()
compute_highway_stats(counters_speed_df, edges_df)

speeds_merged = speeds_merged[speeds_merged['volume'] > 0]
speeds_merged['diff'] = (speeds_merged['speed_counter'] - speeds_merged['speed_probes']).abs()
speeds_merged

speeds_merged['diff'].hist(range=[0,120])

plot_scatters(speeds_merged)

speeds_merged_day = speeds_merged[(speeds_merged['t'] > 23) & (speeds_merged['t'] < 92)]
speeds_merged_day['diff'].hist(range=[0,120])

plot_scatters(speeds_merged_day)

# The difference of the kde plot in comparison to London can be explained. Madrid counters are all inner-city while in London we have speeds only on the highways on the ring around the city. Hence speeds measured by counters in Madrid are lower and affected more often by traffic lights etc.

_ = inspect_diff_area(speeds_merged, 0, 120, 0, 120, bbox_city='madrid')

speeds_merged.to_parquet(COUNTER_PATH / 'madrid' / 'speeds_merged_madrid.parquet', compression="snappy")

_ = inspect_diff_area(speeds_merged, 0, 120, 0, 120)

inspect_diff_area(speeds_merged, 40, 120, 0, 15)

speeds_merged['id'].unique()[:5]

len(speeds_merged['id'].unique())

top_left = inspect_diff_area(speeds_merged_day, 0, 40, 80, 120)
top_left.head(10)

len(speeds_merged_day[speeds_merged_day['id'] == 6699])

plot_counter(counters_assigned_df, edges_df, 6700)

plot_counter(counters_assigned_df, edges_df, 3599)

plot_counter(counters_assigned_df, edges_df, 3801)

plot_counter(counters_assigned_df, edges_df, 3560)

plot_counter(counters_assigned_df, edges_df, 6699)

plot_counter(counters_assigned_df, edges_df, 10205)

plot_counter(counters_assigned_df, edges_df, 6699)

# So seems tunnels are a problem. Let's have a look at the corresponding segments and the speed distribution:

# edges_df, 3599)
counters_assigned_df[counters_assigned_df['id'] == 3599]

# Let's find all counters in tunnels

counters_assigned_df['osmid'] = counters_assigned_df['way'].astype(str)
counters_edges_df = counters_assigned_df.merge(edges_df, on=['u', 'v', 'osmid'])
tunnel_counters_df = counters_edges_df[counters_edges_df['tunnel'] == 'yes']
# tunnel_counters_df

tunnel_counter_ids = list(tunnel_counters_df['id'])
print(len(tunnel_counter_ids))
print(tunnel_counter_ids)

speeds_merged_day_filtered = speeds_merged_day[~speeds_merged_day['id'].isin(tunnel_counter_ids)]
plot_scatters(speeds_merged_day_filtered)

plot_scatters(speeds_merged_day_filtered, 24, 36)  # 6 to 9am

plot_scatters(speeds_merged_day_filtered, 36, 48)  # 9 to 12am

plot_scatters(speeds_merged_day_filtered, 48, 60)  # 12am to 3pm

plot_scatters(speeds_merged_day_filtered, 60, 72)  # 3 to 6pm

plot_scatters(speeds_merged_day_filtered, 72, 84)  # 6 to 9pm

# Find similarities in counters

counters_edges_m30_df = counters_edges_df[counters_edges_df['id'].isin(
    counter_df[counter_df['type'] == 'M30']['id'].unique())]
counters_edges_m30_df = counters_edges_m30_df[~counters_edges_m30_df['id'].isin(tunnel_counter_ids)]
counters_edges_m30_df
# counters_edges_df[counters_edges_df['type'] == 'M30']
# oneway True/False
# length_meters

counters_edges_m30_df['highway'] = counters_edges_m30_df['highway'].str.replace(
    '.*motorway_link.*', 'motorway_link', regex=True)

counters_edges_m30_df['highway'].unique()

for hw in counters_edges_m30_df['highway'].unique():
    hw_ids = list(counters_edges_m30_df[counters_edges_m30_df['highway'] == hw]['id'])
    num_ids = len(hw_ids)
    print(f'{hw} ({num_ids})')
    if num_ids < 5 or hw in ['residential']:
        continue
    plot_scatters(speeds_merged_day_filtered[speeds_merged_day_filtered['id'].isin(hw_ids)],
                  title=f'{hw} ({len(hw_ids)})')

counters_edges_m30_df[counters_edges_m30_df['highway'] == 'residential']

plot_counter(counters_assigned_df, edges_df, 1006)

counters_edges_m30_df[counters_edges_m30_df['highway'] == 'primary']

plot_counter(counters_assigned_df, edges_df, 10211)

good_ids = list(counters_edges_m30_df[counters_edges_m30_df['highway'] == 'motorway']['id'])
good_ids.extend(list(counters_edges_m30_df[counters_edges_m30_df['highway'] == 'trunk']['id']))
print(f'good counters: {len(good_ids)}')
plot_scatters(speeds_merged_day_filtered[speeds_merged_day_filtered['id'].isin(good_ids)])

good_top_left = inspect_diff_area(
    speeds_merged_day_filtered[speeds_merged_day_filtered['id'].isin(good_ids)], 0, 40, 80, 120)
good_top_left.head(10)

plot_counter(counters_assigned_df, edges_df, 6700)
# probably too close to tunnel

plot_counter(counters_assigned_df, edges_df, 3560)
# probably also too close to tunnel

plot_counter(counters_assigned_df, edges_df, 6952)
# maybe caused by the gas or bus station

# let's remove the two near-tunnels
# good_ids.remove(6700)
# good_ids.remove(3560)
print(f'good counters: {len(good_ids)}')
speeds_merged_day_filtered_good = speeds_merged_day_filtered[speeds_merged_day_filtered['id'].isin(good_ids)]
plot_scatters(speeds_merged_day_filtered_good)

speeds_merged_day_filtered_good.to_parquet(
    COUNTER_PATH / 'madrid' / 'speeds_merged_clean_madrid.parquet', compression="snappy")

# Now the above looks nice, the three upper bumps seem to be a combination of space/time mean and speed limits!
#
#
# Let's do some stats: how many diffs are +-10%

good_speeds_merged = speeds_merged_day_filtered[speeds_merged_day_filtered['id'].isin(good_ids)]
print(len(good_speeds_merged))
good_speeds_merged['pdiff'] = good_speeds_merged['diff'] / good_speeds_merged['free_flow_kph'] * 100
print(len(good_speeds_merged[good_speeds_merged['pdiff'] < 10]))
print(len(good_speeds_merged[good_speeds_merged['pdiff'] < 15]))
print(len(good_speeds_merged[good_speeds_merged['pdiff'] < 20]))
print(len(good_speeds_merged[good_speeds_merged['pdiff'] < 25]))
good_speeds_merged['pdiff'].hist()

# Continue with the remaining large diffs:

speeds_merged_day[speeds_merged_day['diff'] > 50]

plot_speed_dayline(speeds_merged, 1001)

plot_counter(counters_assigned_df, edges_df, 1001)

plot_speed_dayline(speeds_merged, 10662)

plot_counter(counters_assigned_df, edges_df, 10662)

plot_speed_dayline(speeds_merged, 6693)

plot_counter(counters_assigned_df, edges_df, 6693)

plot_speed_dayline(speeds_merged, 7131)

plot_counter(counters_assigned_df, edges_df, 7131)

# # Compare Flow Values

counter_flow_df = counter_df[counter_df['type'] != 'M30']
counter_flow_df

counter_flows_oneday = counter_flow_df[counter_flow_df['day'] == '2021-06-01'].merge(
    counters_assigned_df[['id', 'way', 'u', 'v']], on=['id'])
counter_flows_oneday

len(counter_flows_oneday['id'].unique())

list(counter_flows_oneday['id'].unique())[1000:1010]

list(counter_flows_oneday[counter_flows_oneday['volume'] > 3000]['id'].unique())

flows_merged = counter_flows_oneday.merge(speed_df, on=['u', 'v', 't', 'day'])
flows_merged

plot_flow_dayline(flows_merged, 3395)

plot_flow_dayline(flows_merged, 3414)

plot_flow_dayline(flows_merged, 7018)

plot_flow_dayline(flows_merged, 10799)


# +
def speed_hist(s):
    h = np.histogram(s, bins=6)
    return h
    return h[-1][-1]


speed_hists = flows_merged[flows_merged['speed_probes'] > 0][['id', 'speed_probes']]
speed_hists = speed_hists.groupby(by=['id']).agg(list).reset_index()
speed_hists['shist'] = [speed_hist(s) for s in speed_hists['speed_probes']]
print(speed_hists[speed_hists['id'] == 3402]['shist'].values[0])
speed_hists[speed_hists['id'] == 3402]
# -

plot_counter(counters_assigned_df, edges_df, 3677)

for id in list(flows_merged[flows_merged['volume'] > 1000]['id'].unique())[::10]:
    plot_flow_dayline(flows_merged, id)
