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


MOVIE_BBOXES = {
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


def plot_counter(cdf, edf, id, figsize=(930, 300)):
    data = cdf[cdf['id'] == id]
    assert len(data) == 1
    lat = data.lat.values[0]
    lon = data.lon.values[0]
    pt = (lat, lon)
    bb = [(lat-0.001, lon-0.001), (lat+0.001, lon+0.001)]
    
    way = data.way.values[0]
    u = data.u.values[0]
    v = data.v.values[0]
    data = edf[(edf['u'] == u) & (edf['v'] == v)]
    line = [(c[1], c[0]) for c in data.geometry.values[0].coords]
    
    f = folium.Figure(width=figsize[0], height=figsize[1])
    m = folium.Map().add_to(f)
    folium.Marker(pt).add_to(m)
    folium.PolyLine(line, weight=5, opacity=1).add_to(m)
    m.fit_bounds(bb)
    return m


def plot_counters(cdf):
    f = folium.Figure(width=600, height=300)
    m = folium.Map().add_to(f)
    lat_min = 1e7
    lat_max = -1e7
    lon_min = 1e7
    lon_max = -1e7
    for id, lat, lon in zip(cdf['id'], cdf['lat'], cdf['lon']):
        pt = (lat, lon)
        folium.Marker(pt, popup=f'{id}').add_to(m)
        lat_min = min(lat_min, lat)
        lat_max = max(lat_max, lat)
        lon_min = min(lon_min, lon)
        lon_max = max(lon_max, lon)
    bb = [(lat_min, lon_min), (lat_max, lon_max)]
    m.fit_bounds(bb)
    return m


def plot_locations(df, bbox_city=None, ax=None):
    fig = None
    if not ax:
        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(8,8))
    if bbox_city:
        bbox = tuple([c / 1e5 for c in MOVIE_BBOXES[bbox_city]['bounds']])
        lat_min, lat_max, lon_min, lon_max = bbox
        ax.set_xlim([lon_min,lon_max])
        ax.set_ylim([lat_min,lat_max])
    if 'cnt' in df.columns:
        df.plot(kind="scatter", x="lon", y="lat",
            s=30, label="cnt", c='cnt', cmap=plt.get_cmap("jet"),
            colorbar=True, alpha=0.4, ax=ax
        )
        plt.legend()
    else:
        df.plot(kind="scatter", x="lon", y="lat",
            s=30, c='black', cmap=plt.get_cmap("jet"),
            colorbar=False, alpha=0.4, ax=ax
        )
    if fig:
        plt.show()


def inspect_diff_area(df, xmin, xmax, ymin, ymax, xfield='speed_probes', yfield='speed_counter', bbox_city=None):
    diff_df = df[
        (df[xfield] >= xmin) & (df[xfield] < xmax) &
        (df[yfield] >= ymin) & (df[yfield] < ymax)
    ]
    num_area = len(diff_df)
    num_all = len(df)
    perc = (num_area / num_all) * 100
    print(f'Area (x in {(xmin, xmax)}, y in {(ymin, ymax)} contains {num_area} / {num_all} readings ({perc:.2f}%)')
    unique_locations = diff_df[['id', 'lat', 'lon', yfield]].groupby(['id', 'lat', 'lon']).count().reset_index()
    unique_locations = unique_locations.rename(columns={yfield: 'cnt'})
    print(f'Readings are on {len(unique_locations)} / {len(df["id"].unique())} locations')
    unique_locations = unique_locations.sort_values('cnt', ascending=False)
    plot_locations(unique_locations, bbox_city)
    return unique_locations


def plot_kde_scatter(df, ax, city=None, labelsize=24, titlesize=36):
    ax.plot([0, 130], [0, 130], ls="--", c=".3")
    ax.set_xlim([0,130])
    ax.set_ylim([0,130])
    sns.kdeplot(data=df, x='speed_probes', y='speed_counter', fill=True, levels=10, ax=ax)
    if city:
        ax.set_xlabel(f'MeTS-10 Speed [km/h]', fontsize=titlesize)
        ax.set_ylabel('Detector Speed [km/h]', fontsize=titlesize)
        ax.tick_params(axis='x', which='major', labelsize=labelsize)
        ax.tick_params(axis='y', which='major', labelsize=labelsize)
        ax.title.set_text(city)
        ax.title.set_size(titlesize)


def plot_scatters(df, t_from=None, t_to=None, title=None):
    if t_from and t_to:
        df = df[(df['t'] >= t_from) & (df['t'] < t_to)]
    fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(20,10))
    if title:
        fig.suptitle(title)
    ax = axs[0]
    plot_kde_scatter(df, ax)
    ax = axs[1]
    ax.plot([0, 130], [0, 130], ls="--", c=".3")
    ax.set_xlim([0,130])
    ax.set_ylim([0,130])
    df.plot(x='speed_probes', y='speed_counter', kind='scatter', ax=ax)


def plot_speed_dayline(df, id, x_field="t", plot_volume=False, figsize=(16,6), labels={}, labelsize=12, ax=None):
    cols = []
    label_names = []
    for c in ["speed_counter", "speed_probes", "v_kfz_det_hr", "v_pkw_det_hr", "v_lkw_det_hr"]:
        if c in df.columns:
            cols.append(c)
            if c in labels:
                label_names.append(labels[c])
            else:
                label_names.append(c)
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    df[df['id'] == id].plot(x=x_field, y=cols, label=label_names, fontsize=labelsize, ax=ax)
    ax.set_ylabel('Speed [km/h]', fontsize=labelsize)
    ax.set_xlabel('t (15 minute interval)', fontsize=labelsize)
    plt.legend(fontsize=labelsize)
    if plot_volume:
        df[df['id'] == id].plot(x=x_field, kind='bar', y=['volume'], ax=ax)


def plot_flow_dayline(df, id, x_field="t", vol_field="volume"):
    df = df[df['id'] == id].copy()
    df['vol10th'] = df[vol_field] / 10
    cols = []
    for c in ['t', 'vol10th', 'occupation', 'congestion_level', 'sat_class', 'speed_probes', 'free_flow_kph']:
        if c in df.columns:
            cols.append(c)
    df[cols].plot(
        x=x_field, figsize=(16,6))


# +
hw_order = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'unclassified']

def normalize_highway(hw):
    hw = hw.replace('_link', '')
    if hw[0] != '[':
        return hw
    hw = eval(hw)
    for hwo in hw_order:
        if hwo in hw:
            return hwo
    return hw_order[-1]


# -

def compute_highway_stats(counters_df, edges_df):
    counters_df = counters_df.copy()
    counters_df['way'] = counters_df['way'].astype(str)
#     counters_stats_df = counters_df.merge(edges_df, left_on=['u', 'v', 'way'], right_on=['u', 'v', 'osmid'])
    counters_stats_df = counters_df.merge(edges_df, left_on=['u', 'v'], right_on=['u', 'v'])
    counters_stats_df = counters_stats_df[['id', 'highway']]
    counters_stats_df['hwc'] = counters_stats_df['highway'].str.replace('_link', '', regex=True)
    counters_stats_df['hwc'] = counters_stats_df['hwc'].str.replace('.*motorway.*', 'motorway', regex=True)
    counters_stats_df['hwc'] = counters_stats_df['hwc'].str.replace('.*trunk.*', 'trunk', regex=True)
    counters_stats_df['hwc'] = counters_stats_df['hwc'].str.replace('.*primary.*', 'primary', regex=True)
    counters_stats_df['hwc'] = counters_stats_df['hwc'].str.replace('.*secondary.*', 'secondary', regex=True)
    counters_stats_df['hwc'] = counters_stats_df['hwc'].str.replace('.*tertiary.*', 'tertiary', regex=True)
    counters_stats_df['hwc'] = counters_stats_df['hwc'].str.replace('.*residential.*', 'residential', regex=True)
    counters_stats_df['hwc'].unique()
    highway_stats = counters_stats_df[['hwc', 'highway']].groupby(by=['hwc']).count().reset_index()
    highway_stats = highway_stats.rename(columns={'highway': 'count'})
    highway_stats = highway_stats.rename(columns={'hwc': 'highway'})
    highway_stats['share'] = (highway_stats['count'] / len(counters_stats_df) * 100).round()
    highway_stats['share_full'] = (highway_stats['count'] / len(counters_stats_df) * 100)
    highway_stats = highway_stats.set_index('highway')
    return highway_stats.style.format({'share': "{:.0f} %"})
