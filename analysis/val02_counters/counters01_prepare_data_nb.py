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
import pandas as pd
import numpy as np
import geojson
import geopandas
from pathlib import Path
import requests
from math import floor
import json
from datetime import datetime, timedelta
import os
import xml.etree.ElementTree as ET
import locale
import zipfile
import tempfile
import glob

# DATA_PATH expects a working structure with mets10 data generated and city subfolders
# ├── loop_counters
# │   ├── berlin
# │   │   ├── downloads
# │   │   └── speed
# │   ├── london
# │   │   ├── downloads
# │   │   └── speed
# │   └── madrid
# │       ├── all
# │       └── downloads
# └── release20221026_residential_unclassified
#     ├── 2021
#     │   ├── road_graph
#     │   │   └── berlin
#     │   └── speed_classes
#     │       └── berlin
#     └── 2022
#         ├── road_graph
#         │   ├── london
#         │   └── madrid
#         └── speed_classes
#             ├── london
#             └── madrid
DATA_PATH = Path('/private/data/mets10') 
BERLIN_PATH = DATA_PATH / 'loop_counters' / 'berlin'
LONDON_PATH = DATA_PATH / 'loop_counters' / 'london'
MADRID_PATH = DATA_PATH / 'loop_counters' / 'madrid'
# -

# # Berlin
#
# Raw loop counter data was downloaded from
# https://api.viz.berlin.de/daten/verkehrsdetektion/teu_standorte.json and https://api.viz.berlin.de/daten/verkehrsdetektion?path=2020%2FDetektoren+%28einzelne+Fahrspur%29%2F

berlin_locations_df = pd.read_excel(BERLIN_PATH / 'downloads' / 'Stammdaten_Verkehrsdetektion_2022_07_20.xlsx')
berlin_locations_df = berlin_locations_df[[
    'DET_NAME_NEU', 'DET_ID15', 'RICHTUNG', 'SPUR', 'LÄNGE (WGS84)', 'BREITE (WGS84)']]
berlin_locations_df = berlin_locations_df.rename(columns={
    'DET_NAME_NEU': 'detname', 'DET_ID15': 'detid_15', 'LÄNGE (WGS84)': 'lon', 'BREITE (WGS84)': 'lat',
    'SPUR': 'lane', 'RICHTUNG': 'heading'})
berlin_locations_df = berlin_locations_df.replace({'heading': {
    'Nord': 0, 'Nordost': 45, 'Ost': 90, 'Südost': 135,
    'Süd': 180, 'Südwest': 225, 'West': 270, 'Nordwest': 315
}})
berlin_locations_df = berlin_locations_df.groupby(by=['detname', 'detid_15', 'heading', 'lane', 'lon', 'lat']).count()
berlin_locations_df = berlin_locations_df.reset_index()
berlin_locations_df

berlin_locations_df[berlin_locations_df['detid_15'].duplicated()]


# +
def get_counters_merged(year, month, locations_df):
    df = pd.read_csv(BERLIN_PATH / 'downloads' / f'det_val_hr_{year:04d}_{month:02d}.csv.gz', sep=';',
                     compression='gzip')
    print(f'Loaded {len(df)} rows with {len(df["detid_15"].unique())} unique counters')
    df = df.merge(locations_df, on=['detid_15'], how='left')
    df = df.rename(columns={'detid_15': 'id', 'detname': 'name'})
    df['time_bin'] = [f'{d[6:]}-{d[3:5]}-{d[:2]} {h:02d}:00' for d, h in zip(df['tag'], df['stunde'])]
    df.to_parquet(BERLIN_PATH / 'speed' / f'counters_{year:04d}-{month:02d}.parquet', compression="snappy")
    return df
    
berlin202107_df = get_counters_merged(2021, 7, berlin_locations_df)
berlin202107_df
# -

berlin201906_df = get_counters_merged(2019, 6, berlin_locations_df)
berlin201906_df

# # London
#
# Scraping counter data from the Highways England WebTRIS API https://webtris.highwaysengland.co.uk/

# +
WEBTRIS = 'https://webtris.highwaysengland.co.uk/api/v1'
LONDON_BBOX = [5120500, 5170000, -36900, 6700]

def save_json_rows(data, file_name):
    if not data:
        return
    with open(LONDON_PATH / 'downloads' / f'{file_name}.json', 'w') as f:
        for row in data:
            json.dump(row, f)
            f.write('\n')

def get_webtris_sites(bbox):
    min_lat, max_lat, min_lon, max_lon = tuple(bbox)
    url = f'{WEBTRIS}/sites'
    req = requests.get(url)
    req_json = req.json()
    print(req_json['row_count'])
    for site in req_json['sites']:
        lat_bin = int(floor(site['Latitude'] * 1e3) / 1e3 * 1e5)
        lon_bin = int(floor(site['Longitude'] * 1e3) / 1e3 * 1e5)
        if lat_bin >= min_lat and lat_bin <= max_lat:
            site['lat_bin'] = lat_bin
            site['lon_bin'] = lon_bin
            yield site

def time_ceil(time, delta):
    mod = (time - datetime(1970, 1, 1)) % delta
    if mod:
        return time + (delta - mod)
    return time

def time_bin_format(time):
    return time.strftime('%Y-%m-%d %H:%M')

def get_time_bins(datetime_str):
    time_bins = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    time_bins = time_ceil(time_bins, timedelta(minutes=5))
    return [
        time_bin_format(time_bins-timedelta(minutes=10)),
        time_bin_format(time_bins-timedelta(minutes=5)),
        time_bin_format(time_bins)
    ]

def join_site_info(data, sites_dict):
    for row in data:
        site = sites_dict[row['Site Name']]
        volume = row['Total Volume']
        if volume:
            volume = int(volume)
        else:
            volume = 0
        speed = row['Avg mph']
        if speed:
            speed = float(speed)*1.60934
        else:
            speed = 0
        time_bins = get_time_bins(f"{row['Report Date'][:10]} {row['Time Period Ending']}")
        yield {
            'id': int(site['Id']), 'name': row['Site Name'],
            'lat_bin': site['lat_bin'], 'lon_bin': site['lon_bin'],
            'lat': site['Latitude'], 'lon': site['Longitude'],
            'time_bins': time_bins,
            'volume': volume, 'speed': speed
        }

def get_chunk(id_chunk, date_from, date_to):
    date_from = datetime.strptime(date_from, '%Y-%m-%d').strftime('%d%m%Y')
    date_to = datetime.strptime(date_to, '%Y-%m-%d').strftime('%d%m%Y')
    req_ids_urlenc = ','.join(id_chunk)
    url = f'{WEBTRIS}/reports/{date_from}/to/{date_to}/Daily?sites={req_ids_urlenc}&page=1&page_size=10000'
    while True:
        print(url)
        req = requests.get(url)
        if req.status_code == 204:
            return
        req_json = req.json()
        yield from req_json['Rows']
        header = req_json['Header']
        print(header)
        if 'links' in header and len(header['links']) > 0 and header['links'][-1]['rel'] == 'nextPage':
            url =  header['links'][-1]['href']
            continue
        else:
            break


# -

webtris_sites = list(get_webtris_sites(LONDON_BBOX))
save_json_rows(webtris_sites, 'webtris_sites')
print(len(webtris_sites))
webtris_sites[:2]


# +
def download_webtris_chunks(webtris_sites, date_from, date_to, chunk_size=20):
    sites_dict = { s['Description']: s for s in webtris_sites }
    req_ids = [s['Id'] for s in webtris_sites]
    req_id_chunks = [req_ids[i:i + chunk_size] for i in range(0, len(req_ids), chunk_size)]
    print(len(req_id_chunks))
    for i in range(len(req_id_chunks)):
        print(f'Downloading chunk {i:03}...')
        chunk_data = get_chunk(req_id_chunks[i], date_from=date_from, date_to=date_to)
        joined_data = join_site_info(chunk_data, sites_dict)
        save_json_rows(joined_data, f'webtris_chunk_{date_from}_{date_to}_{i:03}')
        
# download_webtris_chunks(webtris_sites, '2019-07-01', '2019-07-31')
# download_webtris_chunks(webtris_sites, '2019-08-01', '2019-08-31')
# download_webtris_chunks(webtris_sites, '2019-09-01', '2019-09-30')
# download_webtris_chunks(webtris_sites, '2019-10-01', '2019-10-31')
# download_webtris_chunks(webtris_sites, '2019-11-01', '2019-11-30')
# download_webtris_chunks(webtris_sites, '2019-12-01', '2019-12-31')
download_webtris_chunks(webtris_sites, '2020-01-01', '2020-01-31')


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

def convert_webtris_chunks(do_floor=True):
    chunk_files = sorted((LONDON_PATH / 'downloads').glob('webtris_chunk_*.json'))
    df = pd.concat([pd.read_json(cf,  lines=True) for cf in chunk_files])
    df['id'] = df['id'].astype(str)
    df['heading'] = -1
    df['time_bin'] = [x[2] for x in df['time_bins']]
    df = df.rename(columns={'speed': 'speed_counter'})
    df['t'] = [t_from_time_bin(tb, do_floor) for tb in df['time_bin']]
    df['day'] = [day_from_time_bin(tb, do_floor) for tb in df['time_bin']]
    df = df[['id', 'name', 'lat', 'lon', 'heading', 'time_bin', 'volume', 'speed_counter', 't', 'day']]
    return df

webtris_df = convert_webtris_chunks()
webtris_df
# -

webtris_df['name'] = webtris_df['name'].astype(str)
webtris_df.to_parquet(LONDON_PATH / 'speed' / 'webtris_london_201907-202001.parquet', compression="snappy")


# # Madrid
#
# The raw files were downloaded from https://datos.madrid.es/ in zipped CSV files.

def download_dcat_zips(dcat_file, prefix, month_names=False, months=None):
    root = ET.parse(MADRID_PATH / 'downloads' / dcat_file).getroot()
    ns = {'dcat': 'http://www.w3.org/ns/dcat#',
          'dct': 'http://purl.org/dc/terms/'}
    data_urls = {}
    for entry in root.findall('.//dcat:Distribution', ns):
        title = entry.find('dct:title', ns).text
        url = entry.find('dcat:accessURL', ns).text
        if url.endswith('.zip'):
            try:
                if month_names:
                    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
                    title = title.replace('diciiembre', 'diciembre')
                    month = datetime.strptime(title, '%Y. %B')
                else:
                    month = datetime.strptime(title[6:], '%d/%m/%Y')
            except Exception as e:
                print(e)
                continue
            month = datetime.strftime(month, '%Y-%m')
            if months:
                if not month in months:
                    continue
            data_urls[month] = url
            print(f'{month}: {url}')
            r = requests.get(url, allow_redirects=True)
            open(MADRID_PATH / 'downloads' / f"{prefix}-{month}.zip", 'wb').write(r.content)
    return data_urls


# +
url = "https://datos.madrid.es/egob/catalogo/202468-0-intensidad-trafico.dcat"
out_file = MADRID_PATH / 'downloads' / '202468-0-intensidad-trafico.dcat'
os.system(f"wget -O {out_file} {url}")

download_dcat_zips('202468-0-intensidad-trafico.dcat', prefix='locations', month_names=False,
                   months=['2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12'])

# +
url = "https://datos.madrid.es/egob/catalogo/208627-0-transporte-ptomedida-historico.dcat"
out_file = MADRID_PATH / 'downloads' / '208627-0-transporte-ptomedida-historico.dcat'
os.system(f"wget -O {out_file} {url}")

download_dcat_zips('208627-0-transporte-ptomedida-historico.dcat', prefix='data', month_names=True,
                   months=['2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12'])

# +
from math import atan2, cos, sin, degrees
from shapely.geometry import LineString, Point, MultiPoint
from shapely.ops import transform
from scipy.spatial import ConvexHull
import pyproj

ETRS89 = pyproj.CRS('EPSG:25830')
WGS84 = pyproj.CRS('EPSG:4326')
project_wgs84 = pyproj.Transformer.from_crs(ETRS89, WGS84, always_xy=True).transform

def get_heading(lon1, lat1, lon2, lat2):
    angle = atan2(lon2-lon1, lat2-lat1)
    angle = degrees(angle)
    if angle < 0:
        angle += 360
    return angle

def convert_locations_shp(zip_file):
    hdiff = 0
    features = []
    zf = zipfile.ZipFile(zip_file)
    with tempfile.TemporaryDirectory() as tempdir:
        zf.extractall(tempdir)
        shp_path = glob.glob(f'{tempdir}/*.shp')[0]
        print(shp_path)
        shapefile = geopandas.read_file(shp_path)
        for index, row in shapefile.iterrows():
            arrow_polygon = transform(project_wgs84, row['geometry'])
            from_point = LineString([arrow_polygon.exterior.coords[2], arrow_polygon.exterior.coords[3]]).centroid
            to_point = Point(arrow_polygon.exterior.coords[6])
            heading = get_heading(from_point.x, from_point.y, to_point.x, to_point.y)
            points = np.array(list(arrow_polygon.exterior.coords))
            hull_points = points[ConvexHull(points).vertices]
            # find shortest segment index
            shortest_segment_len = 1e9
            shortest_segment_idx = 0
            shortest_segment = None
            idxs = list(range(len(hull_points)))
            idxs.append(0)
            for i1, i2 in zip(idxs[:-1], idxs[1:]):
                s = LineString([hull_points[i1], hull_points[i2]])
                if s.length < shortest_segment_len:
                    shortest_segment_len = s.length
                    shortest_segment = s
                    shortest_segment_idx = i2
            from_point = shortest_segment.centroid
            to_idx = shortest_segment_idx
            for i in range(2):
                to_idx += 1
                if to_idx == len(hull_points):
                    to_idx = 0
            to_point = Point(hull_points[to_idx])
            heading_hull = get_heading(from_point.x, from_point.y, to_point.x, to_point.y)
            if abs(heading_hull - heading) > 0.5:
                hdiff += 1
                heading = heading_hull
            features.append(geojson.Feature(geometry=geojson.Point((row['longitud'], row['latitud'])),
            properties={
                "heading": heading, "name": row['nombre'], "tipo_elem": row['tipo_elem'],
                "id": int(row['id']), "distrito": str(row['distrito']), "cod_cent": str(row['cod_cent'])
            }))
    print(f'{hdiff} / {len(features)} headings differ')
    feature_collection = geojson.FeatureCollection(features)
    geojson_name = os.path.splitext(os.path.basename(zip_file))[0]
    with open(MADRID_PATH / 'downloads' / f'{geojson_name}.geojson', 'w') as f:
        geojson.dump(feature_collection, f)
    return feature_collection

def diff_dicts(a, b, drop_similar=True):
    res = a.copy()
    for k in res:
        if k not in b:
            res[k] = (res[k], None)
    for k in b:
        if k in res:
            res[k] = (res[k], b[k])
        else:
            res[k] = (None, b[k])
    if drop_similar:
        res = {k:v for k,v in res.items() if v[0] != v[1]}
    return res

def check_diff(diff, key, tolerance):
    if key not in diff:
        return False
    a, b = diff[key]
    return abs(a-b) > tolerance

def open_locations(locations_file):
    locations_by_id = {}
    with open(locations_file, 'r') as f:
        fc = geojson.load(f)
        for f in fc['features']:
            id = f['properties']['id']
            heading = f['properties']['heading']
            lat = f['geometry']['coordinates'][1]
            lon = f['geometry']['coordinates'][0]
            name = f['properties']['name']
            tipo_elem = f['properties']['tipo_elem']
            distrito = f['properties']['distrito']
            cod_cent = f['properties']['cod_cent']
            yield {"id": id, "name":name, "lat":lat, "lon":lon, "heading":heading,
                   "tipo_elem":tipo_elem, "distrito":distrito, "cod_cent":cod_cent}

def get_merged_locations():
    merged_locations = {}
    for locations_zip in (MADRID_PATH / 'downloads').glob('locations-*.zip'):
        convert_locations_shp(locations_zip)
        month = str(locations_zip)[-11:-4]
        print(month)
        for l in open_locations(str(locations_zip).replace('.zip', '.geojson')):
            id = l["id"]
            if id in merged_locations:
                matches = False
                for i, lo in enumerate(merged_locations[id]):
                    diff = diff_dicts(lo['data'], l)
                    if (check_diff(diff, 'lat', 0.0001) or 
                        check_diff(diff, 'lon', 0.0001) or 
                        check_diff(diff, 'heading', 5.0)):
                        pass
                    else:
                        matches = True
                        break
                if matches:
                    merged_locations[id][i]['months'].append(month)
                else:
                    merged_locations[id].append({'months': [month], 'data': l})
            else:
                merged_locations[id] = [{'months': [month], 'data': l}]
    return merged_locations


merged_locations = get_merged_locations()
len(merged_locations)

# +
features = []
for id, mll in merged_locations.items():
    for ml in mll:
        props = ml['data']
        props['valid_months'] = ','.join(ml['months'])
        pt = geojson.Point([props['lon'], props['lat']])
        features.append(geojson.Feature(geometry=pt, properties=props))

feature_collection = geojson.FeatureCollection(features)
with open(MADRID_PATH / 'downloads' / 'counter_locations_merged.geojson', 'w') as f:
    geojson.dump(feature_collection, f)

# +
import csv
from io import TextIOWrapper

def time_bin_format(time):
    return time.strftime('%Y-%m-%d %H:%M')

def get_locations_by_id_and_month():
    locations_by_id_and_month = {}
    with open(MADRID_PATH / 'downloads' / 'counter_locations_merged.geojson', 'r') as f:
        fc = geojson.load(f)
        for f in fc['features']:
            id = f['properties']['id']
            heading = f['properties']['heading']
            lat = f['geometry']['coordinates'][1]
            lon = f['geometry']['coordinates'][0]
            for valid_month in f['properties']['valid_months'].split(','):
                if id not in locations_by_id_and_month:
                    locations_by_id_and_month[id] = {}
                locations_by_id_and_month[id][valid_month] = (lat, lon, heading)
    return locations_by_id_and_month

def generate_data_by_counter(month, locations_by_id_and_month):
    count = 0
    cf = MADRID_PATH / 'downloads'/ f'data-{month}.zip'
    print(cf)
    cfa = zipfile.ZipFile(cf, 'r')
    f = cfa.open(f'{(datetime.strptime(month, "%Y-%m")).strftime("%m-%Y")}.csv', 'r')
    csvreader = csv.reader(TextIOWrapper(f, 'utf-8'), delimiter=';')
    header = next(csvreader)
    # print(header)
    for row in csvreader:
        # https://datos.madrid.es/FWProjects/egob/Catalogo/Transporte/Trafico/ficheros/PuntosMedidaTraficoMdrid.pdf
        # ['id', 'fecha', 'tipo_elem', 'intensidad', 'ocupacion', 'carga', 'vmed', 'error', 'periodo_integracion']
        # fecha --> collection time --> time bin
        # tipo_elem --> counter type ('URB' or 'M30')
        # intensidad --> number of vehicles in 15 minutes --> volume
        # occupacion --> occupation time in percent [0..100] of 15 minutes
        # carga --> congestion level in percent [0..100]
        # vmed --> average speed (only on M30 counters)
        locid = int(row[0])
        if locid not in locations_by_id_and_month:
            print(f'WARNING: unknown ID {locid}')
            continue
        valid_month = month
        if valid_month not in locations_by_id_and_month[locid]:
            # Try the months before, stupid logic but sufficient here
            while valid_month not in locations_by_id_and_month[locid]:
                valid_month = (pd.to_datetime(valid_month) - pd.Timedelta("1 day")).strftime("%Y-%m")
                if valid_month[:4] == '2018':
                    break
            if valid_month not in locations_by_id_and_month[locid]:
                print(f'WARNING: no valid month {month} for ID {locid}: {locations_by_id_and_month[locid]}')
                continue
        lat, lon, heading = locations_by_id_and_month[locid][valid_month]
        volume = int(row[3])
        occupation = float(row[4])
        congestion_level = float(row[5])
        speed_avg = float(row[6])
        collection_time = row[1]
        time_bin = time_bin_format(datetime.strptime(collection_time, '%Y-%m-%d %H:%M:%S'))
        count += 1
        if count % 1000000 == 0:
            print(count)
        yield {'id': locid, 'lat': lat, 'lon': lon, 'heading': heading, 'time_bin': time_bin, 'type': row[2],
               'volume': volume, 'occupation': occupation, 'congestion_level': congestion_level,
               'speed_avg': speed_avg}

def process_madrid_month(month, output_path):
    locations_by_id_and_month = get_locations_by_id_and_month()
    month_df = pd.DataFrame(generate_data_by_counter(month, locations_by_id_and_month))
    month_df.to_parquet(output_path / 'all' / f'counters_{month}.parquet', compression="snappy")
    return month_df


# madrid_counters_df = process_madrid_month('2021-06', MADRID_PATH)
# madrid_counters_df
# -

# process_madrid_month('2021-06', MADRID_PATH)
process_madrid_month('2021-07', MADRID_PATH)
# process_madrid_month('2021-08', MADRID_PATH)
# process_madrid_month('2021-09', MADRID_PATH)
# process_madrid_month('2021-10', MADRID_PATH)
# process_madrid_month('2021-11', MADRID_PATH)
# process_madrid_month('2021-12', MADRID_PATH)
