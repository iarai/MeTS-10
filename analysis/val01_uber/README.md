

```
#--------------------------------------------------------
# uber_00: data download
#--------------------------------------------------------
# Barcelona
# -> Overlap 3 months: 2020-01 -- 2020-03
# OSM data from the range: `
wget "https://download.geofabrik.de/europe/spain-200101.osm.pbf"

### Berlin
# -> Overlap 6 months: 2019-01-01--2019-06-30

OSM data from the range: `
wget "https://download.geofabrik.de/europe/germany-190101.osm.pbf"

### London
# -> Overlap 7 months: 2019-07 -- 2020-01
OSM data from the range:
wget "https://download.geofabrik.de/europe/great-britain/england-200101.osm.pbf"




#--------------------------------------------------------
# uber01
#--------------------------------------------------------
sudo apt install osmctools
osmconvert england-200101.osm.pbf --out-osm -o=england-200101.osm
# "london": {"bounds": [5120500, 5170000, -36900, 6700]},
osmconvert england-200101.osm -b=-1.0208,51.0614,0.7784,52.1066  -o=england-200101-truncated.osm
python uber01_osm_to_parquet_bypassing_osmnx.py --data_folder /iarai/public/t4c/osm/ --place england-200101-truncated


osmconvert germany-190101.osm.pbf --out-osm -o=germany-190101.osm
# "berlin": {"bounds": [5235900, 5285400, 1318900, 1362500]},
osmconvert germany-190101.osm -b=12.5,52.0,14.5,53.0 -o=germany-190101-truncated.osm
python uber01_osm_to_parquet_bypassing_osmnx.py --data_folder /iarai/public/t4c/osm/ --place germany-190101-truncated




osmconvert spain-200101.osm.pbf --out-osm -o=spain-200101.osm
# "barcelona": {"bounds": [4125300, 4174800, 192500, 236100]},
osmconvert spain-200101.osm -b=1.2,40.5,3.0,42.5 -o=spain-200101-truncated.osm
python uber01_osm_to_parquet_bypassing_osmnx.py --data_folder /iarai/public/t4c/osm/ --place spain-200101-truncated


#--------------------------------------------------------
# uber02: historic road graph -> notebooks
#--------------------------------------------------------
mkdir -p /iarai/public/t4c/data_pipeline/release20221028_historic_uber/road_graph/barcelona
mkdir -p /iarai/public/t4c/data_pipeline/release20221028_historic_uber/road_graph/berlin
mkdir -p /iarai/public/t4c/data_pipeline/release20221028_historic_uber/road_graph/london


# run uber02_barcelona_match_uber_with_osm.ipynb
# run uber02_berlin_match_uber_with_osm.ipynb
# run uber02_london_match_uber_with_osm.ipynb




#--------------------------------------------------------
# uber02bis: run pipeline from dp04 on (no need to re-run dp01 and dp02 movie aggregation and speed clustering, and we substitute dp03 by our historic road graph).
#--------------------------------------------------------

mkdir -p /iarai/public/t4c/data_pipeline/release20221028_historic_uber/

mkdir -p mkdir -p /iarai/public/t4c/data_pipeline/release20221028_historic_uber/movie_15min
mkdir -p /iarai/public/t4c/data_pipeline/release20221028_historic_uber/ movie_speed_clusters



ln -s /iarai/public/t4c/data_pipeline/release20220930/2021/movie_15min/barcelona/ /iarai/public/t4c/data_pipeline/release20221028_historic_uber/movie_15min/barcelona
ln -s /iarai/public/t4c/data_pipeline/release20220930/2021/movie_speed_clusters/barcelona /iarai/public/t4c/data_pipeline/release20221028_historic_uber/movie_speed_clusters/barcelona

ln -s /iarai/public/t4c/data_pipeline/release20220930/2021/movie_15min/berlin/ /iarai/public/t4c/data_pipeline/release20221028_historic_uber/movie_15min/berlin
ln -s /iarai/public/t4c/data_pipeline/release20220930/2021/movie_speed_clusters/berlin /iarai/public/t4c/data_pipeline/release20221028_historic_uber/movie_speed_clusters/berlin

ln -s /iarai/public/t4c/data_pipeline/release20220930/2022/movie_15min/london/ /iarai/public/t4c/data_pipeline/release20221028_historic_uber/movie_15min/london
ln -s /iarai/public/t4c/data_pipeline/release20220930/2022/movie_speed_clusters/london /iarai/public/t4c/data_pipeline/release20221028_historic_uber/movie_speed_clusters/london

export PIPELINE_ROOT="....../data_pipeline/"
export PYTHONPATH=${PIPELINE_ROOT}
export DATA_ROOT="/iarai/public/t4c/data_pipeline/"
export RELEASE="release20221028_historic_uber"
export CITY=berlin
python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE} -f --city ${CITY}
python ${PIPELINE_ROOT}/dp05_free_flow.py -d ${DATA_ROOT}/${RELEASE} -f --city ${CITY}
python ${PIPELINE_ROOT}/dp06_speed_classes.py -d ${DATA_ROOT}/${RELEASE} --city ${CITY}
python ${PIPELINE_ROOT}/../t4c22/prepare_training_data_cc.py -d ${DATA_ROOT}/${RELEASE} --city ${CITY}

export CITY=barcelona
python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE} -f --city ${CITY}
python ${PIPELINE_ROOT}/dp05_free_flow.py -d ${DATA_ROOT}/${RELEASE} -f --city ${CITY}
python ${PIPELINE_ROOT}/dp06_speed_classes.py -d ${DATA_ROOT}/${RELEASE} --city ${CITY}
python ${PIPELINE_ROOT}/../t4c22/prepare_training_data_cc.py -d ${DATA_ROOT}/${RELEASE} --city ${CITY}

export CITY=london
python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE} -f --city ${CITY}
python ${PIPELINE_ROOT}/dp05_free_flow.py -d ${DATA_ROOT}/${RELEASE} -f --city ${CITY}
python ${PIPELINE_ROOT}/dp06_speed_classes.py -d ${DATA_ROOT}/${RELEASE} --city ${CITY}
python ${PIPELINE_ROOT}/../t4c22/prepare_training_data_cc.py -d ${DATA_ROOT}/${RELEASE} --city ${CITY}




cd ~/workspaces/MCSWTS/analysis/dataset_description/
export RELEASE="release20221028_historic_uber"
export DATA_ROOT="/iarai/public/t4c/data_pipeline/"
rm -fR speed_stats_05_val01_uber
mkdir speed_stats_05_val01_uber
python speed_stats.py -d ${DATA_ROOT}/${RELEASE} --city barcelona --output_folder speed_stats_05_val01_uber --beyond_bounding_box
python speed_stats.py -d ${DATA_ROOT}/${RELEASE} --city berlin --output_folder speed_stats_05_val01_uber --beyond_bounding_box
python speed_stats.py -d ${DATA_ROOT}/${RELEASE} --city london --output_folder speed_stats_05_val01_uber --beyond_bounding_box
cd -

cd ${DATA_ROOT}
zip -r ${RELEASE}_barcelona.zip ${RELEASE}/road_graph/barcelona  ${RELEASE}/speed_classes/barcelona
zip -r ${RELEASE}_berlin.zip ${RELEASE}/road_graph/berlin  ${RELEASE}/speed_classes/berlin
zip -r ${RELEASE}_london.zip ${RELEASE}/road_graph/london  ${RELEASE}/speed_classes/london
cd -



#--------------------------------------------------------
# uber03: spatial coverage -> notebooks
# uber04: speed differences -> notebooks
#--------------------------------------------------------

```
