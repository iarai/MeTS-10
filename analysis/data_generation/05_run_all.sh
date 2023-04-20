#
# Copyright 2022 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
# IARAI licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
set -euxo pipefail
export PIPELINE_ROOT="YOUR_REPO_LOCATION_SHOULD_GO_HERE/data_pipeline/"
export PYTHONPATH=${PIPELINE_ROOT}
export RELEASE="release20221026_residential_unclassified"
export DATA_ROOT="/iarai/public/t4c/data_pipeline/"

# dp01_movie_aggregation.py
python ${PIPELINE_ROOT}/dp01_movie_aggregation.py -d ${DATA_ROOT}/${RELEASE}/2020 --city berlin
python ${PIPELINE_ROOT}/dp01_movie_aggregation.py -d ${DATA_ROOT}/${RELEASE}/2020 --city istanbul
python ${PIPELINE_ROOT}/dp01_movie_aggregation.py -d ${DATA_ROOT}/${RELEASE}/2020 --city moscow

python ${PIPELINE_ROOT}/dp01_movie_aggregation.py -d ${DATA_ROOT}/${RELEASE}/2021 --city antwerp
python ${PIPELINE_ROOT}/dp01_movie_aggregation.py -d ${DATA_ROOT}/${RELEASE}/2021 --city bangkok
python ${PIPELINE_ROOT}/dp01_movie_aggregation.py -d ${DATA_ROOT}/${RELEASE}/2021 --city barcelona
python ${PIPELINE_ROOT}/dp01_movie_aggregation.py -d ${DATA_ROOT}/${RELEASE}/2021 --city berlin
python ${PIPELINE_ROOT}/dp01_movie_aggregation.py -d ${DATA_ROOT}/${RELEASE}/2021 --city chicago
python ${PIPELINE_ROOT}/dp01_movie_aggregation.py -d ${DATA_ROOT}/${RELEASE}/2021 --city istanbul
python ${PIPELINE_ROOT}/dp01_movie_aggregation.py -d ${DATA_ROOT}/${RELEASE}/2021 --city melbourne
python ${PIPELINE_ROOT}/dp01_movie_aggregation.py -d ${DATA_ROOT}/${RELEASE}/2021 --city moscow

python ${PIPELINE_ROOT}/dp01_movie_aggregation.py -d ${DATA_ROOT}/${RELEASE}/2022 --city london
python ${PIPELINE_ROOT}/dp01_movie_aggregation.py -d ${DATA_ROOT}/${RELEASE}/2022 --city madrid
python ${PIPELINE_ROOT}/dp01_movie_aggregation.py -d ${DATA_ROOT}/${RELEASE}/2022 --city melbourne

echo -e "======\n2020\n======"
ls -a ${DATA_ROOT}/${RELEASE}/2020/movie | grep -v '\.' | xargs -I {} -exec bash -c "echo '------'; echo {}; echo '------'; ls ${DATA_ROOT}/${RELEASE}/2020/movie/{} | wc -l; ls ${DATA_ROOT}/${RELEASE}/2020/movie_15min/{} | wc -l"
echo -e "======\n2021\n======"
ls -a ${DATA_ROOT}/${RELEASE}/2021/movie | grep -v '\.' | xargs -I {} -exec bash -c "echo '------'; echo {}; echo '------'; ls ${DATA_ROOT}/${RELEASE}/2021/movie/{} | wc -l; ls ${DATA_ROOT}/${RELEASE}/2021/movie_15min/{} | wc -l"
echo -e "======\n2022\n======"
ls -a ${DATA_ROOT}/${RELEASE}/2022/movie | grep -v '\.' | xargs -I {} -exec bash -c "echo '------'; echo {}; echo '------'; ls ${DATA_ROOT}/${RELEASE}/2022/movie/{} | wc -l; ls ${DATA_ROOT}/${RELEASE}/2022/movie_15min/{} | wc -l"

# dp02_speed_clusters.py
python ${PIPELINE_ROOT}/dp02_speed_clusters.py -d ${DATA_ROOT}/${RELEASE}/2020 --city berlin
python ${PIPELINE_ROOT}/dp02_speed_clusters.py -d ${DATA_ROOT}/${RELEASE}/2020 --city istanbul
python ${PIPELINE_ROOT}/dp02_speed_clusters.py -d ${DATA_ROOT}/${RELEASE}/2020 --city moscow

python ${PIPELINE_ROOT}/dp02_speed_clusters.py -d ${DATA_ROOT}/${RELEASE}/2021 --city antwerp
python ${PIPELINE_ROOT}/dp02_speed_clusters.py -d ${DATA_ROOT}/${RELEASE}/2021 --city bangkok
python ${PIPELINE_ROOT}/dp02_speed_clusters.py -d ${DATA_ROOT}/${RELEASE}/2021 --city barcelona
python ${PIPELINE_ROOT}/dp02_speed_clusters.py -d ${DATA_ROOT}/${RELEASE}/2021 --city berlin
python ${PIPELINE_ROOT}/dp02_speed_clusters.py -d ${DATA_ROOT}/${RELEASE}/2021 --city chicago
python ${PIPELINE_ROOT}/dp02_speed_clusters.py -d ${DATA_ROOT}/${RELEASE}/2021 --city istanbul
python ${PIPELINE_ROOT}/dp02_speed_clusters.py -d ${DATA_ROOT}/${RELEASE}/2021 --city melbourne
python ${PIPELINE_ROOT}/dp02_speed_clusters.py -d ${DATA_ROOT}/${RELEASE}/2021 --city moscow

python ${PIPELINE_ROOT}/dp02_speed_clusters.py -d ${DATA_ROOT}/${RELEASE}/2022 --city london
python ${PIPELINE_ROOT}/dp02_speed_clusters.py -d ${DATA_ROOT}/${RELEASE}/2022 --city madrid
python ${PIPELINE_ROOT}/dp02_speed_clusters.py -d ${DATA_ROOT}/${RELEASE}/2022 --city melbourne

echo -e "======\n2020\n======"
ls -a ${DATA_ROOT}/${RELEASE}/2020/movie | grep -v '\.' | xargs -I {} -exec bash -c "echo '------'; echo {}; echo '------'; ls ${DATA_ROOT}/${RELEASE}/2020/movie/{} | wc -l; ls ${DATA_ROOT}/${RELEASE}/2020/movie_speed_clusters/{} | wc -l"
echo -e "======\n2021\n======"
ls -a ${DATA_ROOT}/${RELEASE}/2021/movie | grep -v '\.' | xargs -I {} -exec bash -c "echo '------'; echo {}; echo '------'; ls ${DATA_ROOT}/${RELEASE}/2021/movie/{} | wc -l; ls ${DATA_ROOT}/${RELEASE}/2021/movie_speed_clusters/{} | wc -l"
echo -e "======\n2022\n======"
ls -a ${DATA_ROOT}/${RELEASE}/2022/movie | grep -v '\.' | xargs -I {} -exec bash -c "echo '------'; echo {}; echo '------'; ls ${DATA_ROOT}/${RELEASE}/2022/movie/{} | wc -l; ls ${DATA_ROOT}/${RELEASE}/2022/movie_speed_clusters/{} | wc -l"

# dp03_road_graph.py
python ${PIPELINE_ROOT}/dp03_road_graph.py -d ${DATA_ROOT}/${RELEASE}/2020 -f --city berlin
python ${PIPELINE_ROOT}/dp03_road_graph.py -d ${DATA_ROOT}/${RELEASE}/2020 -f --city istanbul
python ${PIPELINE_ROOT}/dp03_road_graph.py -d ${DATA_ROOT}/${RELEASE}/2020 -f --city moscow

python ${PIPELINE_ROOT}/dp03_road_graph.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city antwerp
python ${PIPELINE_ROOT}/dp03_road_graph.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city bangkok
python ${PIPELINE_ROOT}/dp03_road_graph.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city barcelona
python ${PIPELINE_ROOT}/dp03_road_graph.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city berlin
python ${PIPELINE_ROOT}/dp03_road_graph.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city chicago
python ${PIPELINE_ROOT}/dp03_road_graph.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city istanbul
python ${PIPELINE_ROOT}/dp03_road_graph.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city london
python ${PIPELINE_ROOT}/dp03_road_graph.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city madrid
python ${PIPELINE_ROOT}/dp03_road_graph.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city melbourne
python ${PIPELINE_ROOT}/dp03_road_graph.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city moscow
python ${PIPELINE_ROOT}/dp03_road_graph.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city newyork
python ${PIPELINE_ROOT}/dp03_road_graph.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city vienna
python ${PIPELINE_ROOT}/dp03_road_graph.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city warsaw
python ${PIPELINE_ROOT}/dp03_road_graph.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city zurich

python ${PIPELINE_ROOT}/dp03_road_graph.py -d ${DATA_ROOT}/${RELEASE}/2022 -f --city london
python ${PIPELINE_ROOT}/dp03_road_graph.py -d ${DATA_ROOT}/${RELEASE}/2022 -f --city madrid
python ${PIPELINE_ROOT}/dp03_road_graph.py -d ${DATA_ROOT}/${RELEASE}/2022 -f --city melbourne

echo "expected: 20"
echo "found:"
find ${DATA_ROOT}/${RELEASE} -name "road_graph_edges.parquet" | wc -l

# dp04_intersecting_cells.py
python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE}/2020 -f --city berlin
python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE}/2020 -f --city istanbul
python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE}/2020 -f --city moscow

python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city antwerp
python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city bangkok
python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city barcelona
python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city berlin
python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city chicago
python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city istanbul
python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city london
python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city madrid
python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city melbourne
python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city moscow
python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city newyork
python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city vienna
python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city warsaw
python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE}/2021 -f --city zurich

python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE}/2022 -f --city london
python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE}/2022 -f --city madrid
python ${PIPELINE_ROOT}/dp04_intersecting_cells.py -d ${DATA_ROOT}/${RELEASE}/2022 -f --city melbourne

echo "expected: 20"
echo "found:"
find ${DATA_ROOT}/${RELEASE} -name "road_graph_edges.parquet" | wc -l

# dp05_free_flow.py
python ${PIPELINE_ROOT}/dp05_free_flow.py -d ${DATA_ROOT}/${RELEASE}/2020 --city berlin -f
python ${PIPELINE_ROOT}/dp05_free_flow.py -d ${DATA_ROOT}/${RELEASE}/2020 --city istanbul -f
python ${PIPELINE_ROOT}/dp05_free_flow.py -d ${DATA_ROOT}/${RELEASE}/2020 --city moscow -f

python ${PIPELINE_ROOT}/dp05_free_flow.py -d ${DATA_ROOT}/${RELEASE}/2021 --city antwerp -f
python ${PIPELINE_ROOT}/dp05_free_flow.py -d ${DATA_ROOT}/${RELEASE}/2021 --city bangkok -f
python ${PIPELINE_ROOT}/dp05_free_flow.py -d ${DATA_ROOT}/${RELEASE}/2021 --city barcelona -f
python ${PIPELINE_ROOT}/dp05_free_flow.py -d ${DATA_ROOT}/${RELEASE}/2021 --city berlin -f
python ${PIPELINE_ROOT}/dp05_free_flow.py -d ${DATA_ROOT}/${RELEASE}/2021 --city chicago -f
python ${PIPELINE_ROOT}/dp05_free_flow.py -d ${DATA_ROOT}/${RELEASE}/2021 --city istanbul -f
python ${PIPELINE_ROOT}/dp05_free_flow.py -d ${DATA_ROOT}/${RELEASE}/2021 --city melbourne -f
python ${PIPELINE_ROOT}/dp05_free_flow.py -d ${DATA_ROOT}/${RELEASE}/2021 --city moscow -f

python ${PIPELINE_ROOT}/dp05_free_flow.py -d ${DATA_ROOT}/${RELEASE}/2022 --city london -f
python ${PIPELINE_ROOT}/dp05_free_flow.py -d ${DATA_ROOT}/${RELEASE}/2022 --city madrid -f
python ${PIPELINE_ROOT}/dp05_free_flow.py -d ${DATA_ROOT}/${RELEASE}/2022 --city melbourne -f

echo "expected: 14"
echo "found:"
find ${DATA_ROOT}/${RELEASE} -name "road_graph_freeflow.parquet" | wc -l

# dp06_speed_classes.py
python ${PIPELINE_ROOT}/dp06_speed_classes.py -d ${DATA_ROOT}/${RELEASE}/2020 --city berlin
python ${PIPELINE_ROOT}/dp06_speed_classes.py -d ${DATA_ROOT}/${RELEASE}/2020 --city istanbul
python ${PIPELINE_ROOT}/dp06_speed_classes.py -d ${DATA_ROOT}/${RELEASE}/2020 --city moscow

python ${PIPELINE_ROOT}/dp06_speed_classes.py -d ${DATA_ROOT}/${RELEASE}/2021 --city antwerp
python ${PIPELINE_ROOT}/dp06_speed_classes.py -d ${DATA_ROOT}/${RELEASE}/2021 --city bangkok
python ${PIPELINE_ROOT}/dp06_speed_classes.py -d ${DATA_ROOT}/${RELEASE}/2021 --city barcelona
python ${PIPELINE_ROOT}/dp06_speed_classes.py -d ${DATA_ROOT}/${RELEASE}/2021 --city berlin
python ${PIPELINE_ROOT}/dp06_speed_classes.py -d ${DATA_ROOT}/${RELEASE}/2021 --city chicago
python ${PIPELINE_ROOT}/dp06_speed_classes.py -d ${DATA_ROOT}/${RELEASE}/2021 --city istanbul
python ${PIPELINE_ROOT}/dp06_speed_classes.py -d ${DATA_ROOT}/${RELEASE}/2021 --city melbourne
python ${PIPELINE_ROOT}/dp06_speed_classes.py -d ${DATA_ROOT}/${RELEASE}/2021 --city moscow

python ${PIPELINE_ROOT}/dp06_speed_classes.py -d ${DATA_ROOT}/${RELEASE}/2022 --city london
python ${PIPELINE_ROOT}/dp06_speed_classes.py -d ${DATA_ROOT}/${RELEASE}/2022 --city madrid
python ${PIPELINE_ROOT}/dp06_speed_classes.py -d ${DATA_ROOT}/${RELEASE}/2022 --city melbourne

ls -a ${DATA_ROOT}/${RELEASE}/2020/movie_15min | grep -v '\.' | xargs -I {} -exec bash -c "echo '------'; echo {}; echo '------'; ls ${DATA_ROOT}/${RELEASE}/2020/movie_15min/{} | wc -l; ls ${DATA_ROOT}/${RELEASE}/2020/speed_classes/{} | wc -l"
ls -a ${DATA_ROOT}/${RELEASE}/2021/movie_15min | grep -v '\.' | xargs -I {} -exec bash -c "echo '------'; echo {}; echo '------'; ls ${DATA_ROOT}/${RELEASE}/2021/movie_15min/{} | wc -l; ls ${DATA_ROOT}/${RELEASE}/2021/speed_classes/{} | wc -l"
ls -a ${DATA_ROOT}/${RELEASE}/2022/movie_15min | grep -v '\.' | xargs -I {} -exec bash -c "echo '------'; echo {}; echo '------'; ls ${DATA_ROOT}/${RELEASE}/2022/movie_15min/{} | wc -l; ls ${DATA_ROOT}/${RELEASE}/2022/speed_classes/{} | wc -l"

# "dp07": prepare_training_data_cc.py
python ${PIPELINE_ROOT}/../t4c22/prepare_training_data_cc.py -d ${DATA_ROOT}/${RELEASE}/2020 --city berlin
python ${PIPELINE_ROOT}/../t4c22/prepare_training_data_cc.py -d ${DATA_ROOT}/${RELEASE}/2020 --city istanbul
python ${PIPELINE_ROOT}/../t4c22/prepare_training_data_cc.py -d ${DATA_ROOT}/${RELEASE}/2020 --city moscow

python ${PIPELINE_ROOT}/../t4c22/prepare_training_data_cc.py -d ${DATA_ROOT}/${RELEASE}/2021 --city antwerp
python ${PIPELINE_ROOT}/../t4c22/prepare_training_data_cc.py -d ${DATA_ROOT}/${RELEASE}/2021 --city bangkok
python ${PIPELINE_ROOT}/../t4c22/prepare_training_data_cc.py -d ${DATA_ROOT}/${RELEASE}/2021 --city barcelona
python ${PIPELINE_ROOT}/../t4c22/prepare_training_data_cc.py -d ${DATA_ROOT}/${RELEASE}/2021 --city berlin
python ${PIPELINE_ROOT}/../t4c22/prepare_training_data_cc.py -d ${DATA_ROOT}/${RELEASE}/2021 --city chicago
python ${PIPELINE_ROOT}/../t4c22/prepare_training_data_cc.py -d ${DATA_ROOT}/${RELEASE}/2021 --city istanbul
python ${PIPELINE_ROOT}/../t4c22/prepare_training_data_cc.py -d ${DATA_ROOT}/${RELEASE}/2021 --city melbourne
python ${PIPELINE_ROOT}/../t4c22/prepare_training_data_cc.py -d ${DATA_ROOT}/${RELEASE}/2021 --city moscow

python ${PIPELINE_ROOT}/../t4c22/prepare_training_data_cc.py -d ${DATA_ROOT}/${RELEASE}/2022 --city london
python ${PIPELINE_ROOT}/../t4c22/prepare_training_data_cc.py -d ${DATA_ROOT}/${RELEASE}/2022 --city madrid
python ${PIPELINE_ROOT}/../t4c22/prepare_training_data_cc.py -d ${DATA_ROOT}/${RELEASE}/2022 --city melbourne

ls -a ${DATA_ROOT}/${RELEASE}/2020/movie_15min | grep -v '\.' | xargs -I {} -exec bash -c "echo '------'; echo {}; echo '------'; ls ${DATA_ROOT}/${RELEASE}/2020/movie_15min/{} | wc -l; ls ${DATA_ROOT}/${RELEASE}/2020/train/{}/labels | wc -l"
ls -a ${DATA_ROOT}/${RELEASE}/2021/movie_15min | grep -v '\.' | xargs -I {} -exec bash -c "echo '------'; echo {}; echo '------'; ls ${DATA_ROOT}/${RELEASE}/2021/movie_15min/{} | wc -l; ls ${DATA_ROOT}/${RELEASE}/2021/train/{}/labels | wc -l"
ls -a ${DATA_ROOT}/${RELEASE}/2022/movie_15min | grep -v '\.' | xargs -I {} -exec bash -c "echo '------'; echo {}; echo '------'; ls ${DATA_ROOT}/${RELEASE}/2022/movie_15min/{} | wc -l; ls ${DATA_ROOT}/${RELEASE}/2022/train/{}/labels | wc -l"

# "dp08": zipping
cd ${DATA_ROOT}
zip -r ${RELEASE}_berlin_2020.zip ${RELEASE}/2020/road_graph/berlin ${RELEASE}/2020/speed_classes/berlin
zip -r ${RELEASE}_istanbul_2020.zip ${RELEASE}/2020/road_graph/istanbul ${RELEASE}/2020/speed_classes/istanbul
zip -r ${RELEASE}_moscow_2020.zip ${RELEASE}/2020/road_graph/moscow ${RELEASE}/2020/speed_classes/moscow

zip -r ${RELEASE}_antwerp.zip ${RELEASE}/2021/road_graph/antwerp ${RELEASE}/2021/speed_classes/antwerp
zip -r ${RELEASE}_bangkok.zip ${RELEASE}/2021/road_graph/bangkok ${RELEASE}/2021/speed_classes/bangkok
zip -r ${RELEASE}_barcelona.zip ${RELEASE}/2021/road_graph/barcelona ${RELEASE}/2021/speed_classes/barcelona
zip -r ${RELEASE}_berlin_2021.zip ${RELEASE}/2021/road_graph/berlin ${RELEASE}/2021/speed_classes/berlin
zip -r ${RELEASE}_chicago.zip ${RELEASE}/2021/road_graph/chicago ${RELEASE}/2021/speed_classes/chicago
zip -r ${RELEASE}_istanbul_2021.zip ${RELEASE}/2021/road_graph/istanbul ${RELEASE}/2021/speed_classes/istanbul
zip -r ${RELEASE}_melbourne_2021.zip ${RELEASE}/2021/road_graph/melbourne ${RELEASE}/2021/speed_classes/melbourne
zip -r ${RELEASE}_moscow_2021.zip ${RELEASE}/2021/road_graph/moscow ${RELEASE}/2021/speed_classes/moscow

zip -r ${RELEASE}_london.zip ${RELEASE}/2022/road_graph/london ${RELEASE}/2022/speed_classes/london
zip -r ${RELEASE}_madrid.zip ${RELEASE}/2022/road_graph/madrid ${RELEASE}/2022/speed_classes/madrid
zip -r ${RELEASE}_melbourne_2022.zip ${RELEASE}/2022/road_graph/melbourne ${RELEASE}/2022/speed_classes/melbourne
cd -

# "dp09": speed stats

cd ~/workspaces/MeTS-10/analysis/dataset_description/
export RELEASE=release20221026_residential_unclassified
rm -fR figures/speed_stats
mkdir -p figures/speed_stats
#python speed_stats.py -d ${DATA_ROOT}/${RELEASE}/2020 --city berlin --output_folder figures/speed_stats
#python speed_stats.py -d ${DATA_ROOT}/${RELEASE}/2020 --city istanbul --output_folder figures/speed_stats
#python speed_stats.py -d ${DATA_ROOT}/${RELEASE}/2020 --city moscow --output_folder figures/speed_stats

python speed_stats.py -d ${DATA_ROOT}/${RELEASE}/2021 --city antwerp --output_folder figures/speed_stats
python speed_stats.py -d ${DATA_ROOT}/${RELEASE}/2021 --city bangkok --output_folder figures/speed_stats
python speed_stats.py -d ${DATA_ROOT}/${RELEASE}/2021 --city barcelona --output_folder figures/speed_stats
python speed_stats.py -d ${DATA_ROOT}/${RELEASE}/2021 --city berlin --output_folder figures/speed_stats
python speed_stats.py -d ${DATA_ROOT}/${RELEASE}/2021 --city chicago --output_folder figures/speed_stats
python speed_stats.py -d ${DATA_ROOT}/${RELEASE}/2021 --city istanbul --output_folder figures/speed_stats
python speed_stats.py -d ${DATA_ROOT}/${RELEASE}/2021 --city melbourne --output_folder figures/speed_stats
python speed_stats.py -d ${DATA_ROOT}/${RELEASE}/2021 --city moscow --output_folder figures/speed_stats

python speed_stats.py -d ${DATA_ROOT}/${RELEASE}/2022 --city london --output_folder figures/speed_stats
python speed_stats.py -d ${DATA_ROOT}/${RELEASE}/2022 --city madrid --output_folder figures/speed_stats
python speed_stats.py -d ${DATA_ROOT}/${RELEASE}/2022 --city melbourne --output_folder figures/speed_stats
cd -

cd ~/workspaces/MeTS-10/analysis/dataset_description/
export RELEASE="release20221028_historic_uber"
export DATA_ROOT="/iarai/public/t4c/data_pipeline/"
rm -fR speed_stats_05_val01_uber
mkdir speed_stats_05_val01_uber
python speed_stats.py -d ${DATA_ROOT}/${RELEASE} --city barcelona --output_folder speed_stats_05_val01_uber --beyond_bounding_box
python speed_stats.py -d ${DATA_ROOT}/${RELEASE} --city berlin --output_folder speed_stats_05_val01_uber --beyond_bounding_box
python speed_stats.py -d ${DATA_ROOT}/${RELEASE} --city london --output_folder speed_stats_05_val01_uber --beyond_bounding_box
cd -

cd ${DATA_ROOT}
zip -r ${RELEASE}_barcelona.zip ${RELEASE}/road_graph/barcelona ${RELEASE}/speed_classes/barcelona
zip -r ${RELEASE}_berlin.zip ${RELEASE}/road_graph/berlin ${RELEASE}/speed_classes/berlin
zip -r ${RELEASE}_london.zip ${RELEASE}/road_graph/london ${RELEASE}/speed_classes/london
cd -

# TODO "dp10": pyqgis stuff
