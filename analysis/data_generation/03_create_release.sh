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

mkdir -p ${DATA_ROOT}/${RELEASE}/2020/movie
ln -s ${DATA_ROOT}/here_extracted/2020/BERLIN/training ${DATA_ROOT}/${RELEASE}/2020/movie/berlin
ln -s ${DATA_ROOT}/here_extracted/2020/ISTANBUL/training ${DATA_ROOT}/${RELEASE}/2020/movie/istanbul
ln -s ${DATA_ROOT}/here_extracted/2020/MOSCOW/training ${DATA_ROOT}/${RELEASE}/2020/movie/moscow


mkdir -p ${DATA_ROOT}/${RELEASE}/2021/movie
ln -s ${DATA_ROOT}/here_extracted/2021/ANTWERP/training ${DATA_ROOT}/${RELEASE}/2021/movie/antwerp
ln -s ${DATA_ROOT}/here_extracted/2021/BANGKOK/training ${DATA_ROOT}/${RELEASE}/2021/movie/bangkok
ln -s ${DATA_ROOT}/here_extracted/2021/BARCELONA/training ${DATA_ROOT}/${RELEASE}/2021/movie/barcelona
ln -s ${DATA_ROOT}/here_extracted/2021/BERLIN/training ${DATA_ROOT}/${RELEASE}/2021/movie/berlin
ln -s ${DATA_ROOT}/here_extracted/2021/CHICAGO/training ${DATA_ROOT}/${RELEASE}/2021/movie/chicago
ln -s ${DATA_ROOT}/here_extracted/2021/ISTANBUL/training ${DATA_ROOT}/${RELEASE}/2021/movie/istanbul
ln -s ${DATA_ROOT}/here_extracted/2021/MELBOURNE/training ${DATA_ROOT}/${RELEASE}/2021/movie/melbourne
ln -s ${DATA_ROOT}/here_extracted/2021/MOSCOW/training ${DATA_ROOT}/${RELEASE}/2021/movie/moscow


mkdir -p ${DATA_ROOT}/${RELEASE}/2022/movie
ln -s ${DATA_ROOT}/here_extracted/2022/movie/london ${DATA_ROOT}/${RELEASE}/2022/movie/london
ln -s ${DATA_ROOT}/here_extracted/2022/movie/madrid ${DATA_ROOT}/${RELEASE}/2022/movie/madrid
ln -s ${DATA_ROOT}/here_extracted/2022/movie/melbourne ${DATA_ROOT}/${RELEASE}/2022/movie/melbourne
