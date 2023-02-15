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

export OLD_RELEASE="release20220930"
export RELEASE="release20221026_residential_unclassified"
export DATA_ROOT="/iarai/public/t4c/data_pipeline/"

ln -s ${DATA_ROOT}/${OLD_RELEASE}/2020/movie_15min ${DATA_ROOT}/${RELEASE}/2020/movie_15min
ln -s ${DATA_ROOT}/${OLD_RELEASE}/2021/movie_15min ${DATA_ROOT}/${RELEASE}/2021/movie_15min
ln -s ${DATA_ROOT}/${OLD_RELEASE}/2022/movie_15min ${DATA_ROOT}/${RELEASE}/2022/movie_15min
ln -s ${DATA_ROOT}/${OLD_RELEASE}/2020/movie_speed_clusters ${DATA_ROOT}/${RELEASE}/2020/movie_speed_clusters
ln -s ${DATA_ROOT}/${OLD_RELEASE}/2021/movie_speed_clusters ${DATA_ROOT}/${RELEASE}/2021/movie_speed_clusters
ln -s ${DATA_ROOT}/${OLD_RELEASE}/2022/movie_speed_clusters ${DATA_ROOT}/${RELEASE}/2022/movie_speed_clusters
