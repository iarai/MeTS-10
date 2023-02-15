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
export DATA_ROOT="YOUR_LOCATION_SHOULD_GO_HERE"

# extract here_downloads/YEAR -> here_extracted/YEAR
mkdir -p ${DATA_ROOT}/here_extracted/2020
tar xf ${DATA_ROOT}/here_downloads/2020/BERLIN.tar -C ${DATA_ROOT}/here_extracted/2020
tar xf ${DATA_ROOT}/here_downloads/2020/ISTANBUL.tar -C ${DATA_ROOT}/here_extracted/2020
tar xf ${DATA_ROOT}/here_downloads/2020/MOSCOW.tar -C ${DATA_ROOT}/here_extracted/2020


mkdir -p ${DATA_ROOT}/here_extracted/2021
tar zxf ${DATA_ROOT}/here_downloads/2021/ANTWERP.tar.gz -C ${DATA_ROOT}/here_extracted/2021
tar zxf ${DATA_ROOT}/here_downloads/2021/BANGKOK.tar.gz -C ${DATA_ROOT}/here_extracted/2021
tar zxf ${DATA_ROOT}/here_downloads/2021/BARCELONA.tar.gz -C ${DATA_ROOT}/here_extracted/2021
tar zxf ${DATA_ROOT}/here_downloads/2021/BERLIN.tar.gz -C ${DATA_ROOT}/here_extracted/2021
tar zxf ${DATA_ROOT}/here_downloads/2021/CHICAGO.tar.gz -C ${DATA_ROOT}/here_extracted/2021
tar zxf ${DATA_ROOT}/here_downloads/2021/ISTANBUL.tar.gz -C ${DATA_ROOT}/here_extracted/2021
tar zxf ${DATA_ROOT}/here_downloads/2021/MELBOURNE.tar.gz -C ${DATA_ROOT}/here_extracted/2021
tar zxf ${DATA_ROOT}/here_downloads/2021/MOSCOW.tar.gz -C ${DATA_ROOT}/here_extracted/2021
tar zxf ${DATA_ROOT}/here_downloads/2021/NEWYORK.tar.gz -C ${DATA_ROOT}/here_extracted/2021
tar zxf ${DATA_ROOT}/here_downloads/2021/VIENNA.tar.gz -C ${DATA_ROOT}/here_extracted/2021


mkdir -p ${DATA_ROOT}/here_extracted/2022/
unzip ${DATA_ROOT}/here_downloads/2022/LONDON_2022.zip -d ${DATA_ROOT}/here_extracted/2022/
unzip ${DATA_ROOT}/here_downloads/2022/MADRID_2022.zip -d ${DATA_ROOT}/here_extracted/2022/
unzip ${DATA_ROOT}/here_downloads/2022/MELBOURNE_2022.zip -d ${DATA_ROOT}/here_extracted/2022/
