#  Copyright 2022 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import os  # This is is needed in the pyqgis console also
from pathlib import Path

from pyqgis_speed_stats_lib import pyqgis_export
from qgis.core import *


def main(historic_uber: bool):
    layer_name = "road graph edges"
    PROJECT_FILE = "road_graph.qgz"

    if not historic_uber:
        STATS_FOLDER = "../figures/speed_stats/"
        RELEASE = "release20221026_residential_unclassified/"
        PATHS = [
            ["2020", "moscow"],
            ["2020", "istanbul"],
            ["2020", "berlin"],
            ["2021", "antwerp"],
            ["2021", "bangkok"],
            ["2021", "barcelona"],
            ["2021", "berlin"],
            ["2021", "chicago"],
            ["2021", "istanbul"],
            ["2021", "melbourne"],
            ["2021", "moscow"],
            ["2022", "london"],
            ["2022", "madrid"],
            ["2022", "melbourne"],
        ]

    else:
        STATS_FOLDER = "../figures/speed_stats_05_val01_uber/"
        RELEASE = "release20221028_historic_uber"
        PATHS = [
            ["", "london"],
            ["", "berlin"],
            ["", "barcelona"],
        ]

    # Supply path to qgis install location
    # QgsApplication.setPrefixPath("/path/to/qgis/installation", True)
    # Create a reference to the QgsApplication.  Setting the
    # second argument to False disables the GUI.
    qgs = QgsApplication([], False)

    # Load providers
    qgs.initQgis()

    for YEAR, CITY in PATHS:

        if YEAR:
            image_location = f"{STATS_FOLDER}/road_graph_{CITY}_{YEAR}.jpg"
            datasource = Path(f"/iarai/public/t4c/data_pipeline/") / RELEASE / YEAR / "road_graph" / CITY / "road_graph_freeflow.gpkg"
        else:
            image_location = f"{STATS_FOLDER}/road_graph_{CITY}.jpg"
            datasource = Path(f"/iarai/public/t4c/data_pipeline/") / RELEASE / "road_graph" / CITY / "road_graph_freeflow.gpkg"

        assert datasource.exists(), datasource
        datasource = str(datasource)

        layout_name = "road_graph"

        pyqgis_export(CITY, PROJECT_FILE, datasource, image_location, layer_name, layout_name)

    # Finally, exitQgis() is called to remove the
    # provider and layer registries from memory
    # del layer
    qgs.exitQgis()


def new_main(PROJECT_FILE, DATA_FOLDER, STATS_FOLDER, CITY, YEAR=None):

    layer_name = "road graph"

    # Supply path to qgis install location
    # QgsApplication.setPrefixPath("/path/to/qgis/installation", True)
    # Create a reference to the QgsApplication.  Setting the
    # second argument to False disables the GUI.
    qgs = QgsApplication([], False)

    # Load providers
    qgs.initQgis()

    if YEAR is not None:
        image_location = f"{STATS_FOLDER}/road_graph_{CITY}_{YEAR}.jpg"
        datasource = DATA_FOLDER / YEAR / "road_graph" / CITY / "road_graph_freeflow.gpkg"
    else:
        image_location = f"{STATS_FOLDER}/road_graph_{CITY}.jpg"
        datasource = DATA_FOLDER / "road_graph" / CITY / "road_graph_freeflow.gpkg"

    assert datasource.exists(), datasource
    datasource = str(datasource)

    layout_name = "road_graph"

    pyqgis_export(CITY, PROJECT_FILE, datasource, image_location, layer_name, layout_name)

    # Finally, exitQgis() is called to remove the
    # provider and layer registries from memory
    # del layer
    qgs.exitQgis()


if __name__ == "__main__":
    try:
        main(historic_uber=False)
    except:
        pass
    main(historic_uber=True)
