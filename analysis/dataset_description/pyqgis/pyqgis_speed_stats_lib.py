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
from time import sleep

from qgis.core import QgsLayoutExporter
from qgis.core import QgsLayoutPoint
from qgis.core import QgsProject
from qgis.core import QgsRectangle


T4C_BBOXES = {
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
}

SCALEBAR_POSITION = {True: QgsLayoutPoint(8.32427, 172.579, 0), False: QgsLayoutPoint(8.32427, 217.579, 0)}
COPYRIGHT_POSITION = {True: QgsLayoutPoint(219.142, 172.579, 0), False: QgsLayoutPoint(219.142, 217.579, 0)}
COMPASS_POSITION = {True: QgsLayoutPoint(194.142, 162.579, 0), False: QgsLayoutPoint(189.142, 207.579, 0)}


def pyqgis_export(CITY, PROJECT_FILE, datasource, image_location, layer_name, layout_name):
    print(f"CITY={CITY}")
    BBOX = T4C_BBOXES[CITY]["bounds"]
    rotate = T4C_BBOXES[CITY].get("rotate", False)

    # https://docs.qgis.org/testing/en/docs/pyqgis_developer_cookbook/intro.html#using-pyqgis-in-standalone-scripts
    # https://gis.stackexchange.com/questions/393816/automating-print-layout-with-export-map-canvas-as-image-using-pyqgis
    project = QgsProject.instance()
    print(f"PROJECT_FILE={PROJECT_FILE}")
    project.read(PROJECT_FILE)
    layout = project.layoutManager().layoutByName(layout_name)
    layer = QgsProject.instance().mapLayersByName(layer_name)[0]
    # https://docs.qgis.org/3.22/en/docs/pyqgis_developer_cookbook/loadlayer.html
    print(f"datasource={datasource}")
    layer.setDataSource(datasource, layer_name, "ogr")
    layer.reload()
    layer.triggerRepaint()
    sleep(1)

    y_min, y_max, x_min, x_max = [c / 1e5 for c in BBOX]
    y_min -= 0.01
    y_max += 0.01
    x_min -= 0.01
    x_max += 0.2
    layout.referenceMap().setExtent(QgsRectangle(x_min, y_min, x_max, y_max))

    # https://gis.stackexchange.com/questions/340302/finding-x-y-position-of-layout-element-using-pyqgis
    scalebar = layout.itemById("scalebar")
    scalebar.attemptMove(SCALEBAR_POSITION[rotate])

    copyright = layout.itemById("copyright")
    copyright.attemptMove(COPYRIGHT_POSITION[rotate])

    compass = layout.itemById("compass")
    compass.attemptMove(COMPASS_POSITION[rotate])

    exporter = QgsLayoutExporter(layout)
    settings = QgsLayoutExporter.ImageExportSettings()
    settings.cropToContents = True
    settings.dpi = 150
    print(f"image_location={image_location}")
    result = exporter.exportToImage(image_location, settings)
    print(f"result {result}")  # 0 = Export was successful!
