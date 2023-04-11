from pathlib import Path

import geopandas as gpd
from shapely import LineString

from data_pipeline.data_helpers import get_bearing

MARGIN = 10


def main(DATA_ROOT: Path, RELEASE: str, YEAR: str, CITY: str):
    print(f"{DATA_ROOT}/{RELEASE}/{YEAR}/{CITY}")

    gdf_edges = gpd.read_parquet(DATA_ROOT / RELEASE / YEAR / "road_graph" / CITY / "road_graph_edges.parquet")
    num_diagonals = 0
    num_horizontal_verticals = 0

    diagonal = []
    horizontal_vertical = []

    for g in gdf_edges["geometry"]:
        g: LineString = g
        coordinates = list(g.coords)
        orientations = set()
        for (lon_from, lat_from), (lon_to, lat_to) in zip(coordinates, coordinates[1:]):
            alpha = get_bearing(lat_from, lon_from, lat_to, lon_to)
            assert 0 <= alpha < 360, (alpha, coordinates)
            # clock-wise
            for b, h in zip([0, 90, 180, 270, 45, 90 + 45, 180 + 45, 270 + 45], ["N", "E", "S", "W", "NE", "SE", "SW", "NW"]):
                if (b - MARGIN) % 360 <= alpha < (b + MARGIN) % 360:
                    orientations.add(h)

        if len(orientations) == 1:
            orientation = list(orientations)[0]
            if orientation in [
                "N",
                "E",
                "S",
                "W",
            ]:
                num_horizontal_verticals += 1
                diagonal.append(False)
                horizontal_vertical.append(True)
            else:
                num_diagonals += 1
                diagonal.append(True)
                horizontal_vertical.append(False)
        else:
            diagonal.append(False)
            horizontal_vertical.append(False)
    print(f"num_diagonals={num_diagonals} ({num_diagonals / len(gdf_edges) * 100:.2f}%)")
    print(f"num_horizontal_verticals={num_horizontal_verticals} ({num_horizontal_verticals / len(gdf_edges) * 100:.2f}%)")
    gdf_edges["diagonal"] = diagonal
    gdf_edges["horizontal_vertical"] = horizontal_vertical

    # TODO merge with data (1): coverage
    # TODO merge with data (2): mapped volume


if __name__ == "__main__":

    DATA_ROOT = Path("/iarai/public/t4c/data_pipeline/")
    RELEASE = "release20221026_residential_unclassified"

    d = {
        "2021": [
            "antwerp",
            "bangkok",
            "barcelona",
            "berlin",
            "chicago",
            "istanbul",
            "melbourne",
            "moscow",
        ],
        "2022": ["london", "madrid", "melbourne"],
    }
    for YEAR, CITIES in d.items():
        for CITY in CITIES:
            main(DATA_ROOT=DATA_ROOT, RELEASE=RELEASE, YEAR=YEAR, CITY=CITY)

# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/bangkok
# num_diagonals=91609 (13.18%)
# num_horizontal_verticals=201316 (28.97%)
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/barcelona
# num_diagonals=34724 (29.23%)
# num_horizontal_verticals=22003 (18.52%)
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/berlin
# num_diagonals=19940 (22.43%)
# num_horizontal_verticals=20875 (23.49%)
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/chicago
# num_diagonals=9399 (5.01%)
# num_horizontal_verticals=118970 (63.43%)
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/istanbul
# num_diagonals=61564 (22.79%)
# num_horizontal_verticals=61255 (22.68%)
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/melbourne
# num_diagonals=31915 (13.84%)
# num_horizontal_verticals=103833 (45.02%)
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/moscow
# num_diagonals=13259 (27.69%)
# num_horizontal_verticals=10177 (21.26%)
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2022/london
# num_diagonals=65380 (24.12%)
# num_horizontal_verticals=59654 (22.01%)
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2022/madrid
# num_diagonals=35551 (24.79%)
# num_horizontal_verticals=32018 (22.33%)
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2022/melbourne
# num_diagonals=31915 (13.84%)
# num_horizontal_verticals=103833 (45.02%)
