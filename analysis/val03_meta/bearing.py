from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import LineString

from data_pipeline.data_helpers import get_bearing

MARGIN = 10


def main(DATA_ROOT: Path, RELEASE: str, YEAR: str, CITY: str, NUM_SPEED_FILES: int = 10):
    print(f"{DATA_ROOT}/{RELEASE}/{YEAR}/{CITY}")

    gdf_edges = gpd.read_parquet(DATA_ROOT / RELEASE / YEAR / "road_graph" / CITY / "road_graph_edges.parquet")

    # classify diagonals and horizonal/vertical segments
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

    # read speed files
    speed_files = list((DATA_ROOT / RELEASE / YEAR / "speed_classes" / CITY).rglob("speed_classes_*.parquet"))
    dfs = []
    for f in speed_files[:NUM_SPEED_FILES]:
        df = pd.read_parquet(f)
        dfs.append(df)
    df_speeds = pd.concat(dfs)
    df_speeds = df_speeds.merge(
        gdf_edges,
        on=[
            "u",
            "v",
            # TODO
            # "gkey"
        ],
    )

    # coverage
    num_data_points_diagonal = len(df_speeds[df_speeds["diagonal"]])
    num_data_points_horizontal_vertical = len(df_speeds[df_speeds["horizontal_vertical"]])

    # TODO output table, also output relative difference
    # - num edges
    # - num diagonals/vergicals
    # - coverage overall
    # - coverage diagonals/verticals
    print(f"coverage diagonal {num_data_points_diagonal / (num_diagonals * NUM_SPEED_FILES * 24 * 4)}")
    print(f"coverage horizontal_vertical {num_data_points_horizontal_vertical / (num_horizontal_verticals * NUM_SPEED_FILES * 24 * 4)}")

    # TODO mapped volume
    # mapped volume


if __name__ == "__main__":

    DATA_ROOT = Path("/iarai/public/t4c/data_pipeline/")
    RELEASE = "release20221026_residential_unclassified"
    # TODO remove
    # RELEASE = "release20220930" #noqa

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

# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/antwerp
# num_diagonals=21443 (26.26%)
# num_horizontal_verticals=17349 (21.24%)
# coverage diagonal 0.11326156360272972
# coverage horizontal_vertical 0.1594839159797875
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/bangkok
# num_diagonals=91609 (13.18%)
# num_horizontal_verticals=201316 (28.97%)
# coverage diagonal 0.030841936472762864
# coverage horizontal_vertical 0.038412516474928306
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/barcelona
# num_diagonals=34724 (29.23%)
# num_horizontal_verticals=22003 (18.52%)
# coverage diagonal 0.06395180566754982
# coverage horizontal_vertical 0.06724788475813905
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/berlin
# num_diagonals=19940 (22.43%)
# num_horizontal_verticals=20875 (23.49%)
# coverage diagonal 0.2854005244901371
# coverage horizontal_vertical 0.36870773453093814
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/chicago
# num_diagonals=9399 (5.01%)
# num_horizontal_verticals=118970 (63.43%)
# coverage diagonal 0.09626400858247332
# coverage horizontal_vertical 0.07043226933960943
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/istanbul
# num_diagonals=61564 (22.79%)
# num_horizontal_verticals=61255 (22.68%)
# coverage diagonal 0.48294261527299504
# coverage horizontal_vertical 0.6132773004924768
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/melbourne
# num_diagonals=31915 (13.84%)
# num_horizontal_verticals=103833 (45.02%)
# coverage diagonal 0.042159773356311034
# coverage horizontal_vertical 0.05492624976324161
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/moscow
# num_diagonals=13259 (27.69%)
# num_horizontal_verticals=10177 (21.26%)
# coverage diagonal 0.6729685968021721
# coverage horizontal_vertical 0.7120562911467033
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2022/london
# num_diagonals=65380 (24.12%)
# num_horizontal_verticals=59654 (22.01%)
# coverage diagonal 0.14509804858774344
# coverage horizontal_vertical 0.19085137487287804
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2022/madrid
# num_diagonals=35551 (24.79%)
# num_horizontal_verticals=32018 (22.33%)
# coverage diagonal 0.32089284291112297
# coverage horizontal_vertical 0.3926448273887605
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2022/melbourne
# num_diagonals=31915 (13.84%)
# num_horizontal_verticals=103833 (45.02%)
# coverage diagonal 0.04393653062823124
# coverage horizontal_vertical 0.05975265971961387
