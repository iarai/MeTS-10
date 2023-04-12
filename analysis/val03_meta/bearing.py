from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import LineString

from data_pipeline.data_helpers import get_bearing

MARGIN = 10


def main(DATA_ROOT: Path, RELEASE: str, YEAR: str, CITY: str, NUM_SPEED_FILES: int = 10):

    gdf_edges = gpd.read_parquet(DATA_ROOT / RELEASE / YEAR / "road_graph" / CITY / "road_graph_edges.parquet")

    # classify diagonals and horizonal/vertical segments
    num_diagonal = 0
    num_horizontal_vertical = 0
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
                num_horizontal_vertical += 1
                diagonal.append(False)
                horizontal_vertical.append(True)
            else:
                num_diagonal += 1
                diagonal.append(True)
                horizontal_vertical.append(False)
        else:
            diagonal.append(False)
            horizontal_vertical.append(False)
    percentage_diagonal = num_diagonal / len(gdf_edges) * 100
    percentage_horizontal_vertical = num_horizontal_vertical / len(gdf_edges) * 100
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

    coverage = len(df_speeds) / (len(gdf_edges) * NUM_SPEED_FILES * 24 * 4)
    coverage_diagonal = num_data_points_diagonal / (num_diagonal * NUM_SPEED_FILES * 24 * 4)
    coverage_horizontal_vertial = num_data_points_horizontal_vertical / (num_horizontal_vertical * NUM_SPEED_FILES * 24 * 4)

    # TODO mapped volume?
    # mapped volume

    print(
        f"{CITY.title()} ({YEAR}) & {len(gdf_edges)} & {num_horizontal_vertical} ({percentage_horizontal_vertical:.1f}\\%) & "
        f"{num_diagonal} ({percentage_diagonal:.1f}\\%) & {coverage:.2f} & {coverage_horizontal_vertial:.2f} & {coverage_diagonal:.2f} & {coverage_horizontal_vertial-coverage:+.2f}\\\\"
    )


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
    print(
        "city (year) & \\#edges & \\#N/E/S/W & \\#NE/SE/SW/NW & coverage & coverage N/E/S/W & coverage NE/SE/SW/NW & coverage N/E/S/W - coverage NE/SE/SW/NW \\\\"
    )
    for YEAR, CITIES in d.items():
        for CITY in CITIES:
            main(DATA_ROOT=DATA_ROOT, RELEASE=RELEASE, YEAR=YEAR, CITY=CITY)
