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
    ignored = 0

    for g in gdf_edges["geometry"]:
        g: LineString = g
        coordinates = list(g.coords)
        orientations = set()
        for (lat_from, lon_from), (lat_to, lon_to) in zip(coordinates, coordinates[1:]):
            alpha = get_bearing(lat_from, lon_from, lat_to, lon_to)
            if not (0 <= alpha < 360):
                ignored += 1
                continue
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
            else:
                num_diagonals += 1
        else:
            pass
    print(f"num_diagonals={num_diagonals} ({num_diagonals / len(gdf_edges)*100:.2f}%)")
    print(f"num_horizontal_verticals={num_horizontal_verticals} ({num_horizontal_verticals / len(gdf_edges)*100:.2f}%)")
    # TODO what happens here?
    print(f"ignored={ignored}, what happens here?")


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


# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/antwerp
# num_diagonals=20612 0.25239080656813645
# num_horizontal_verticals=15745 0.19279513144844307
# ignored=0, what happens here?
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/bangkok
# num_diagonals=0 0.0
# num_horizontal_verticals=0 0.0
# ignored=1689699, what happens here?
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/barcelona
# num_diagonals=33847 0.2848762340821291
# num_horizontal_verticals=20562 0.17306187033405435
# ignored=0, what happens here?
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/berlin
# num_diagonals=19894 0.223824846425598
# num_horizontal_verticals=19164 0.21561170990751782
# ignored=0, what happens here?
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/chicago
# num_diagonals=39183 0.20889801140907394
# num_horizontal_verticals=62914 0.3354161113184411
# ignored=0, what happens here?
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/istanbul
# num_diagonals=61639 0.22820046721878945
# num_horizontal_verticals=57874 0.21426164992651114
# ignored=0, what happens here?
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/melbourne
# num_diagonals=0 0.0
# num_horizontal_verticals=0 0.0
# ignored=774842, what happens here?
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021/moscow
# num_diagonals=9410 0.19654531403387848
# num_horizontal_verticals=9845 0.20563109635106627
# ignored=0, what happens here?
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2022/london
# num_diagonals=64074 0.2363700083002859
# num_horizontal_verticals=55953 0.2064115097297796
# ignored=0, what happens here?
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2022/madrid
# num_diagonals=36153 0.25210945454038297
# num_horizontal_verticals=30255 0.21098032105549436
# ignored=0, what happens here?
# /iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2022/melbourne
# num_diagonals=0 0.0
# num_horizontal_verticals=0 0.0
# ignored=774842, what happens here?
