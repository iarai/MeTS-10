import ast
from pathlib import Path

import pandas as pd

BASEDIR = Path("/iarai/public/t4c/data_pipeline/release20221026_residential_unclassified_no_trust_filtering")


def simplified_filter(hw):
    return hw in [
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
    ]


def congestion_class(probe_volume, congestion_factor):
    if probe_volume < 5 and congestion_factor < 0.4:
        return 2
    if probe_volume < 3 and congestion_factor < 0.8:
        return 1
    return 0


def main():
    # TODO other cities?
    for city in ["madrid"]:
        df_edges = pd.read_parquet(BASEDIR / "road_graph" / city / "road_graph_edges.parquet")
        df_edges["highway"] = [ast.literal_eval(hw) if hw.startswith("[") else hw for hw in df_edges["highway"]]
        df_edges["highway"] = [hw[0] if isinstance(hw, list) else hw for hw in df_edges["highway"]]

        speed_classes_files = (BASEDIR / "speed_classes" / city).rglob("speed_classes_*.parquet")
        for f in speed_classes_files:
            print(f)
            df = pd.read_parquet(f)
            df = df.merge(df_edges, on=["u", "v", "gkey"])
            df["congestion_class"] = [
                congestion_class(probe_volume, congestion_factor) for probe_volume, congestion_factor in zip(df["volume"], df["congestion_factor"])
            ]
            df["h"] = df["t"] // 4

            print(f"total number of speed labels: {len(df)}")
            df_filtered = df[df["congestion_class"] > 0]

            print(f"ratio filtered: {len(df_filtered) / len(df):.2f}")
            df_filter_yellow = df[df["congestion_class"] == 1]
            print(f"ratio filtered: {len(df_filter_yellow) / len(df):.2f}")
            df_filter_red = df[df["congestion_class"] == 2]
            print(f"ratio filtered: {len(df_filter_red) / len(df):.2f}")

            hw_counts = {rec["highway"]: rec["u"] for rec in df.groupby("highway").count().reset_index().to_dict("records")}

            df_filtered_hw = df_filtered.groupby("highway").count().reset_index()
            df_filter_yellow_hw = df_filter_yellow.groupby("highway").count().reset_index()
            df_filter_red_hw = df_filter_red.groupby("highway").count().reset_index()

            df_filtered_hw["absolute"] = df_filtered_hw["u"]
            df_filter_yellow_hw["absolute"] = df_filter_yellow_hw["u"]
            df_filter_red_hw["absolute"] = df_filter_red_hw["u"]
            df_filtered_hw["relative"] = [c / hw_counts[hw] for hw, c in zip(df_filtered_hw["highway"], df_filtered_hw["absolute"])]
            df_filter_yellow_hw["relative"] = [c / hw_counts[hw] for hw, c in zip(df_filter_yellow_hw["highway"], df_filter_yellow_hw["absolute"])]
            df_filter_red_hw["relative"] = [c / hw_counts[hw] for hw, c in zip(df_filter_red_hw["highway"], df_filter_red_hw["absolute"])]
            print("filtered yellow or red")
            print(df_filtered_hw[["highway", "absolute", "relative"]])
            print("filtered yellow")
            print(df_filter_yellow_hw[["highway", "absolute", "relative"]])
            print("filtered red")
            print(df_filter_red_hw[["highway", "absolute", "relative"]])

            print("filtered yellow or red")
            df_filtered_h = df_filtered.groupby("h").count().reset_index()
            df_filtered_h["relative"] = df_filtered_h["u"] / len(df_filtered)
            df_filtered_h["absolute"] = df_filtered_h["u"]
            print(df_filtered_h[["h", "absolute", "relative"]])

            # TODO aggregate over multiple files first
            break


if __name__ == "__main__":
    main()
