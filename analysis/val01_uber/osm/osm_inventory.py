import warnings
from pathlib import Path

import osmnx as ox
import pandas as pd

if __name__ == "__main__":
    ox.config(use_cache=True, log_console=True)
    OBASEPATH = Path("/iarai/public/t4c/osm")

    for f in OBASEPATH.rglob("**/*.graphml"):
        print(f"loading {f}")
        # ox.load_graphml(f)
    for f in OBASEPATH.rglob("**/*.parquet"):
        print(f"loading {f}")
        df = pd.read_parquet(f)
        print(df)
        if "edges" in f.name:
            df_by_key = df.groupby(["u", "v", "osmid"]).agg(count=("geometry", "count"))
            print(df_by_key)
            check = (df_by_key["count"] == 1).all()
            if not check:
                warnings.warn(f'Keys u,v,gkey not unique: {df_by_key[df_by_key["count"] > 1]}')
            else:
                print(f"-> check done for {f}")
