from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if __name__ == "__main__":

    csv_files = list(Path("/iarai/public/t4c/uber").rglob("*.csv"))
    for i, csv_f in enumerate(csv_files):
        parquet_f = csv_f.with_suffix(".parquet")
        print(f"{i}/{len(csv_files)}: {csv_f} -> {parquet_f}")
        if parquet_f.exists():
            print(f" -> skipping {parquet_f} already exists")
            continue
        df = pd.read_csv(csv_f)
        table = pa.Table.from_pandas(df)

        pq.write_table(table, parquet_f, compression="snappy")
        print(f" -> written {parquet_f}")
