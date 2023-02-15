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
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.parquet import ParquetFile

from data_pipeline.h5_helpers import load_h5_file


def print_inventory_parquet(fn):
    print("")
    print("")
    print(f"### `{fn.relative_to(BASEDIR)}`")
    print("")
    schema = pq.read_schema(fn, memory_map=True)
    schema = [(name, str(pa_dtype)) for name, pa_dtype in zip(schema.names, schema.types)]

    pf = ParquetFile(fn)
    rows = next(pf.iter_batches(batch_size=1))
    df = pa.Table.from_batches([rows]).to_pandas().reset_index()
    first_row = df.iloc[0]

    df = pq.read_table(fn).to_pandas()
    print(len(df))

    print("| Key | Attribute     | Example      | Data Type | Description |")
    print("|-----|---------------|--------------|-----------|-------------|")
    for k, v in schema:
        if k.startswith("__"):
            continue
        print(f"|     | {k} | {first_row[k]} | {v} |    |")
        print(f"{df[k].min()}-{df[k].max()}")


def print_inventory_h5(fn):
    print("")
    print("")
    print(f"### `{fn.relative_to(BASEDIR)}`")
    print("")
    data = load_h5_file(fn)
    print(f"* dtype: `{data.dtype}`")
    print(f"* shape: `{data.shape}`")


if __name__ == "__main__":
    BASEDIR = Path("/iarai/public/t4c/data_pipeline/release20221026_residential_unclassified/2021")
    city = "**/berlin/**/*"

    subfolders_done = set()
    for fn in BASEDIR.rglob(city):
        if fn.parent in subfolders_done:
            continue

        if fn.name.endswith(".parquet"):
            print_inventory_parquet(fn)
            if fn.parent.parent.name != "road_graph":
                subfolders_done.add(fn.parent)
        elif fn.name.endswith(".h5"):
            subfolders_done.add(fn.parent)
            print_inventory_h5(fn)

        # TODO gpkg, graphml
