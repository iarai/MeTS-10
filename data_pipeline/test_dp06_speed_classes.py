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
import tempfile
from pathlib import Path

import pandas
import pytest
from dp06_speed_classes import main
from dummy_competition_setup_for_testing import create_dummy_competition_setup


@pytest.mark.parametrize(
    "aggregation,expected_len,additional_params",
    [
        (aggregation, expected_len, additional_params)
        for aggregation, expected_len in [("5", 6 * 12), ("15", 6 * 4), ("20", 6 * 3)]
        for additional_params in [[], ["--no_trust_filtering"], ["--disable_free_flow_normalization"]]
    ],
)
def test_speed_classes(aggregation, expected_len, additional_params):
    dates = ["1970-01-01", "1970-01-02"]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        create_dummy_competition_setup(
            basedir=tmp_dir,
            city="london",
            skip_train_labels=True,
            skip_speed_classes=True,
            train_dates=dates,
            skip_movie=False,
            skip_movie_15min=False,
            skip_movie_20min=False,
            skip_free_flow=False,
            seed=667,
        )
        assert (tmp_dir / "road_graph" / "london" / "road_graph_freeflow.parquet").exists()
        aggregation_suffix = f"_{aggregation}min" if aggregation != "5" else ""
        for date in dates:
            assert (tmp_dir / (f"movie{aggregation_suffix}") / "london" / f"{date}_london_8ch{aggregation_suffix}.h5").exists()
            assert not (tmp_dir / "speed_classes" / "london" / f"speed_classes_{date}.parquet").exists()

        main(["-d", str(tmp_dir), "-c", "london", "--aggregation", aggregation] + additional_params)

        for date in dates:
            assert (tmp_dir / "speed_classes" / "london" / f"speed_classes_{date}.parquet").exists()

        sc_df = pandas.read_parquet(tmp_dir / "speed_classes" / "london" / f"speed_classes_{dates[0]}.parquet")
        print(sc_df.columns)
        print(sc_df)
        print(sc_df.groupby(["u", "v"]).count())
        assert len(sc_df) == expected_len
        assert sc_df["median_speed_kph"].min() > 0
        assert sc_df["median_speed_kph"].max() < 120
