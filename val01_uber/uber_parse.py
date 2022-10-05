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
import json
from pathlib import Path

# view-source:https://movement.uber.com/explore/atlanta/travel-times/query?lat.=33.7489&lng.=-84.3881&z.=12&lang=en-US&si=166&ti=757&ag=censustracts&dt[tpb]=ALL_DAY&dt[wd;]=1,2,3,4,5,6,7&dt[dr][sd]=2020-03-01&dt[dr][ed]=2020-03-31&cd=&sa;=&sdn=
# __REDUX_STATE__

if __name__ == '__main__':
    with Path("uber.json").open() as f:
        j = json.load(f)
        for city in j["availableCities"]["data"]:
            city_name = city["slug"]
            available = city["availableSolutionTypes"]
            if "Speeds" in available:
                lat = city["lat"]
                lon = city["lng"]
                print(f"{city_name}: {lat,lon}")
                #print(f"{city_name}: {len()}")
                #print(f"https://movement.uber.com/cities/{city_name.lower()}/downloads/speeds?lang=en-US&tp[y]=2020&tp[q]=1")


# barcelona, berlin, london, madrid, (new york)