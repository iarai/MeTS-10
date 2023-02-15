# Val 02: Comparision with Loop Counter Speed or Flow values

### Preparations

After running the MeTS-10 pipeline you should have your version of the segment speeds output.

`counters01_prepare_data.ipynb` was used locally to scrape, parse and convert the counter data for London and Madrid.

For continuing you should have the following folder structure
```
├── loop_counters
│   ├── berlin
│   │   ├── downloads
│   │   └── speed
│   ├── london
│   │   ├── downloads
│   │   └── speed
│   └── madrid
│       ├── all
│       └── downloads
└── release20221026_residential_unclassified
    ├── 2021
    │   ├── road_graph
    │   │   └── berlin
    │   └── speed_classes
    │       └── berlin
    └── 2022
        ├── road_graph
        │   ├── london
        │   └── madrid
        └── speed_classes
            ├── london
            └── madrid
```

### Matching of Loop Counters to OSM Segments

The matching is done in the script `counters02_match_counters.ipynb`. For London the nearest segment is used and for motorways, if available, the name is checked. For Madrid the nearest segment with the most similar heading direction is used.

The matched locations with corresponding u, v and way IDs are stored in the files
* `london_locations_matched.parquet` (and `london_locations_matched.geojson` for visualization)
* `madrid_locations_matched.parquet` (and `madrid_locations_matched.geojson` for visualization)


### First explorations Madrid

All loop counters are in the same monthly files.
Values with speed (type=M30) or only flow (type=URB) can be distinguised accordingly.
The script `counters03_explore_madrid.ipynb` uses the matched counter locations to compare and visualize the speed and flow values.


### First explorations London

Loop counters come from two providers TfL (TIMS) and Higways England (WEBTRIS). TIMS values only contain flow readings while WEBTRIS do also contain speed values.
The script `counters03_explore_london.ipynb` uses the matched counter locations to compare and visualize the WEBTRIS speed values.


### First explorations Berlin

Berlin has fewer counters but all come with a speed reading. The script `counters03_explore_berlin.ipynb` uses the matched counter locations to compare and visualize the speed and flow values.
