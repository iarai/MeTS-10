## Data Overlap

### Barcelona

2021 & Barcelona & 361 & 2019-01-02--2019-06-30 (180), 2020-01-02--2020-06-30 (181)
barcelona: (41.3851, 2.1734) 2020-01 -- 2020-03

-> Overlap 3 months: 2020-01 -- 2020-03

OSM data from the range: `wget "https://download.geofabrik.de/europe/spain-200101.osm.pbf"`

### Berlin

2020 & Berlin & 181 & 2019-01-01--2019-06-30 (181)
2021 & Berlin & 180 & 2019-01-02--2019-06-30 (180)
berlin: (52.519171, 13.4060912) 2018-01 -- 2020-03

-> Overlap 6 months: 2019-01-01--2019-06-30

OSM data from the range: `wget "https://download.geofabrik.de/europe/germany-190101.osm.pbf"`

### London

london: (51.51262, -0.130438) 2018-01 -- 2020-03
2022 & London & 110 & 2019-07-01--2019-12-31 (184), 2020-01-01--2020-01-31 (31)

-> Overlap 7 months: 2019-07 -- 2020-01

OSM data from the range: `wget "https://download.geofabrik.de/europe/great-britain/england-200101.osm.pbf"`

### Madrid

No temporal overlap

### New York

new_york: (40.744842, -73.991483) 2018-01 -- 2020-03
no public here data (internally we have 2019 and 2020)

-> Overlap 0 months

```
/iarai/public/t4c/uber$ ls *--*.zip |xargs -n 1 unzip -l
Archive:  movement-speeds-hourly-barcelona-2020-1--2020-3.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
  1045815  2022-10-06 10:16   movement-speeds-hourly-barcelona-2020-1.parquet
  1673906  2022-10-06 10:33   movement-speeds-hourly-barcelona-2020-2.parquet
  1332439  2022-10-06 10:25   movement-speeds-hourly-barcelona-2020-3.parquet
---------                     -------
  4052160                     3 files
Archive:  movement-speeds-hourly-berlin-2019-1--2019-6.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
 12550184  2022-10-06 10:43   movement-speeds-hourly-berlin-2019-1.parquet
 13822427  2022-10-06 10:43   movement-speeds-hourly-berlin-2019-2.parquet
 15610946  2022-10-06 10:43   movement-speeds-hourly-berlin-2019-4.parquet
 17010702  2022-10-06 10:42   movement-speeds-hourly-berlin-2019-5.parquet
 15930150  2022-10-06 11:13   movement-speeds-hourly-berlin-2019-3.parquet
 19243218  2022-10-06 11:13   movement-speeds-hourly-berlin-2019-6.parquet
---------                     -------
 94167627                     6 files
Archive:  movement-speeds-hourly-london-2019-7--2020-1.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
603277858  2022-10-06 10:24   movement-speeds-hourly-london-2019-10.parquet
548268857  2022-10-06 10:33   movement-speeds-hourly-london-2019-11.parquet
663547934  2022-10-06 10:16   movement-speeds-hourly-london-2019-12.parquet
620830917  2022-10-06 10:43   movement-speeds-hourly-london-2019-7.parquet
598552491  2022-10-06 10:40   movement-speeds-hourly-london-2019-8.parquet
535193785  2022-10-06 10:29   movement-speeds-hourly-london-2020-1.parquet
589665647  2022-10-06 11:18   movement-speeds-hourly-london-2019-9.parquet
---------                     -------
4159337489                     7 files

```

## UBER Speed Data

```
https://movement.uber.com/cities/barcelona/downloads/speeds?lang=en-US&tp[y]=2020&tp[q]=1
https://movement.uber.com/cities/berlin/downloads/speeds?lang=en-US&tp[y]=2020&tp[q]=1
https://movement.uber.com/cities/cincinnati/downloads/speeds?lang=en-US&tp[y]=2020&tp[q]=1
https://movement.uber.com/cities/kyiv/downloads/speeds?lang=en-US&tp[y]=2020&tp[q]=1
https://movement.uber.com/cities/london/downloads/speeds?lang=en-US&tp[y]=2020&tp[q]=1
https://movement.uber.com/cities/madrid/downloads/speeds?lang=en-US&tp[y]=2020&tp[q]=1
https://movement.uber.com/cities/nairobi/downloads/speeds?lang=en-US&tp[y]=2020&tp[q]=1
https://movement.uber.com/cities/new_york/downloads/speeds?lang=en-US&tp[y]=2020&tp[q]=1
https://movement.uber.com/cities/san_francisco/downloads/speeds?lang=en-US&tp[y]=2020&tp[q]=1
https://movement.uber.com/cities/sao_paulo/downloads/speeds?lang=en-US&tp[y]=2020&tp[q]=1
https://movement.uber.com/cities/seattle/downloads/speeds?lang=en-US&tp[y]=2020&tp[q]=1

```

| city          | lon lat                    | UBER date range    | num quarters |
|---------------|----------------------------|--------------------|--------------|
| barcelona     | (41.3851, 2.1734)          | 2020-01 -- 2020-03 | 1            |
| berlin        | (52.519171, 13.4060912)    | 2018-01 -- 2020-03 | 9            |
| cincinnati    | (39.104166, -84.518997)    | 2018-01 -- 2020-03 | 9            |
| kyiv          | (50.4106, 30.609)          | 2018-01 -- 2020-03 | 9            |
| london        | (51.51262, -0.130438)      | 2018-01 -- 2020-03 | 9            |
| madrid        | (40.4166909, -3.7003454)   | 2018-01 -- 2020-03 | 9            |
| nairobi       | (-1.2831, 36.8209)         | 2018-01 -- 2020-03 | 9            |
| new_york      | (40.744842, -73.991483)    | 2018-01 -- 2020-03 | 9            |
| san_francisco | (37.7749295, -122.4194155) | 2018-01 -- 2020-03 | 9            |
| sao_paulo     | (-23.5505, -46.6333)       | 2018-01 -- 2020-03 | 9            |
| seattle       | (47.6062095, -122.3320708) | 2018-01 -- 2020-03 | 9            |
