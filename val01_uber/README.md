# Val 02: Comparision with UBER Speed Data

## Setup

Use the `t4c22` environment from https://github.com/iarai/NeurIPS2022-traffic4cast

| File             | Description |
|------------------|-------------|
| uber.json        ||
| uber_parse.py    ||
| uber_read_all.py ||
| uber_validation.ipynb/_nb.py||

## UBER Seed Data

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

```
barcelona: (41.3851, 2.1734) 2020-01 -- 2020-03
berlin: (52.519171, 13.4060912) 2018-01 -- 2020-03
cincinnati: (39.104166, -84.518997) 2018-01 -- 2020-03
kyiv: (50.4106, 30.609) 2018-01 -- 2020-03
london: (51.51262, -0.130438) 2018-01 -- 2020-03
madrid: (40.4166909, -3.7003454) 2018-01 -- 2020-03
nairobi: (-1.2831, 36.8209) 2018-01 -- 2020-03
new_york: (40.744842, -73.991483) 2018-01 -- 2020-03
san_francisco: (37.7749295, -122.4194155) 2018-01 -- 2020-03
sao_paulo: (-23.5505, -46.6333) 2018-01 -- 2020-03
seattle: (47.6062095, -122.3320708) 2018-01 -- 2020-03
```