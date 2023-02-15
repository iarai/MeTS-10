Data Pipeline and Analysis Code for paper "Metropolitan Segment Traffic Speeds from Massive Floating Car Data in 10 Cities" (MeTS-10)
===============================================================================

[![Tests](https://github.com/chenkins/MeTS-10/actions/workflows/tests.yaml/badge.svg)](https://github.com/chenkins/MeTS-10/actions/workflows/tests.yaml)
[![Code style: flake8](https://img.shields.io/badge/Code%20style-flake8-yellow.svg)](https://github.com/pycqa/flake8/)
[![Code formatter: black](https://img.shields.io/badge/Code%20formatter-black-000000.svg)](https://github.com/psf/black)

## About this repo

This is a github repo to share code for the MeTS-10 Dataset Paper.

## MeTS-10 Data Pipeline

The major parts of the speed classification pipeline are provided as a series of scripts
that allow to generate a road graph and derive speed classifications from available traffic movies end-to-end.

The following diagram gives an overview:
<img src="./data_pipeline/img/data_pipeline_scripts.svg">

Find the technical data description in [README_DATA_SPECIFICATION.md](README_DATA_SPECIFICATION.md).

## MeTS-10 analysis

This part contains code to generate the figures in the paper.

## TL;DR

```
conda env update -f environment.yaml
conda activate mets-10
python data_pipeline/dp01_movie_aggregation.py --help
```

## Setup
[Jupytext](https://jupytext.readthedocs.io/en/latest/install.html)

## Contribution conventions

For the data pipeline, we  run formatter and linter using `pre-commit` (https://pre-commit.com/), see
configuration `.pre-commit-config.yaml`:

```
pre-commit install # first time only
pre-commit run --all
```

See https://blog.mphomphego.co.za/blog/2019/10/03/Why-you-need-to-stop-using-Git-Hooks.html

In order to temporarily skip running `pre-commit`, run `git commit -n`.
