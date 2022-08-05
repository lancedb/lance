# Python bindings for Lance file format

Lance is a cloud-native columnar data format designed for managing large-scale computer vision datasets in production
environments. Lance delivers blazing fast performance for image and video data use cases from analytics to point 
queries to training scans. 

## Why use Lance

You should use lance if you're a ML engineer looking to be 10x more productive when working with computer vision 
datasets:

1. Lance saves you from having to manage multiple systems and formats for metadata, 
raw assets, labeling updates, and vector indices.
2. Lance's custom column encoding means you don't need to choose between fast analytics and fast point queries.
3. Lance has a first-class Apache Arrow integration so it's easy to create and query Lance datasets (e.g., you can
directly query lance datasets using DuckDB with no extra work)
4. Did we mention Lance is fast.

## Try Lance

Install Lance from pip (use a venv, not conda):

```bash
pip install pylance duckdb
```

In python:

```python
import lance
import duckdb

# Understand Label distribution of Oxford Pet Dataset
ds = lance.dataset("s3://eto-public/datasets/oxford_pet/pet.lance")
duckdb.query('select label, count(1) from ds group by label').to_arrow_table()
```

### Caveat emptor

- DON'T use Conda as it prefers it's on ld path and libstd etc
- Currently only wheels are on pypi and no sdist. See below for instructions on building from source. 
- Python 3.8-3.10 is supported on Linux x86_64
- Python 3.10 on MacOS (both x86_64 and Arm64) is supported

## Developing Lance

Install python3, pip, and venv, and setup a virtual environment for Lance.
Again, DO NOT USE CONDA (at least for now).

```bash
sudo apt install python3-pip python3-venv python3-dev
python3 -m venv ${HOME}/.venv/lance
```

# Arrow C++ libs

Install Arrow C++ libs using instructions from [Apache Arrow](https://arrow.apache.org/install/).
These instructions don't include Arrow's python lib so after you go through the above, don't forget to
apt install `libarrow-python-dev` or yum install `libarrow-python-devel`.

# Build pyarrow

Assume CWD is where you want to put the repo:

```bash
source ${HOME}/.venv/lance/bin/activate
cd /path/to/lance/python/thirdparty
./build.sh
```

Make sure pyarrow works properly:

```python
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
```

# Build Lance

1. Build the cpp lib. See [lance/cpp/README.md](../cpp/README.md) for instructions.
2. Build the python module in venv:

```bash
source ${HOME}/.venv/lance/bin/activate
python setup.py develop
```

Test the installation using the same queries in [Try Lance section](#try-lance).