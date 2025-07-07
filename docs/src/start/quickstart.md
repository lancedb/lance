# Quickstart

This quickstart guide will walk you through the core features of Lance including creating datasets, versioning, and vector search.

## Prerequisites

First, let's import the necessary libraries:

```python
import shutil
import lance
import numpy as np
import pandas as pd
import pyarrow as pa
```

## Creating Datasets

Via PyArrow it's really easy to create Lance datasets.

### Basic Dataset Creation

Create a simple dataframe:

```python
df = pd.DataFrame({"a": [5]})
df
```

Write it to Lance:

```python
shutil.rmtree("/tmp/test.lance", ignore_errors=True)

dataset = lance.write_dataset(df, "/tmp/test.lance")
dataset.to_table().to_pandas()
```

### Converting from Parquet

You can easily convert existing Parquet files to Lance:

```python
shutil.rmtree("/tmp/test.parquet", ignore_errors=True)
shutil.rmtree("/tmp/test.lance", ignore_errors=True)

tbl = pa.Table.from_pandas(df)
pa.dataset.write_dataset(tbl, "/tmp/test.parquet", format='parquet')

parquet = pa.dataset.dataset("/tmp/test.parquet")
parquet.to_table().to_pandas()
```

Write to Lance in 1 line:

```python
dataset = lance.write_dataset(parquet, "/tmp/test.lance")

# Make sure it's the same
dataset.to_table().to_pandas()
```

## Versioning

Lance supports versioning natively, allowing you to track changes over time.

### Appending Data

We can append rows:

```python
df = pd.DataFrame({"a": [10]})
tbl = pa.Table.from_pandas(df)
dataset = lance.write_dataset(tbl, "/tmp/test.lance", mode="append")

dataset.to_table().to_pandas()
```

### Overwriting Data

We can overwrite the data and create a new version:

```python
df = pd.DataFrame({"a": [50, 100]})
tbl = pa.Table.from_pandas(df)
dataset = lance.write_dataset(tbl, "/tmp/test.lance", mode="overwrite")

dataset.to_table().to_pandas()
```

### Accessing Previous Versions

The old version is still there:

```python
dataset.versions()
```

You can access any version:

```python
# Version 1
lance.dataset('/tmp/test.lance', version=1).to_table().to_pandas()

# Version 2
lance.dataset('/tmp/test.lance', version=2).to_table().to_pandas()
```

### Tags

We can create tags for important versions:

```python
dataset.tags.create("stable", 2)
dataset.tags.create("nightly", 3)
dataset.tags.list()
```

Tags can be checked out like versions:

```python
lance.dataset('/tmp/test.lance', version="stable").to_table().to_pandas()
```

## Vector Search

Lance provides high-performance vector search capabilities with ANN (Approximate Nearest Neighbor) indexes.

### Data Preparation

For this tutorial, let's use the SIFT 1M dataset:

- Download `ANN_SIFT1M` from: http://corpus-texmex.irisa.fr/
- Direct link: `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`

```bash
rm -rf sift* vec_data.lance
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
```

### Convert to Lance

```python
from lance.vector import vec_to_table
import struct

uri = "vec_data.lance"

with open("sift/sift_base.fvecs", mode="rb") as fobj:
    buf = fobj.read()
    data = np.array(struct.unpack("<128000000f", buf[4 : 4 + 4 * 1000000 * 128])).reshape((1000000, 128))
    dd = dict(zip(range(1000000), data))

table = vec_to_table(dd)
lance.write_dataset(table, uri, max_rows_per_group=8192, max_rows_per_file=1024*1024)
```

Load the dataset:

```python
uri = "vec_data.lance"
sift1m = lance.dataset(uri)
```

### KNN Search (No Index)

First, let's sample some query vectors:

```python
import duckdb
# Make sure DuckDB v0.7+ is installed
samples = duckdb.query("SELECT vector FROM sift1m USING SAMPLE 100").to_df().vector
```

```
0     [29.0, 10.0, 1.0, 50.0, 7.0, 89.0, 95.0, 51.0,...
1     [7.0, 5.0, 39.0, 49.0, 17.0, 12.0, 83.0, 117.0...
2     [0.0, 0.0, 0.0, 10.0, 12.0, 31.0, 6.0, 0.0, 0....
3     [0.0, 2.0, 9.0, 1.793662034335766e-43, 30.0, 1...
4     [54.0, 112.0, 16.0, 0.0, 0.0, 7.0, 112.0, 44.0...
                            ...
95    [1.793662034335766e-43, 33.0, 47.0, 28.0, 0.0,...
96    [1.0, 4.0, 2.0, 32.0, 3.0, 7.0, 119.0, 116.0, ...
97    [17.0, 46.0, 12.0, 0.0, 0.0, 3.0, 23.0, 58.0, ...
98    [0.0, 11.0, 30.0, 14.0, 34.0, 7.0, 0.0, 0.0, 1...
99    [20.0, 8.0, 121.0, 98.0, 37.0, 77.0, 9.0, 18.0...
Name: vector, Length: 100, dtype: object
```

Perform nearest neighbor search without an index:

```python
import time

start = time.time()
tbl = sift1m.to_table(columns=["id"], nearest={"column": "vector", "q": samples[0], "k": 10})
end = time.time()

print(f"Time(sec): {end-start}")
print(tbl.to_pandas())
```

Expected output:
```
Time(sec): 0.10735273361206055
       id                                             vector    score
0  144678  [29.0, 10.0, 1.0, 50.0, 7.0, 89.0, 95.0, 51.0,...      0.0
1  575538  [2.0, 0.0, 1.0, 42.0, 3.0, 38.0, 152.0, 27.0, ...  76908.0
2  241428  [11.0, 0.0, 2.0, 118.0, 11.0, 108.0, 116.0, 21...  92877.0
...
```

Without the index, this scans through the whole dataset to compute the distance. For real-time serving, we can do much better with an ANN index.

### Building an Index

Lance supports IVF_PQ, IVF_HNSW_PQ and IVF_HNSW_SQ indexes:

```python
sift1m.create_index(
    "vector",
    index_type="IVF_PQ", # IVF_PQ, IVF_HNSW_PQ and IVF_HNSW_SQ are supported
    num_partitions=256,  # IVF
    num_sub_vectors=16,  # PQ
)
```

```
Building vector index: IVF256,PQ16
CPU times: user 2min 23s, sys: 2.77 s, total: 2min 26s
Wall time: 22.7 s
Sample 65536 out of 1000000 to train kmeans of 128 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
```

!!! warning "Index Creation Performance"
    If you're trying this on your own data, make sure your vector (dimensions / num_sub_vectors) % 8 == 0, or else index creation will take much longer than expected due to SIMD misalignment.

### ANN Search with Index

Let's look for nearest neighbors again with the ANN index:

```python
sift1m = lance.dataset(uri)

import time

tot = 0
for q in samples:
    start = time.time()
    tbl = sift1m.to_table(nearest={"column": "vector", "q": q, "k": 10})
    end = time.time()
    tot += (end - start)

print(f"Avg(sec): {tot / len(samples)}")
print(tbl.to_pandas())
```

Expected output:
```
Avg(sec): 0.0009334301948547364
       id                                             vector         score
0  378825  [20.0, 8.0, 121.0, 98.0, 37.0, 77.0, 9.0, 18.0...  16560.197266
1  143787  [11.0, 24.0, 122.0, 122.0, 53.0, 4.0, 0.0, 3.0...  61714.941406
2  356895  [0.0, 14.0, 67.0, 122.0, 83.0, 23.0, 1.0, 0.0,...  64147.218750
3  535431  [9.0, 22.0, 118.0, 118.0, 4.0, 5.0, 4.0, 4.0, ...  69092.593750
4  308778  [1.0, 7.0, 48.0, 123.0, 73.0, 36.0, 8.0, 4.0, ...  69131.812500
5  222477  [14.0, 73.0, 39.0, 4.0, 16.0, 94.0, 19.0, 8.0,...  69244.195312
6  672558  [2.0, 1.0, 0.0, 11.0, 36.0, 23.0, 7.0, 10.0, 0...  70264.828125
7  365538  [54.0, 43.0, 97.0, 59.0, 34.0, 17.0, 10.0, 15....  70273.710938
8  659787  [10.0, 9.0, 23.0, 121.0, 38.0, 26.0, 38.0, 9.0...  70374.703125
9  603930  [32.0, 32.0, 122.0, 122.0, 70.0, 4.0, 15.0, 12...  70583.375000
```

!!! note "Performance Note"
    Your actual numbers will vary by your storage. These numbers are from local disk on an M2 MacBook Air. If you're querying S3 directly, HDD, or network drives, performance will be slower.

### Tuning Search Parameters

The latency vs recall is tunable via:
- **nprobes**: how many IVF partitions to search
- **refine_factor**: determines how many vectors are retrieved during re-ranking

```python
%%time

sift1m.to_table(
    nearest={
        "column": "vector",
        "q": samples[0],
        "k": 10,
        "nprobes": 10,
        "refine_factor": 5,
    }
).to_pandas()
```

**Parameter Explanation:**
- `q` => sample vector
- `k` => how many neighbors to return
- `nprobes` => how many partitions (in the coarse quantizer) to probe
- `refine_factor` => controls "re-ranking". If k=10 and refine_factor=5 then retrieve 50 nearest neighbors by ANN and re-sort using actual distances then return top 10. This improves recall without sacrificing performance too much

!!! note "Memory Usage"
    The latencies above include file I/O as Lance currently doesn't hold anything in memory. Along with index building speed, creating a purely in-memory version of the dataset would make the biggest impact on performance.

### Features and Vectors Together

Usually we have other feature or metadata columns that need to be stored and fetched together. If you're managing data and the index separately, you have to do a bunch of annoying plumbing to put stuff together. With Lance it's a single call:

```python
tbl = sift1m.to_table()
tbl = tbl.append_column("item_id", pa.array(range(len(tbl))))
tbl = tbl.append_column("revenue", pa.array((np.random.randn(len(tbl))+5)*1000))
```

You can then query both vectors and metadata together:

```python
# Get vectors and metadata together
result = sift1m.to_table(
    columns=["item_id", "revenue"],
    nearest={"column": "vector", "q": samples[0], "k": 10}
)
print(result.to_pandas())
```