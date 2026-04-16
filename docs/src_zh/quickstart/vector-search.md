---
title: 向量搜索
description: 使用 ANN 索引进行高性能向量搜索，包括 IVF_PQ、IVF_HNSW_PQ 和 IVF_HNSW_SQ
---

# Lance 向量索引和向量搜索

Lance 提供高性能的向量搜索（Vector Search）能力，支持 ANN（近似最近邻，Approximate Nearest Neighbor）索引。

完成本教程后，你将能够构建和使用 ANN 索引来大幅加速向量搜索操作，同时保持高精度。你还将学习如何调优搜索参数以获得最佳性能，以及如何在单次操作中将向量搜索与元数据查询结合使用。

## 安装 Python SDK

```bash
pip install pylance
```

## 设置你的环境

首先，导入必要的库：

```python
import shutil
import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import duckdb
```

## 准备你的向量嵌入

在本教程中，下载并准备 SIFT 1M 数据集用于向量搜索实验。

- 从以下地址下载 `ANN_SIFT1M`：http://corpus-texmex.irisa.fr/
- 直接链接：`ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`

你可以直接使用 `wget`：

```bash
rm -rf sift* vec_data.lance
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
```

## 将数据转换为 Lance 格式

然后，将原始向量数据转换为 Lance 格式，以实现高效存储和查询。

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

现在你可以加载数据集：

```python
uri = "vec_data.lance"
sift1m = lance.dataset(uri)
```

## 无索引搜索

你将先执行无索引的向量搜索来查看基线性能，然后与索引搜索进行对比。

首先，采样一些查询向量：

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

现在，执行无索引的最近邻搜索：

```python
import time

start = time.time()
tbl = sift1m.to_table(columns=["id"], nearest={"column": "vector", "q": samples[0], "k": 10})
end = time.time()

print(f"Time(sec): {end-start}")
print(tbl.to_pandas())
```

预期输出：
```
Time(sec): 0.10735273361206055
       id                                             vector    score
0  144678  [29.0, 10.0, 1.0, 50.0, 7.0, 89.0, 95.0, 51.0,...      0.0
1  575538  [2.0, 0.0, 1.0, 42.0, 3.0, 38.0, 152.0, 27.0, ...  76908.0
2  241428  [11.0, 0.0, 2.0, 118.0, 11.0, 108.0, 116.0, 21...  92877.0
...
```

在没有索引的情况下，搜索会扫描整个数据集来计算每个数据点之间的距离。要获得实际可用的实时性能，使用 ANN 索引将获得更好的表现。

## 构建搜索索引

构建 ANN 索引后，你可以大幅加速向量搜索操作，同时保持高精度。在本示例中，我们将构建 `IVF_PQ` 索引：

```python
sift1m.create_index(
    "vector",
    index_type="IVF_PQ", # specify the IVF_PQ index type
    num_partitions=256,  # IVF
    num_sub_vectors=16,  # PQ
)
```

示例输出应如下所示：

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

!!! warning "索引创建性能"
    如果你使用自己的数据，请确保你的向量维度满足 (dimensions / num_sub_vectors) % 8 == 0，否则由于 SIMD 对齐问题，索引创建将比预期慢得多。

## 使用 ANN 索引进行向量搜索

现在你可以使用新创建的索引执行相同的搜索操作，并查看显著的性能提升。

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

预期输出：
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

!!! note "性能说明"
    你的实际数据会因存储介质而异。这些数据来自 M2 MacBook Air 上的本地磁盘。如果你直接查询 S3、HDD 或网络驱动器，性能会较慢。

## 调优搜索参数

你需要调整搜索参数来平衡速度和精度，为你的用例找到最优设置。

延迟与召回率（Recall）可通过以下参数调优：
- **nprobes**：搜索多少个 IVF 分区
- **refine_factor**：决定重排序时检索多少个向量

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

**参数说明：**
- `q` => 查询向量
- `k` => 返回多少个最近邻
- `nprobes` => 探测粗量化器中的多少个分区
- `refine_factor` => 控制"重排序"。如果 k=10 且 refine_factor=5，则通过 ANN 检索 50 个最近邻，然后使用实际距离重新排序后返回前 10 个。这可以在不显著牺牲性能的情况下提高召回率

!!! note "内存使用"
    上述延迟包含文件 I/O，因为 Lance 目前不在内存中保留任何内容。除了索引构建速度外，创建数据集的纯内存版本将对性能产生最大影响。

## 组合特征和向量

你可以向向量数据集添加元数据列，并在单次操作中同时查询向量和特征。

在实际场景中，用户有其他特征或元数据列需要一起存储和获取。如果你分别管理数据和索引，就需要做大量烦人的管道工作来将它们拼接在一起。

使用 Lance，你可以通过 `add_columns()` 直接向数据集添加列。对于基本用例，你可以使用 SQL：

```python
sift1m.add_columns(
    {
        "item_id": "id + 1000000",
        "revenue": "random() * 1000 + 5000",
    }
)
```
对于更复杂的列，你可以提供 Python 函数来生成新的列数据：
```python
@lance.batch_udf()
def add_columns_func(batch: pa.Table) -> pd.DataFrame:
    """Add item_id and revenue columns to a batch of data.

    Args:
        batch: PyArrow Table containing the original data

    Returns:
        Pandas DataFrame with added item_id and revenue columns
    """
    item_ids: np.ndarray = np.arange(batch.num_rows)
    revenue: np.ndarray = (np.random.randn(batch.num_rows) + 5) * 1000
    return pd.DataFrame({"item_id": item_ids, "revenue": revenue})


sift1m.add_columns(add_columns_func)
```
然后你可以同时查询向量和元数据：

```python
# Get vectors and metadata together
result = sift1m.to_table(
    columns=["item_id", "revenue"],
    nearest={"column": "vector", "q": samples[0], "k": 10}
)
print(result.to_pandas())
```

## 下一步

请查看 **[全文搜索](../quickstart/full-text-search.md)**，我们将展示如何在 Lance 中创建和查询 BM25 索引以进行基于关键词的搜索。
