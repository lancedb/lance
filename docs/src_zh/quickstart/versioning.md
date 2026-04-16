---
title: 版本管理
description: 了解如何使用追加、覆写、标签和分支对 Lance 数据集进行版本管理
---

# 使用 Lance 进行数据集版本管理

Lance 原生支持版本管理（Versioning），允许你跟踪数据随时间的变化。

在本教程中，你将学习如何在保留历史版本的同时向现有数据集追加新数据，使用版本号或有意义的标签（Tag）访问特定版本。你还将了解如何利用 Lance 的原生版本管理能力实现适当的数据治理实践。

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
```

## 向数据集追加新数据

你可以向现有数据集添加新行，创建新版本的同时保留原始数据。以下是追加行的方法：

```python
df = pd.DataFrame({"a": [10]})
tbl = pa.Table.from_pandas(df)
dataset = lance.write_dataset(tbl, "/tmp/test.lance", mode="append")

dataset.to_table().to_pandas()
```

## 覆写数据集

你可以用新数据完全替换数据集，创建新版本的同时保持旧版本可访问。

以下是覆写数据并创建新版本的方法：

```python
df = pd.DataFrame({"a": [50, 100]})
tbl = pa.Table.from_pandas(df)
dataset = lance.write_dataset(tbl, "/tmp/test.lance", mode="overwrite")

dataset.to_table().to_pandas()
```

## 访问之前的数据集版本

你还可以查看有哪些可用版本，然后访问数据集的特定版本。

列出数据集的所有版本：

```python
dataset.versions()
```

你也可以访问任何可用的版本：

```python
# Version 1
lance.dataset('/tmp/test.lance', version=1).to_table().to_pandas()

# Version 2
lance.dataset('/tmp/test.lance', version=2).to_table().to_pandas()
```

## 为重要版本打标签

为重要版本创建命名标签（Tag），使其更容易通过有意义的名称进行引用。

```python
dataset.tags.create("stable", 2)
dataset.tags.create("nightly", 3)
dataset.tags.list()
```

标签可以像版本号一样被检出：

```python
lance.dataset('/tmp/test.lance', version="stable").to_table().to_pandas()
```

关于高级标签操作（例如在特定分支上给版本打标签），请参阅[标签和分支](../guide/tags_and_branches.md)。

## 使用分支

分支（Branch）管理数据集演进的并行路线。你可以从现有版本或标签创建分支，独立地对其进行读写操作，以及检出不同的分支。

```python
# Create branch from current latest version
experiment_branch = ds.create_branch("experiment")

# Write to the branch (affects only that branch's history)
tbl = pa.Table.from_pandas(pd.DataFrame({"a": [42]}))
lance.write_dataset(tbl, experiment_branch, mode="append")
```

更多详细信息，请参阅[标签和分支](../guide/tags_and_branches.md)。

## 下一步

现在你已经掌握了 Lance 的数据集版本管理，请查看 **[Lance 向量索引和向量搜索](vector-search.md)**。你可以学习如何在 Lance 表之上构建高性能向量搜索能力。

这将教你如何为版本化的数据集构建快速、可扩展的搜索能力。
