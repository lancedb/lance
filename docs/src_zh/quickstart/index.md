---
title: 快速开始
description: 开始使用 Lance - 创建数据集、从 Parquet 转换，学习基础知识
---

# Lance 表入门

本快速入门指南将带你了解 Lance 的核心功能，包括创建数据集（Dataset）、版本管理（Versioning）和向量搜索（Vector Search）。

完成本教程后，你将能够从 pandas DataFrame 创建 Lance 数据集，并将现有的 Parquet 文件转换为 Lance 格式。你还将了解使用 Lance 数据集的基本工作流程，并准备好探索版本管理和向量搜索等高级功能。

## 安装 Python SDK

开始使用 Lance 最简单的方式是通过我们的 Python SDK `pylance`：

```bash
pip install pylance
```

要获取最新功能和错误修复，可以安装预览版本：

```bash
pip install --pre --extra-index-url https://pypi.fury.io/lance-format/pylance
```

> Note: 预览版本与正式版本接受相同级别的测试。

## 设置你的环境

首先，导入必要的库：

```python
import shutil
import lance
import numpy as np
import pandas as pd
import pyarrow as pa
```

## 创建你的第一个数据集

Lance 构建在 Apache Arrow 之上，因此可以非常方便地处理 pandas DataFrame 和 Arrow 表。你可以从多种数据源创建 Lance 数据集，包括 pandas DataFrame、Arrow 表和现有的 Parquet 文件。Lance 会自动为你处理转换和优化。

### 创建一个简单的数据集

你将创建一个简单的 DataFrame 并将其写入 Lance 格式。这演示了创建 Lance 数据集的基本工作流程。

创建一个简单的 DataFrame：

```python
df = pd.DataFrame({"a": [5]})
df
```

接下来将这个 DataFrame 写入 Lance 格式并验证数据是否正确保存：

```python
shutil.rmtree("/tmp/test.lance", ignore_errors=True)

dataset = lance.write_dataset(df, "/tmp/test.lance")
dataset.to_table().to_pandas()
```

### 转换已有的 Parquet 文件

你将把一个已有的 Parquet 文件转换为 Lance 格式。这展示了如何将现有数据迁移到 Lance。

首先，创建一个 Parquet 文件，然后将其转换为 Lance：

```python
shutil.rmtree("/tmp/test.parquet", ignore_errors=True)
shutil.rmtree("/tmp/test.lance", ignore_errors=True)

tbl = pa.Table.from_pandas(df)
pa.dataset.write_dataset(tbl, "/tmp/test.parquet", format='parquet')

parquet = pa.dataset.dataset("/tmp/test.parquet")
parquet.to_table().to_pandas()
```

现在只需一行代码即可将 Parquet 数据集转换为 Lance 格式：

```python
dataset = lance.write_dataset(parquet, "/tmp/test.lance")

# Make sure it's the same
dataset.to_table().to_pandas()
```

## 下一步

现在你已经掌握了创建 Lance 数据集的基础知识，以下是你可以继续探索的内容：

- **[使用 Lance 进行数据集版本管理](versioning.md)** - 了解如何通过原生版本管理功能跟踪数据变更
- **[Lance 向量索引和向量搜索](vector-search.md)** - 使用 ANN 索引构建高性能向量搜索能力
