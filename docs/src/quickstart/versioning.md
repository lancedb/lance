---
title: Versioning
description: Learn how to version your Lance datasets with append, overwrite, and tag features
---

# Versioning Your Datasets with Lance

Lance supports versioning natively, allowing you to track changes over time. 

In this tutorial, you'll learn how to append new data to existing datasets while preserving historical versions and access specific versions using version numbers or meaningful tags. You'll also understand how to implement proper data governance practices with Lance's native versioning capabilities.

## Install the Python SDK

```bash
pip install pylance
```

## Set Up Your Environment

First, you should import the necessary libraries:

```python
import shutil
import lance
import numpy as np
import pandas as pd
import pyarrow as pa
```

## Append New Data to Your Dataset

You can add new rows to your existing dataset, creating a new version while preserving the original data. Here is how to append rows:

```python
df = pd.DataFrame({"a": [10]})
tbl = pa.Table.from_pandas(df)
dataset = lance.write_dataset(tbl, "/tmp/test.lance", mode="append")

dataset.to_table().to_pandas()
```

## Overwrite Your Dataset

You can completely replace your dataset with new data, creating a new version while keeping the old version accessible.

Here is how to overwrite the data and create a new version:

```python
df = pd.DataFrame({"a": [50, 100]})
tbl = pa.Table.from_pandas(df)
dataset = lance.write_dataset(tbl, "/tmp/test.lance", mode="overwrite")

dataset.to_table().to_pandas()
```

## Access Previous Dataset Versions

You can also check what versions are available and then access specific versions of your dataset.

List all versions of a dataset with this request:

```python
dataset.versions()
```

You can also access any available version:

```python
# Version 1
lance.dataset('/tmp/test.lance', version=1).to_table().to_pandas()

# Version 2
lance.dataset('/tmp/test.lance', version=2).to_table().to_pandas()
```

## Tag Your Important Versions

Create named tags for important versions, making it easier to reference specific versions by meaningful names. To create tags for relevant versions, do this:

```python
dataset.tags.create("stable", 2)
dataset.tags.create("nightly", 3)
dataset.tags.list()
```

Tags can be checked out like versions:

```python
lance.dataset('/tmp/test.lance', version="stable").to_table().to_pandas()
```

## Next Steps

Now that you've mastered dataset versioning with Lance, check out **[Vector Indexing and Vector Search With Lance](vector-search.md)**. You can learn how to build high-performance vector search capabilities on top of your Lance tables.

This will teach you how to build fast, scalable search capabilities for your versioned datasets. 