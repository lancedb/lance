# Quick Start

We've provided Linux and MacOS wheels for Lance in PyPI. You can install Lance python binding via:

```sh
pip install pylance
```

## Exploratory Data Analysis via DuckDB

Thanks for its [Apache Arrow](https://arrow.apache.org/)-first APIs, `lance` can be used as a native Arrow extension. For example, it enables users to directly use [DuckDB](https://duckdb.org/)
to analyze `lance` dataset via [DuckDB's Arrow integration](https://duckdb.org/docs/guides/python/sql_on_arrow).

```python
# pip install pylance duckdb
import lance
import duckdb
```

### Understand Label distribution of Oxford Pet Dataset

```python
ds = lance.dataset("s3://eto-public/datasets/oxford_pet/pet.lance")
duckdb.query('select label, count(1) from ds group by label').to_arrow_table()
```

## Deep Learning with PyTorch

`Lance` provides a PyTorch [`IterableDataset`](https://pytorch.org/docs/stable/data.html#iterable-style-datasets) via `lance.pytorch.data.LanceDataset`,
allowing users to use Lance datasets with their existing PyTorch pipelines.

For example, using lance dataset in [Pytorch Lightning](https://www.pytorchlightning.ai/) trainer:

```python

from lance.pytorch.data import LanceDataset
import torchvision.transforms as T
import pytorch_lightning as pl
from torchdata.datapipes.iter import IterableWrapper

transform = T.Compose([...])
dataset = LanceDataset(
    "s3://eto-public/datasets/coco/coco.lance",
    columns=["image", "annotations.category_id", "annotations.bbox"],
    batch_size=64,
    transform=transform,
)
dp = IterableWrapper(dataset).shuffle(buffer_size=256)
train_loader = torch.utils.data.DataLoader(
    dp,
    batch_size=batch_size,
    collate_fn=collate_fn,
    pin_memory=True,
)

# Use all GPUs to train.
trainer = pl.Trainer(accelerator="gpu", devices=-1)
trainer.fit(model, train_dataloaders=train_loader)
```
