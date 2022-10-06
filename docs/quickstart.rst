Quick Start
===========

We've provided Linux and MacOS wheels for Lance in PyPI. You can install Lance python bindings via:

.. code-block:: bash

    pip install pylance

Exploratory Data Analysis
-------------------------

Thanks for its Apache Arrow-first APIs, `lance`` can be used as a native Arrow extension.
For example, it enables users to directly use DuckDB to analyze lance dataset via DuckDB's Arrow integration.

.. code-block:: python

    # pip install pylance duckdb
    import lance
    import duckdb


Understand Label distribution of Oxford Pet Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ds = lance.dataset("s3://eto-public/datasets/oxford_pet/pet.lance")
    duckdb.query("select label, count(1) from ds group by label").to_arrow_table()


Model Training and Evaluation
-----------------------------

`Lance` offers a PyTorch
`IterableDataset <https://pytorch.org/docs/stable/data.html#iterable-style-datasets>`_ to natively integrate
with PyTorch ecosystem.

For example, using `Lance` with Pytorch Lightning:

.. code-block:: python

    from lance.pytorch.data import LanceDataset
    from torchdata.datapipes.iter import IterableWrapper
    import pytorch_lightning as pl
    import torchvision.transform as T

    transform = T.Compose([T.PILToTensor()])

    dataset = LanceDataset(
        "s3://eto-public/datasets/coco/coco.lance",
        columns=["image", "annotations.category_id", "annotations.bbox"],
        batch_size=32,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        IterableWrapper(dataset).shuffle(buffer_size=64)
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    # Let PyTorch Lightning to use all GPUs
    trainer = pl.Trainer(accelerator="gpu", devices=-1)
    model = ...
    trainer.fit(model, train_dataloaders=train_loader)