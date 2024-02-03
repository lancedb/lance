PyTorch Integration
-------------------

Machine learning users can use :class:`~lance.torch.data.LanceDataset`, a
subclass of :class:`torch.utils.data.IterableDataset`, that to use
Lance data directly PyTorch training and inference loops.


It starts with creating a ML dataset for training. With the :doc:`./huggingface`,
it takes just one line of Python to convert a HuggingFace dataset to a Lance dataset.

.. code-block:: python

    # Huggingface datasets
    import datasets
    import lance

    hf_ds = datasets.load_dataset(
        "poloclub/diffusiondb",
        split="train",
        # name="2m_first_1k",  # for a smaller subset of the dataset
    )
    lance.write_dataset(hf_ds, "diffusiondb_train.lance")

Then, you can use the Lance dataset in PyTorch training and inference loops.
Not that the PyTorch dataset automatically convert data into :class:`torch.Tensor`

.. code-block:: python

    import torch
    import lance.torch.data

    # Load lance dataset into a PyTorch IterableDataset.
    # with only columns "image" and "prompt".
    dataset = lance.torch.data.LanceDataset(
        "diffusiondb_train.lance",
        columns=["image", "prompt"],
        batch_size=128,
        batch_readahead=8,  # Control multi-threading reads.
    )

    # Create a PyTorch DataLoader
    dataloader = torch.utils.data.DataLoader(dataset)

    # Inference loop
    for batch in dataloader:
        inputs, targets = batch["prompt"], batch["image"]
        outputs = model(inputs)
        ...

:class:`~lance.torch.data.LanceDataset` can composite with the :class:`~lance.sampler.Sampler` classes
to control the sampling strategy. For example, you can use :class:`~lance.sampler.ShardedFragmentSampler`
to use it in a distributed training environment. If not specified, it is a full scan.

.. code-block:: python

    from lance.sampler import ShardedFragmentSampler
    from lance.torch.data import LanceDataset

    # Load lance dataset into a PyTorch IterableDataset.
    # with only columns "image" and "prompt".
    dataset = LanceDataset(
        "diffusiondb_train.lance",
        columns=["image", "prompt"],
        batch_size=128,
        batch_readahead=8,  # Control multi-threading reads.
        sampler=ShardedFragmentSampler(
            rank=1,  # Rank of the current process
            world_size=8,  # Total number of processes
        ),
    )

Available samplers:

- :class:`lance.sampler.ShardedFragmentSampler`
- :class:`lance.sampler.ShardedBatchSampler`
