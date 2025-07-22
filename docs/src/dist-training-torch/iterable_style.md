# Training on Stream data with `LanceDataset` (Iterable-Style)

The basic iterable-style pattern is for use cases where you need to stream data sequentially or implement custom sampling logic. This can be due to various reasons like - your data is tool large to fit in a node or in a massively parallel multi-node multi-GPU setting, you need to read data from a central source, etc. It uses the `LanceDataset` and a Lance-native sampler.

**Important:** This simple pattern requires setting `num_workers=0` in the `DataLoader`, which means data will be loaded serially. This is significantly slower than the map-style approach and should only be used when the flexibility of iterable datasets is a requirement.

## Implementation Guide

1.  **Instantiate `LanceDataset`**: This is the iterable-style counterpart to `SafeLanceDataset`.

    ```python
    import lance
    from lance.torch.data import LanceDataset

    uri = "s3://my-bucket/path/to/my_dataset.lance"
    dataset = LanceDataset(uri, batch_size=32)
    ```

2.  **Create a `ShardedBatchSampler`**: Because `DistributedSampler` only works with map-style datasets, you must use Lance's `ShardedBatchSampler`. This sampler handles distributing batches to each DDP rank.

    ```python
    from lance.torch.sampler import ShardedBatchSampler

    # This sampler yields batches of keys for each DDP rank
    sampler = ShardedBatchSampler(dataset)
    ```

3.  **Create the `DataLoader`**: Combine the dataset and sampler. **Crucially, `num_workers` must be 0.**

    ```python
    from torch.utils.data import DataLoader

    # NOTE: num_workers MUST be 0 for this simple iterable pattern
    data_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    ```

4.  **Training Loop**: Unlike the map-style pattern, you do not need to call `set_epoch`. The sampler handles sharding automatically.

    ```python
    # Inside your training loop
    for epoch in range(num_epochs):
        for batch in data_loader:
            # Your training step here
            images, labels = batch
            # ... train on batch
    ```

## Complete Example

For a complete, runnable script, see the [iterable-style DDP example](https://github.com/lancedb/lance-distributed-training/blob/main/iterable_ddp.py) in our examples repository.