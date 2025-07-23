# Training on Stream data with `LanceDataset` (Iterable-Style)

The basic iterable-style pattern is for use cases where you need to stream data sequentially or implement custom sampling logic. This can be due to various reasons like - your data is tool large to fit in a node or in a massively parallel multi-node multi-GPU setting, you need to read data from a central source, etc. It uses the `LanceDataset` and a Lance-native sampler.

**Important:** This simple pattern requires setting `num_workers=0` in the `DataLoader`, which means data will be loaded serially. This is significantly slower than the map-style approach and should only be used when the flexibility of iterable datasets is a requirement.



## Implementation Guide

1.  **Create a `ShardedBatchSampler`**: Because `DistributedSampler` only works with map-style datasets, you must use Lance's `ShardedBatchSampler`. This sampler handles distributing batches to each DDP rank.

    ```python
    from lance.torch.sampler import ShardedBatchSampler

    # This sampler yields batches of keys for each DDP rank
    sampler = ShardedBatchSampler(dataset)
    ```

2.  **Instantiate `LanceDataset`**: This is the iterable-style counterpart to `SafeLanceDataset`.

    ```python
    import lance
    from lance.torch.data import LanceDataset

    uri = "path/to/my_dataset.lance"
    dataset = LanceDataset(uri, batch_size=32, sampler=sampler)
    ```

3.  **Create the `DataLoader`**: Combine the dataset and sampler. **Crucially, `num_workers` must be 0.**

    !!! Note
        You can implement an iterable style dataloader with many workers, but it involves sharding at two levels:
        Global Sharding (DDP Rank) - at the GPU rank level, and local sharding within Dataloaders the same global rank.
        Its both tricky to sync and also more expensive as you'll have to open many streams.


        

    ```python
    from torch.utils.data import DataLoader

    # NOTE: num_workers MUST be 0 for this simple iterable pattern

    data_loader = DataLoader(dataset, num_workers=0)
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

### Handling binary data

A key difference from the map-style pattern is that data transformations are typically applied before the data reaches the DataLoader, via a `to_tensor_fn` callable.

Since the DataLoader doesn't manage batching or decoding for iterable datasets, you must pass a `to_tensor_fn` when constructing LanceDataset handle/decode binary data. This function is called on each raw row (dictionary) as it’s streamed from disk.

```python
# Runs inside the main thread for each row
def decode_tensor_image(row):
    image = Image.open(io.BytesIO(row['image'])).convert("RGB")
    image_tensor = transforms.ToTensor()(image)
    label = torch.tensor(row['label'], dtype=torch.long)
    return {'image': image_tensor, 'label': label}

# Example setup

dataset = LanceDataset(
    uri="data/FOOD101.lance",
    batch_size=512,
    sampler=sampler,
    to_tensor_fn=decode_tensor_image
)
```

## Complete Example

For a complete, runnable script, see the [iterable-style DDP example](https://github.com/lancedb/lance-distributed-training/blob/main/iterable_ddp.py) in our examples repository.


### Further reading

See [this discussion](https://discuss.pytorch.org/t/iterable-pytorch-dataset-with-multiple-workers/135475/2) on pytorch forum to learn more about implementing Iterable style dataloaders with multiple workers