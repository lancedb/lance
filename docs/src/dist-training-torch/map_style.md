# Training with `SafeLanceDataset` (Map-Style)

For any training scenarios where the dataset is indexed and can be randomly accesed, the map-style `SafeLanceDataset` is recommended. It integrates with PyTorch's `DataLoader` and `DistributedSampler` to deliver high data loading throughput with minimal configuration.

## The Map-Style Paradigm

The performance of this pattern comes from the design of map-style datasets. `SafeLanceDataset` implements `__len__` and `__getitem__`, providing the `DataLoader` with the total dataset size and a way to access any item by its index.

In a DDP setting, the `DistributedSampler` assigns a unique, non-overlapping subset of indices to each GPU process. The `DataLoader` (with `num_workers > 0`) distributes these indices among its worker subprocesses. Each worker independently calls `__getitem__` to fetch data in parallel, which is the basis for high-throughput data loading in PyTorch.

## Implementation Guide

Setting up a DDP-aware data loader with `SafeLanceDataset` involves four main steps.

1.  **`SafeLanceDataset`**: Create an instance of the dataset by pointing it to your `.lance` dataset URI.

    ```python
    import lance
    from lance.torch.data import SafeLanceDataset

    # URI to your Lance dataset
    uri = "path/to/my_dataset.lance"

    # Create the map-style dataset
    dataset = SafeLanceDataset(uri)
    ```

2.  **Create a PyTorch `DistributedSampler`**: The sampler is responsible for ensuring each DDP process (GPU) receives a different shard of the data.

    ```python
    from torch.utils.data.distributed import DistributedSampler

    # The sampler will automatically use the DDP rank and world size
    sampler = DistributedSampler(dataset)
    ```

3.  **Create the `DataLoader`**: Combine the dataset and sampler in a `DataLoader`. Set `num_workers > 0` to enable parallel data loading.

    ```python
    from torch.utils.data import DataLoader

    # Use multiple workers for parallel I/O
    # Set pin_memory=True for faster CPU-to-GPU memory transfers
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=32,
        num_workers=4,
        pin_memory=True
    )
    ```

4.  **Update Sampler Epoch**: In your training loop, you must call `sampler.set_epoch(epoch)` at the beginning of each epoch. This is crucial for proper shuffling across epochs.

    ```python
    # Inside training loop
    for epoch in range(num_epochs):
        # IMPORTANT: Ensure shuffling is different each epoch
        sampler.set_epoch(epoch)

        for batch in data_loader:
            # Your training step here
            images, labels = batch
            # ... train on batch
    ```


### Handling binary data

In the map-style (`SafeLanceDataset`), the DataLoader's workers fetch the raw data, and the transformation happens later in the `collate_fn`

```
def collate_fn(batch_of_dicts):
    """
    Collates a list of dictionaries from SafeLanceDataset into a single batch.
    This function handles decoding the image bytes and applying transforms.
    """
    images = []
    labels = []
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    for item in batch_of_dicts:
        image_bytes = item["image"]
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(img)
        images.append(img_tensor)
        labels.append(item["label"])
        
    return {
        "image": torch.stack(images),
        "label": torch.tensor(labels, dtype=torch.long)
    }

loader = get_safe_loader(
        dataset,
        sampler=sampler,
        collate_fn=collate_fn,
    )
```


## Complete Example

For a complete, runnable script demonstrating this pattern, please see the [map-style DDP example](https://github.com/lancedb/lance-distributed-training/blob/main/lance_map_style.py) in our examples repository.