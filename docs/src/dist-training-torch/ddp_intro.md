
# Introduction to DDP Training with Lance


Lance is a modern columnar format designed for high-performance ML workloads. Its architecture enables fast random access and efficient scans. LanceDB's ecosystem provides tools that integrate this storage format with training frameworks, supporting the AI workflow from data storage to optimized training pipelines. 

This guide focuses on the PyTorch integration and demonstrates how to use Lance to build efficient data loaders for Distributed Data Parallel (DDP) training.

## Data Loading Strategies: Map-Style vs. Iterable-Style

PyTorch offers two paradigms for creating datasets: map-style and iterable-style. Lance provides a corresponding class for each.

### Map-Style: `torch.utils.data.Dataset`
A map-style dataset is one that can be indexed and has a known length. It must implement the `__getitem__(self, index)` and `__len__(self)` methods. This design allows the `DataLoader` to fetch any specific item from the dataset by its index, enabling features like random shuffling and straightforward parallel loading across multiple workers. 

*   **Lance Integration:** `lance.torch.data.SafeLanceDataset`

### Iterable-Style: `torch.utils.data.IterableDataset`
An iterable-style dataset works like a Python generator. It implements the `__iter__(self)` method, which yields data items sequentially as a stream. It often has no known length and does not support random access to a specific index. This makes it ideal for handling data that doesn't fit the indexed collection model.

*   **Lance Integration:** `lance.torch.data.LanceDataset`

## Different pytorch Dataloading strategies

For most projects, the choice is between the high-performance map-style pattern and the flexible but slower simple iterable-style pattern. The table below provides a detailed comparison to help beginners select the best approach for their needs.

| Feature | Map-Style | Simple Iterable-Style |
| :--- | :--- | :--- |
| **Primary Use Case** | Standard, indexable datasets (e.g., image classification) | Streaming data, very large datasets, or custom sampling logic. |
| **Performance** | **Highest**. The recommended default for throughput. | **Lower**. I/O is serialized in a single process. |
| **Parallelism** | **High**. Data decoding is parallelized across multiple workers. | **Low**. Data is fetched and decoded sequentially in a single stream. |
| **Shuffling** | Natively supported via PyTorch samplers (e.g., `DistributedSampler`).  | Requires custom implementation (e.g., buffering data). |
| **Epoch Length** | Deterministic. The dataset has a known length (`__len__`).  | Not deterministic by default. Requires manual loop control. |
| **PyTorch `num_workers`** | `> 0` (Recommended for performance). | `0` (mutli-worker requires custom implementation). |
| **Sampler** | PyTorch `DistributedSampler`. | Custom (E.g. Lance `ShardedBatchSampler`. ) |

### Choosing Lance dataloader
It is recommended to use map‑style datasets by default in DDP setting. Map‑style give you their size ahead of time, easier to shuffle, and allow for easy parallel loading. A common use case for using iterable style dataset is when your dataset index doesn't fit into memory. 
Lance doesn't materialise the entire map in memory so using map-style dataset in almost all cases should work fin and is recommended. 

### Launching distributed training jobs

It is recommended to launch pytorch distributed training jobs using torchrun. Torchrun is a command-line tool provided by PyTorch for launching distributed training jobs. It offers:

1. Simplified Launch: Eliminates the need for manual environment variable setup or mp.spawn calls within your script.
2. Scalability: Facilitates scaling training from single-node to multi-node environments with minimal code changes.
3. Fault Tolerance: Enables robust training by allowing graceful restarts from snapshots.

**Running sinlge-node multi-gpu jobs**

```
torchrun --nproc_per_node=<N> your_training_script.py
```

**Multi node multi-gpu jobs**

For multi-node training, you need to specify the number of nodes (--nnodes), the rendezvous endpoint (--rdzv_endpoint), and the number of processes per node.

```
torchrun --nnodes=2 --nproc_per_node=4 --rdzv_endpoint="master_ip:29500" your_training_script.py
```

Learn more on [torchrun docs](https://docs.pytorch.org/docs/stable/elastic/run.html)