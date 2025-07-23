# Lance Samplers

Lance provides specialized samplers designed to work with the lance datasets. These samplers integrate tightly with the **iterable-style** `LanceDataset`, enabling efficient, balanced data distribution in both single-process and distributed (DDP) setups.


Overview of Lance samplers
- **ShardedBatchSampler**  
- **ShardedFragmentSampler**  
- **FullScanSampler**  

---


## ShardedBatchSampler

- **Purpose:** Distributes entire batches evenly across DDP ranks.
- **Mechanism:** Assigns batch indices round-robin: rank 0 gets batches 0, N, 2N...; 
                                                    rank 1 gets 1, N+1, ...
-  Balanced workload; prevents DDP deadlocks.
-  May perform small-range reads within fragments.

```python
from lance.torch.data import ShardedBatchSampler

sampler = ShardedBatchSampler(
    dataset_uri="data/FOOD101.lance",
    rank=rank,
    world_size=world_size
)
```

## ShardedFragmentSampler

- **Purpose**: Distributes entire fragments to DDP ranks.

- **Mechanism**: Splits fragments evenly among ranks (e.g., half of fragments to each rank in case of world-size = 2).
- Each rank reads whole fragments sequentially.
- High I/O throughput by reading contiguous data blocks.
- Fragments may vary in size if not carefully distributed; can cause workload imbalance and DDP synchronization delays or deadlock.

```python
from lance.torch.data import ShardedFragmentSampler

sampler = ShardedFragmentSampler(
    dataset_uri="data/FOOD101.lance",
    rank=rank,
    world_size=world_size
)

```

## FullScanSampler
- **Purpose**: Sequentially scans all fragments in a single-process scenario (e.g., inference or validation).

- **Mechanism**: Iterates through every fragment in dataset order without sharding.
- **Use Case**: Single-worker loops only. Not DDP-aware (will duplicate data if used with multiple processes).

```python

from lance.torch.data import FullScanSampler

sampler = FullScanSampler(
    dataset_uri="data/FOOD101.lance"
)
```