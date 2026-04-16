# PyTorch 集成

机器学习用户可以使用 `lance.torch.data.LanceDataset`，这是 `torch.utils.data.IterableDataset` 的子类，
可以在 PyTorch 训练和推理循环中直接使用 Lance 数据。

首先需要创建一个用于训练的 ML 数据集。通过 [HuggingFace 集成](huggingface.md)，
只需一行 Python 代码即可将 HuggingFace 数据集转换为 Lance 数据集。

```python
import datasets # pip install datasets
import lance

hf_ds = datasets.load_dataset(
    "poloclub/diffusiondb",
    split="train",
    # name="2m_first_1k",  # for a smaller subset of the dataset
)
lance.write_dataset(hf_ds, "diffusiondb_train.lance")
```

然后，你可以在 PyTorch 训练和推理循环中使用 Lance 数据集。

注意：

1. PyTorch 数据集会自动将数据转换为 `torch.Tensor`

2. Lance 不是 fork 安全的。如果你使用多进程（Multiprocessing），请使用 spawn 方式。
安全的 DataLoader 使用的是 spawn 方法。

## 非安全 DataLoader

```python
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
```

## 安全 DataLoader

```python
from lance.torch.data import SafeLanceDataset, get_safe_loader

dataset = SafeLanceDataset(temp_lance_dataset)
# use spawn method to avoid fork-safe issue
loader = get_safe_loader(
    dataset,
    num_workers=2,
    batch_size=16,
    drop_last=False,
)

total_samples = 0
for batch in loader:
    total_samples += batch["id"].shape[0]
```

`lance.torch.data.LanceDataset` 可以与 `lance.sampler.Sampler` 类组合使用以控制采样策略。例如，你可以使用 `lance.sampler.ShardedFragmentSampler` 在分布式训练环境中使用。如果不指定，默认为全表扫描。

```python
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
```

可用的采样器（Sampler）：

- `lance.sampler.ShardedFragmentSampler`
- `lance.sampler.ShardedBatchSampler`

!!! warning

    对于多进程场景，你可能不应该使用 fork 方式，因为 Lance 内部是多线程的，而 fork 和多线程无法很好地协同工作。
    请参考[这个讨论](https://discuss.python.org/t/concerns-regarding-deprecation-of-fork-with-alive-threads/33555)。
