# 使用 Lance 文本数据集训练 LLM

使用 Lance 文本数据集进行大语言模型的预训练/微调既简单又节省内存。
本示例是 [使用 Lance 创建用于 LLM 训练的文本数据集](llm_dataset_creation.md) 的后续。如果还没有阅读过，请先查看该示例。

在本示例中，我们将使用 Hugging Face transformers 在之前创建的分词后的 "wikitext_500K" Lance 数据集上训练一个 LLM。

## 导入和设置

让我们先完成所有必要的导入和基本定义来设置我们的环境。

```python
import numpy as np
import lance

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

# We'll be training the pre-trained GPT2 model in this example
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Also define some hyperparameters
lr = 3e-4
nb_epochs = 10
block_size = 1024
batch_size = 8
device = 'cuda:0'
dataset_path = 'wikitext_500K.lance'
```

基本设置完成后，让我们定义自定义 Dataset 和 Sampler，用于从 Lance 数据集中流式传输 token。

## 数据加载设置

我们首先定义一个工具函数，帮助我们从 Lance 数据集中以"块"的方式加载任意数量的 token。

```python
def from_indices(dataset, indices):
    """Load the elements on given indices from the dataset"""
    chunk = dataset.take(indices).to_pylist()
    chunk = list(map(lambda x: x['input_ids'], chunk))
    return chunk
```

现在让我们定义自定义 Dataset 和 Sampler 来加载 token。

```python
class LanceDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        block_size,
    ):
        # Load the lance dataset from the saved path
        self.ds = lance.dataset(dataset_path)
        self.block_size = block_size

        # Doing this so the sampler never asks for an index at the end of text
        self.length = self.ds.count_rows() - block_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Generate a window of indices starting from the current idx to idx+block_size
        and return the tokens at those indices
        """
        window = np.arange(idx, idx + self.block_size)
        sample = from_indices(self.ds, window)

        return {"input_ids": torch.tensor(sample), "labels": torch.tensor(sample)}
```

当采样器给出一个随机索引时，Dataset 会从当前索引开始加载接下来 `block_size` 个 token。
这实际上构成了一个样本，因为加载的 token 是因果连续的。

但我们还需要确保从数据集获取的 token 不会重叠。让我们通过一个例子来理解：

假设对于某个任意的 block size，在训练循环中数据集返回以下 token：

`"Vienna is the capital of Austria"` 在 index = 12 处作为样本 #1，以及

`"is the capital of Austria and"` 在 index = 13 处作为样本 #2，以此类推

这里的问题是，如果我们允许 DataLoader 获取任意数量索引的"样本"，它们可能会重叠（如上所示）。
这对模型来说不好，因为在看到足够多的重叠 token 后可能会开始过拟合（Overfitting）。

为了解决这个问题，我们定义了一个自定义 Sampler，它只返回彼此相隔 `block_size` 的索引，确保我们不会看到任何重叠的样本。

```python
class LanceSampler(Sampler):
    r"""Samples tokens randomly but `block_size` indices apart.

    Args:
        data_source (Dataset): dataset to sample from
        block_size (int): minimum index distance between each random sample
    """

    def __init__(self, data_source, block_size=512):
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.available_indices = list(range(0, self.num_samples, block_size))
        np.random.shuffle(self.available_indices)

    def __iter__(self):
        yield from self.available_indices

    def __len__(self) -> int:
        return len(self.available_indices)
```

现在当我们使用 `LanceSampler` 作为采样器从数据集中获取 token 时，模型在训练过程中看到的所有批次中的所有样本都保证是不重叠的。

实现方式是生成一个从 0 到数据集末尾（也就是 Lance 数据集长度减去 block size）的索引列表，每个索引之间相隔 `block_size`。然后我们打乱这个列表并从中生成索引。

数据加载部分基本就是这些了！现在我们只需要训练模型！

## 模型训练

现在你可以像使用任何其他数据集一样训练模型！

```python
# Define the dataset, sampler and dataloader
dataset = LanceDataset(dataset_path, block_size)
sampler = LanceSampler(dataset, block_size)
dataloader = DataLoader(
    dataset,
    shuffle=False,
    batch_size=batch_size,
    sampler=sampler,
    pin_memory=True
)

# Define the optimizer, training loop and train the model!
model = model.to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for epoch in range(nb_epochs):
    print(f"========= Epoch: {epoch+1} / {nb_epochs} =========")
    epoch_loss = []
    prog_bar = tqdm(dataloader, total=len(dataloader))
    for batch in prog_bar:
        optimizer.zero_grad(set_to_none=True)

        # Put both input_ids and labels to the device
        for k, v in batch.items():
            batch[k] = v.to(device)

        # Perform one forward pass and get the loss
        outputs = model(**batch)
        loss = outputs.loss

        # Perform backward pass
        loss.backward()
        optimizer.step()

        prog_bar.set_description(f"loss: {loss.item():.4f}")

        epoch_loss.append(loss.item())

    # Calculate training perplexity for this epoch
    try:
        perplexity = np.exp(np.mean(epoch_loss))
    except OverflowError:
        perplexity = float("-inf")

    print(f"train_perplexity: {perplexity}")
```

一个提示：如果你的 Lance 数据集很大（像 wikitext_500K 一样），并且你想调试模型以查找错误，可以将 DataLoader 包装在 `iter()` 函数中，只运行几个批次。

基本就是这些了！

使用 Lance、自定义 Dataset 和 Sampler 的最大优势是，得益于 Lance 提供的极速随机访问（Random Access），你可以获得高达 **95%** 的平均 GPU 利用率和极低的 CPU 开销。
