# 使用 Lance 创建用于 LLM 训练的文本数据集

Lance 可用于创建和缓存用于大语言模型（Large Language Model）预训练/微调的文本（或代码）数据集。
当需要在数据子集上训练模型或分块处理数据而不一次性将所有数据下载到磁盘时，这种需求就会出现。当你只需要 TB 或 PB 级别数据集的一个子集时，这会成为一个相当棘手的问题。

在本示例中，我们将通过分批下载文本数据集、对其进行分词并保存为 Lance 数据集来解决这个问题。
无论你需要多少或多少数据样本，平均内存消耗大约只有 3-4 GB！

本示例使用的是 [wikitext](https://huggingface.co/datasets/Salesforce/wikitext) 数据集，
这是一个从 Wikipedia 的优秀和精选文章中提取的超过 1 亿个 token 的集合。

## 准备和预处理原始数据集

首先定义数据集和分词器（Tokenizer）

```python
import lance
import pyarrow as pa

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm  # optional for progress tracking

tokenizer = AutoTokenizer.from_pretrained('gpt2')

dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', streaming=True)['train']
dataset = dataset.shuffle(seed=1337)
```

`load_dataset` 中的 `streaming` 参数尤为重要，因为如果不将其设置为 `True`，datasets 库会先下载整个数据集，即使你只想使用其中的一个子集。
当 `streaming` 设置为 `True` 时，样本将在需要时按需下载。

现在我们定义一个函数来帮助我们逐个对样本进行分词。

```python
def tokenize(sample, field='text'):
    return tokenizer(sample[field])['input_ids']
```

这个函数将接收 HuggingFace 数据集中的一个样本，并对 `field` 列中的值进行分词。这就是你想要分词的主要文本。

## 创建 Lance 数据集

现在我们已经设置好了原始数据集和预处理代码，
让我们定义主函数，它接收数据集、样本数量和字段，并返回一个 PyArrow 批次（Batch），稍后将被写入 Lance 数据集。

```python
def process_samples(dataset, num_samples=100_000, field='text'):
    current_sample = 0
    for sample in tqdm(dataset, total=num_samples):
        # If we have added all 5M samples, stop
        if current_sample == num_samples:
            break
        if not sample[field]:
            continue
        # Tokenize the current sample
        tokenized_sample = tokenize(sample, field)
        # Increment the counter
        current_sample += 1
        # Yield a PyArrow RecordBatch
        yield pa.RecordBatch.from_arrays(
            [tokenized_sample], 
            names=["input_ids"]
        )
```

这个函数会逐个遍历 HuggingFace 数据集中的样本，对样本进行分词并生成一个包含所有 token 的 PyArrow `RecordBatch`。我们会一直这样做，直到达到 `num_samples` 指定的样本数量或数据集末尾，以先到者为准。

请注意，这里所说的"样本"是指原始数据集中的一个示例（一行）。一个示例的具体含义取决于数据集本身，它可能是一行文本，也可能是整个文件。在本示例中，长度从一行到一段文本不等。

我们还需要定义一个 Schema 来告诉 Lance 我们期望表中的数据类型。由于我们的数据集仅包含长整数类型的 token，`int64` 是合适的数据类型。

```python
schema = pa.schema([
    pa.field("input_ids", pa.int64())
])
```

最后，我们需要定义一个 `reader`，它将从 `process_samples` 函数读取 Record Batch 流，该函数生成包含单个分词样本的 Record Batch。

```python
reader = pa.RecordBatchReader.from_batches(
    schema, 
    process_samples(dataset, num_samples=500_000, field='text') # For 500K samples
)
```

最终我们使用 `lance.write_dataset` 将数据集写入磁盘。

```python
# Write the dataset to disk
lance.write_dataset(
    reader, 
    "wikitext_500K.lance",
    schema
)
```

如果你想在保存到磁盘之前对 token 进行其他预处理（如掩码等），可以在 `process_samples` 函数中添加。

大功告成！你的数据集已完成分词并保存到磁盘！
