# Tensorflow 集成

Lance 可以作为常规的 [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
在 [Tensorflow](https://www.tensorflow.org/) 中使用。

!!! warning

    此功能处于实验阶段，API 可能会在未来发生变化。

## 从 Lance 读取数据

使用 `lance.tf.data.from_lance`，你可以轻松创建 `tf.data.Dataset`。

```python
import tensorflow as tf
import lance

# Create tf dataset
ds = lance.tf.data.from_lance("s3://my-bucket/my-dataset")

# Chain tf dataset with other tf primitives

for batch in ds.shuffling(32).map(lambda x: tf.io.decode_png(x["image"])):
    print(batch)
```

基于 Lance [列式格式](../format/index.md)，使用 `lance.tf.data.from_lance` 支持高效的列选择、过滤等操作。

```python
ds = lance.tf.data.from_lance(
    "s3://my-bucket/my-dataset",
    columns=["image", "label"],
    filter="split = 'train' AND collected_time > timestamp '2020-01-01'",
    batch_size=256)
```

默认情况下，Lance 会从投影列推断 Tensor 规格。你也可以手动指定 `tf.TensorSpec`。

```python
batch_size = 256
ds = lance.tf.data.from_lance(
    "s3://my-bucket/my-dataset",
    columns=["image", "labels"],
    batch_size=batch_size,
    output_signature={
        "image": tf.TensorSpec(shape=(), dtype=tf.string),
        "labels": tf.RaggedTensorSpec(
            dtype=tf.int32, shape=(batch_size, None), ragged_rank=1),
    },
```

## 分布式训练和数据打乱

由于 [Lance 数据集是一组 Fragment](../format/index.md)，我们可以将 Fragment 分发和打乱到不同的 Worker。

```python
import tensorflow as tf
from lance.tf.data import from_lance, lance_fragments

world_size = 32
rank = 10
seed = 123  #
epoch = 100

dataset_uri = "s3://my-bucket/my-dataset"

# Shuffle fragments distributedly.
fragments =
    lance_fragments("s3://my-bucket/my-dataset")
    .shuffling(32, seed=seed)
    .repeat(epoch)
    .enumerate()
    .filter(lambda i, _: i % world_size == rank)
    .map(lambda _, fid: fid)

ds = from_lance(
    uri,
    columns=["image", "label"],
    fragments=fragments,
    batch_size=32
    )
for batch in ds:
    print(batch)
```

!!! warning

    对于多进程场景，你可能不应该使用 fork 方式，因为 Lance 内部是多线程的，而 fork 和多线程无法很好地协同工作。
    请参考[这个讨论](https://discuss.python.org/t/concerns-regarding-deprecation-of-fork-with-alive-threads/33555)。
