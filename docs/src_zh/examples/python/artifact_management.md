# 使用 Lance 进行深度学习制品管理

除了数据集之外，Lance 文件格式还可用于保存和版本管理深度学习模型权重。
实际上，使用 Lance 文件格式管理 PyTorch 模型权重，可以比普通的权重保存方法更加流畅。

在本示例中，我们将演示如何使用 Lance 保存、版本管理和加载 PyTorch 模型的权重。具体来说，我们将加载一个预训练的 ResNet 模型，将其保存为 Lance 文件格式，再加载回 PyTorch 并验证权重是否确实相同。
我们还将演示如何借助零拷贝自动版本管理（Zero-copy Automatic Versioning），在单个 Lance 数据集中对模型权重进行版本管理。

**核心思路：** 当你在 PyTorch 中保存模型的权重（即状态字典 state dictionary）时，权重以键值对的形式存储在 `OrderedDict` 中，其中键表示权重名称，值表示对应的权重张量（Tensor）。
为了尽可能地模拟这种结构，我们将权重保存为三列。第一列是权重名称，第二列是扁平化后的权重列表，第三列是权重的原始形状，以便在加载到模型时可以重新构建。

## 导入和设置

我们首先导入和加载所有必要的模块。

```python
import os
import shutil
import lance
import pyarrow as pa
import torch
from collections import OrderedDict
```

我们还将定义一个 `GLOBAL_SCHEMA`，它规定了权重表的结构。

```python
GLOBAL_SCHEMA = pa.schema(
    [
        pa.field("name", pa.string()),
        pa.field("value", pa.list_(pa.float64(), -1)),
        pa.field("shape", pa.list_(pa.int64(), -1)), # Is a list with variable shape because weights can have any number of dims
    ]
)
```

如前所述，权重表将有三列——一列存储权重名称，一列存储扁平化的权重值，一列存储原始权重形状以便重新加载。

## 保存和版本管理模型

首先我们关注模型保存部分。让我们编写一个工具函数，它接收模型的状态字典，遍历每个权重，将其扁平化，然后以 PyArrow `RecordBatch` 的形式返回权重名称、扁平化权重和权重的原始形状。

```python
def _save_model_writer(state_dict):
    """Yields a RecordBatch for each parameter in the model state dict"""
    for param_name, param in state_dict.items():
        param_shape = list(param.size())
        param_value = param.flatten().tolist()
        yield pa.RecordBatch.from_arrays(
            [
                pa.array(
                    [param_name],
                    pa.string(),
                ),
                pa.array(
                    [param_value],
                    pa.list_(pa.float64(), -1),
                ),
                pa.array(
                    [param_shape],
                    pa.list_(pa.int64(), -1),
                ),
            ],
            ["name", "value", "shape"],
        )
```

关于版本管理：假设你在一些新数据上训练了模型，但不想覆盖旧的检查点（Checkpoint），现在你可以将新训练的模型权重作为 Lance 权重数据集中的一个版本保存。
这样你就可以从一个 Lance 权重数据集中加载特定版本的权重，而不需要为每个模型检查点创建单独的文件夹。

让我们编写一个函数来处理模型保存的工作，无论是否需要版本管理。

```python
def save_model(state_dict: OrderedDict, file_name: str, version=False):
    """Saves a PyTorch model in lance file format

    Args:
        state_dict (OrderedDict): Model state dict
        file_name (str): Lance model name
        version (bool): Whether to save as a new version or overwrite the existing versions,
            if the lance file already exists
    """
    # Create a reader
    reader = pa.RecordBatchReader.from_batches(
        GLOBAL_SCHEMA, _save_model_writer(state_dict)
    )

    if os.path.exists(file_name):
        if version:
            # If we want versioning, we use the overwrite mode to create a new version
            lance.write_dataset(
                reader, file_name, schema=GLOBAL_SCHEMA, mode="overwrite"
            )
        else:
            # If we don't want versioning, we delete the existing file and write a new one
            shutil.rmtree(file_name)
            lance.write_dataset(reader, file_name, schema=GLOBAL_SCHEMA)
    else:
        # If the file doesn't exist, we write a new one
        lance.write_dataset(reader, file_name, schema=GLOBAL_SCHEMA)
```

上面的函数接收模型状态字典、Lance 保存文件名和权重版本。函数首先使用全局 Schema 和之前编写的工具函数创建一个 `RecordBatchReader`。
如果权重 Lance 数据集已经存在于目录中，我们会将其保存为新版本（如果启用了版本管理）或删除旧文件并保存为新的。否则将正常保存权重。

## 加载模型

从 Lance 权重数据集加载权重到模型中就是保存的逆过程。关键部分是将扁平化的权重重新恢复为原始形状，这很容易实现，因为你已经保存了对应的形状信息。
我们将其分为三个函数以提高可读性。

第一个函数是 `_load_weight` 函数，它接收从 Lance 权重数据集中获取的"权重"，并将其作为原始形状的 torch 张量返回。从 Lance 权重数据集中获取的"权重"将是一个字典，每列对应的值以键的形式存在。

```python
def _load_weight(weight: dict) -> torch.Tensor:
    """Converts a weight dict to a torch tensor"""
    return torch.tensor(weight["value"], dtype=torch.float64).reshape(weight["shape"])
```

你还可以选择添加指定权重数据类型的选项。

下一个函数是将 Lance 权重数据集中的所有权重加载到状态字典中，这就是 PyTorch 在将权重加载到模型时所期望的格式。

```python
def _load_state_dict(file_name: str, version: int = 1, map_location=None) -> OrderedDict:
    """Reads the model weights from lance file and returns a model state dict
    If the model weights are too large, this function will fail with a memory error.

    Args:
        file_name (str): Lance model name
        version (int): Version of the model to load
        map_location (str): Device to load the model on

    Returns:
        OrderedDict: Model state dict
    """
    ds = lance.dataset(file_name, version=version)
    weights = ds.take([x for x in range(ds.count_rows())]).to_pylist()
    state_dict = OrderedDict()

    for weight in weights:
        state_dict[weight["name"]] = _load_weight(weight).to(map_location)

    return state_dict
```

`load_state_dict` 函数需要 Lance 权重数据集文件名、版本号和权重加载的目标设备。
我们将 Lance 权重数据集中的所有权重加载到内存中，然后使用前面编写的工具函数将它们逐个转换为权重，并放到指定设备上。

需要注意的是，如果保存的权重大于内存，此函数将会失败。为简单起见，我们假设要加载的权重可以放入内存中，不需要处理分片（Sharding）问题。

最后，我们编写一个更高层的函数，这是我们唯一需要调用来加载权重的函数。

```python
def load_model(
    model: torch.nn.Module, file_name: str, version: int = 1, map_location=None
):
    """Loads the model weights from lance file and sets them to the model

    Args:
        model (torch.nn.Module): PyTorch model
        file_name (str): Lance model name
        version (int): Version of the model to load
        map_location (str): Device to load the model on
    """
    state_dict = _load_state_dict(file_name, version=version, map_location=map_location)
    model.load_state_dict(state_dict)
```

`load_model` 函数需要模型、Lance 权重数据集名称、要加载的权重版本和设备映射位置。它只需调用 `_load_state_dict` 工具来获取状态字典，然后将该状态字典加载到模型中。

## 总结

总而言之，你只需要调用两个函数：`save_model` 和 `load_model` 来分别保存和加载模型，只要权重可以放入内存并且是 PyTorch 格式的就可以。

虽然目前还是实验性的，但这种方法为深度学习制品管理定义了一种新方式。
