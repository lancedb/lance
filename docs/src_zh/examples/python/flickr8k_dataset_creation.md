# 使用 Lance 创建多模态数据集

得益于 Lance 文件格式存储不同模态数据的能力，Lance 最重要的使用场景之一就是存储多模态（Multi-Modal）数据集。
在这个简短的示例中，我们将介绍如何获取一个多模态数据集并将其存储为 Lance 文件格式。

本示例选择的数据集是 [Flickr8k 数据集](https://github.com/goodwillyoga/Flickr8k_dataset)。Flickr8k 是一个基于句子的图像描述和搜索基准集合，包含 8,000 张图像，每张图像配有五个不同的标题，提供对显著实体和事件的清晰描述。
这些图像是从六个不同的 Flickr 群组中选择的，通常不包含任何知名人物或地点，但经过人工筛选以展示各种场景和情境。

我们将使用上述 Flickr8k 数据集创建一个图像-标题对数据集，用于多模态模型训练，并以 Lance 数据集的形式保存，包含图像文件名、每张图像的所有标题（保持顺序）以及图像本身（二进制格式）。

## 导入和设置

我们假设你已经下载了数据集，具体来说是 "Flickr8k.token.txt" 文件和 "Flicker8k_Dataset/" 文件夹，并且两者都在当前目录中。
可以从[这里](https://github.com/goodwillyoga/Flickr8k_dataset?tab=readme-ov-file)下载（同时下载数据集和文本压缩文件）。

我们还假设你已经安装了 pyarrow 和 pylance，以及 opencv（用于读取图像）和 tqdm（用于进度条显示）。

现在让我们从导入开始，并定义标题文件和图像数据集文件夹。

```python
import os
import cv2
import random

import lance
import pyarrow as pa

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

captions = "Flickr8k.token.txt"
image_folder = "Flicker8k_Dataset/"
```

## 加载和处理

在 Flickr8k 数据集中，每张图像有多个对应的有序标题。
我们将把每张图像的所有标题放入一个列表中，列表中的位置代表它们原始出现的顺序。
让我们将标注（图像路径和对应标题）加载到一个列表中，列表的每个元素是一个由图像名称、标题编号和标题本身组成的元组。

```python
with open(captions, "r") as fl:
    annotations = fl.readlines()

# Converts the annotations where each element of this list is a tuple consisting of image file name, caption number and caption itself
annotations = list(map(lambda x: tuple([*x.split('\t')[0].split('#'), x.split('\t')[1]]), annotations))
```

现在，对于同一张图像的所有标题，我们将它们放入一个按顺序排列的列表中。

```python
captions = []
image_ids = set(ann[0] for ann in annotations)
for img_id in tqdm(image_ids):
    current_img_captions = []
    for ann_img_id, num, caption in annotations:
        if img_id == ann_img_id:
            current_img_captions.append((num, caption))
            
    # Sort by the annotation number
    current_img_captions.sort(key=lambda x: x[0])
    captions.append((img_id, tuple([x[1] for x in current_img_captions])))
```

## 转换为 Lance 数据集

现在我们的标题列表格式正确了，我们将编写一个 `process()` 函数，它接收标题列表作为参数，并生成包含 `image_id`、`image` 和 `captions` 的 PyArrow Record Batch。
Record Batch 中的图像将以二进制格式存储，每张图像的所有标题将在一个列表中并保持顺序。

```python
def process(captions):
    for img_id, img_captions in tqdm(captions):
        try:
            with open(os.path.join(image_folder, img_id), 'rb') as im:
                binary_im = im.read()
                
        except FileNotFoundError:
            print(f"img_id '{img_id}' not found in the folder, skipping.")
            continue
        
        img_id = pa.array([img_id], type=pa.string())
        img = pa.array([binary_im], type=pa.binary())
        capt = pa.array([img_captions], pa.list_(pa.string(), -1))
        
        yield pa.RecordBatch.from_arrays(
            [img_id, img, capt], 
            ["image_id", "image", "captions"]
        )
```

我们还需要定义 Schema 来告诉 PyArrow 表中预期的数据类型。

```python
schema = pa.schema([
    pa.field("image_id", pa.string()),
    pa.field("image", pa.binary()),
    pa.field("captions", pa.list_(pa.string(), -1)),
])
```

我们包含了 `image_id`（即原始图像名称），以便将来更容易引用和调试。

最后，我们定义一个 reader 来迭代读取 Record Batch，然后将它们写入磁盘上的 Lance 数据集。

```python
reader = pa.RecordBatchReader.from_batches(schema, process(captions))
lance.write_dataset(reader, "flickr8k.lance", schema)
```

基本就是这些了！如果你想在 Notebook 中执行此操作，可以在我们的 deeplearning-recipes 仓库中查看这个示例，地址在[这里](https://github.com/lancedb/lance-deeplearning-recipes/tree/main/examples/flickr8k-dataset)。

更多使用 Lance 数据集的深度学习示例，请务必查看 [lance-deeplearning-recipes](https://github.com/lancedb/lance-deeplearning-recipes) 仓库！
