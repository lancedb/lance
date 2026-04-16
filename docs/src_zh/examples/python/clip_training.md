# 使用 Lance 数据集训练多模态模型

在本示例中，我们将使用 Lance 图像-文本数据集训练一个 CLIP 模型，用于基于自然图像的搜索。
具体来说，我们将使用 [flickr_8k Lance 数据集](https://www.kaggle.com/datasets/heyytanay/flickr-8k-lance)。

模型架构和部分训练代码改编自 Manan Goel 的 [Implementing CLIP with PyTorch Lightning](https://wandb.ai/manan-goel/coco-clip/reports/Implementing-CLIP-With-PyTorch-Lightning--VmlldzoyMzg4Njk1)，并做了必要的修改以适配一个最小化的、兼容 Lance 的训练示例。

## 导入和设置

除了 Lance 之外，我们还需要 [PyTorch](https://pytorch.org/get-started/locally/) 和 [timm](https://github.com/huggingface/pytorch-image-models) 来训练我们的 CLIP 模型。

```python
import cv2
import lance

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
from transformers import AutoModel, AutoTokenizer

import itertools
from tqdm import tqdm

import warnings
warnings.simplefilter('ignore')
```

现在，我们将定义一个 Config 类来存放训练所需的所有超参数（Hyperparameters）。

```python
class Config:
    img_size = (128, 128)
    bs = 32
    head_lr = 1e-3
    img_enc_lr = 1e-4
    text_enc_lr = 1e-5
    max_len = 18
    img_embed_dim = 2048
    text_embed_dim = 768
    projection_dim = 256
    temperature = 1.0
    num_epochs = 2
    img_encoder_model = 'resnet50'
    text_encoder_model = 'bert-base-cased'
```

还需要两个工具函数来帮助我们从 Lance 数据集中加载图像和文本。
请记住，我们的 Lance 数据集包含图像、图像名称和给定图像的所有标题。我们只需要图像和其中一个标题。
为了简单起见，在加载标题时，我们会选择最长的那个（基于一个较为简单的假设：最长的标题包含关于图像的更多信息）。

```python
def load_image(ds, idx):
    # Utility function to load an image at an index and convert it from bytes format to img format
    raw_img = ds.take([idx], columns=['image']).to_pydict()
    raw_img = np.frombuffer(b''.join(raw_img['image']), dtype=np.uint8)
    img = cv2.imdecode(raw_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def load_caption(ds, idx):
    # Utility function to load an image's caption. Currently we return the longest caption of all
    captions = ds.take([idx], columns=['captions']).to_pydict()['captions'][0]
    return max(captions, key=len)
```

由于图像在 Lance 数据集中以字节形式存储，`load_image()` 函数会加载对应图像的字节数据，然后使用 numpy 和 opencv 将其转换为图像。

## 数据集和数据增强

由于我们的 CLIP 模型要求相同尺寸的图像和分词后的标题，我们将定义一个自定义 PyTorch Dataset，它接收 Lance 数据集路径以及任何数据增强（针对图像），并返回预处理后的图像和分词后的标题（以字典形式）。

```python
class CLIPLanceDataset(Dataset):
    """Custom Dataset to load images and their corresponding captions"""
    def __init__(self, lance_path, max_len=18, tokenizer=None, transforms=None):
        self.ds = lance.dataset(lance_path)
        self.max_len = max_len
        # Init a new tokenizer if not specified already
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased') if not tokenizer else tokenizer
        self.transforms = transforms

    def __len__(self):
        return self.ds.count_rows()

    def __getitem__(self, idx):
        # Load the image and caption
        img = load_image(self.ds, idx)
        caption = load_caption(self.ds, idx)

        # Apply transformations to the images
        if self.transforms:
            img = self.transforms(img)

        # Tokenize the caption
        caption = self.tokenizer(
            caption,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        # Flatten each component of tokenized caption otherwise they will cause size mismatch errors during training
        caption = {k: v.flatten() for k, v in caption.items()}

        return img, caption
```

现在自定义数据集准备好了，我们还定义一些非常基础的图像增强操作。

```python
train_augments = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(Config.img_size),
        transforms.Normalize([0.5], [0.5]),
    ]
)
```

这些变换非常基础：将所有图像调整为相同尺寸，然后对它们进行归一化以稳定后续的训练。

## 模型和设置

由于我们训练的是 CLIP 模型，我们需要以下组件：
* `ImageEncoder` 使用预训练的视觉模型（本例中使用 `resnet50`）将图像转换为特征向量。
* `TextEncoder` 使用预训练的语言模型（本例中使用 `bert-base-cased`）将文本标题转换为特征向量。
* `Head` 即投影模块（Projection Module），将这些特征向量投影到公共嵌入空间（Embedding Space）中。

深入讨论 CLIP 模型的细节和架构特点超出了本示例的范围，如果你想了解更多，可以阅读官方论文，地址在[这里](https://arxiv.org/abs/2103.00020/)。

了解了模型的基本概况后，让我们定义所有需要的模块。

```python
class ImageEncoder(nn.Module):
    """Encodes the Image"""
    def __init__(self, model_name, pretrained = True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )

        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, img):
        return self.backbone(img)

class TextEncoder(nn.Module):
    """Encodes the Caption"""
    def __init__(self, model_name):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)

        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, captions):
        output = self.backbone(**captions)
        return output.last_hidden_state[:, 0, :]

class Head(nn.Module):
    """Projects both into Embedding space"""
    def __init__(self, embedding_dim, projection_dim):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)

        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected

        return self.layer_norm(x)
```

除了模型定义，我们还将定义两个工具函数来简化训练：`forward()` 对组合模型执行一次前向传播，`loss_fn()` 接收 `forward` 函数输出的图像和文本嵌入，然后计算损失。

```python
def loss_fn(img_embed, text_embed, temperature=0.2):
    """
    https://arxiv.org/abs/2103.00020/
    """
    # Calculate logits, image similarity and text similarity
    logits = (text_embed @ img_embed.T) / temperature
    img_sim = img_embed @ img_embed.T
    text_sim = text_embed @ text_embed.T
    # Calculate targets by taking the softmax of the similarities
    targets = F.softmax(
        (img_sim + text_sim) / 2 * temperature, dim=-1
    )
    img_loss = (-targets.T * nn.LogSoftmax(dim=-1)(logits.T)).sum(1)
    text_loss = (-targets * nn.LogSoftmax(dim=-1)(logits)).sum(1)
    return (img_loss + text_loss) / 2.0

def forward(img, caption):
    # Transfer to device
    img = img.to('cuda')
    for k, v in caption.items():
        caption[k] = v.to('cuda')

    # Get embeddings for both img and caption
    img_embed = img_head(img_encoder(img))
    text_embed = text_head(text_encoder(caption))

    return img_embed, text_embed
```

为了训练模型，我们将定义模型、分词器和优化器，供下一节使用。

```python
# Define image encoder, image head, text encoder, text head and a tokenizer for tokenizing the caption
img_encoder = ImageEncoder(model_name=Config.img_encoder_model).to('cuda')
img_head = Head(Config.img_embed_dim, Config.projection_dim).to('cuda')

tokenizer = AutoTokenizer.from_pretrained(Config.text_encoder_model)
text_encoder = TextEncoder(model_name=Config.text_encoder_model).to('cuda')
text_head = Head(Config.text_embed_dim, Config.projection_dim).to('cuda')

# Since we are optimizing two different models together, we will define parameters manually
parameters = [
    {"params": img_encoder.parameters(), "lr": Config.img_enc_lr},
    {"params": text_encoder.parameters(), "lr": Config.text_enc_lr},
    {
        "params": itertools.chain(
            img_head.parameters(),
            text_head.parameters(),
        ),
        "lr": Config.head_lr,
    },
]

optimizer = torch.optim.Adam(parameters)
```

## 训练

在实际训练模型之前，还剩最后一步：初始化我们的 Lance 数据集和 DataLoader。

```python
# We assume the flickr8k.lance dataset is in the same directory
dataset = CLIPLanceDataset(
    lance_path="flickr8k.lance",
    max_len=Config.max_len,
    tokenizer=tokenizer,
    transforms=train_augments
)

dataloader = DataLoader(
    dataset,
    shuffle=False,
    batch_size=Config.bs,
    pin_memory=True
)
```

DataLoader 初始化完成后，让我们开始训练模型。

```python
img_encoder.train()
img_head.train()
text_encoder.train()
text_head.train()

for epoch in range(Config.num_epochs):
    print(f"{'='*20} Epoch: {epoch+1} / {Config.num_epochs} {'='*20}")

    prog_bar = tqdm(dataloader)
    for img, caption in prog_bar:
        optimizer.zero_grad(set_to_none=True)

        img_embed, text_embed = forward(img, caption)
        loss = loss_fn(img_embed, text_embed, temperature=Config.temperature).mean()

        loss.backward()
        optimizer.step()

        prog_bar.set_description(f"loss: {loss.item():.4f}")
    print()
```

训练循环非常直观。我们将图像编码器、图像投影头、文本编码器和文本投影头设置为训练模式。
然后在每个 epoch 中，遍历 Lance 数据集，训练模型并将损失报告到进度条。

```console
==================== Epoch: 1 / 2 ====================
loss: 2.0799: 100%|██████████| 253/253 [02:14<00:00,  1.88it/s]

==================== Epoch: 2 / 2 ====================
loss: 1.3064: 100%|██████████| 253/253 [02:10<00:00,  1.94it/s]
```

基本就是这些了！使用 Lance 数据集训练任何类型的模型与使用其他类型的数据集非常相似，但它还带来了更快的速度和更便捷的使用体验！
