from datasets import load_dataset, DownloadConfig

import lance
import pyarrow as pa
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_transforms = v2.Compose([
    v2.PILToTensor(),
    v2.Resize((256, 256)),
    v2.ToDtype(torch.float32),
    # https://discuss.pytorch.org/t/discussion-why-normalise-according-to-imagenet-mean-and-std-dev-for-transfer-learning/115670
    # scale to 256 because the original channels were on Z[0, 256)
    v2.Normalize(
        mean=[0.485 * 256, 0.456 * 256, 0.406 * 256],
        std=[0.229 * 256, 0.224 * 256, 0.225 * 256],
    ),
])

def _convert(data):
    return {
        "image": torch.stack([_transforms(x) for x in data['image']]).reshape(-1, 3, 256, 256).half(),
        "label": torch.tensor(data["label"]).reshape(-1, 1),
    }

def load_huggingface_imagenet_1k():
    return load_dataset(
        "imagenet-1k",
        cache_dir=".",
        download_config=DownloadConfig(num_proc=16, cache_dir="."),
        data_dir=".",
        num_proc=24,
    )

def _get_imagenet_1k_dataloader():
    dataset = load_huggingface_imagenet_1k()

    torch_ds = dataset["train"].with_format("torch").with_transform(_convert)

    dl = DataLoader(torch_ds, batch_size=1024, num_workers=16)

    return dl

_schema = pa.schema({
    "image": pa.list_(pa.float16(), 3 * 256 * 256),
    "label": pa.uint32(),
})

def _dump_to_lance():
    dl = _get_imagenet_1k_dataloader()
    def _iter():
        for batch in tqdm(dl):
            import pdb; pdb.set_trace()
            images = batch["image"].numpy().astype("float16").reshape(-1)
            labels = batch["label"].numpy().astype("uint32")
            image_fsl = pa.FixedSizeListArray.from_arrays(pa.array(images), 3 * 256 * 256)
            yield pa.RecordBatch.from_arrays([image_fsl, pa.array(labels)], schema=_schema)

    lance.write_dataset(_iter(), "imagenet.lance", schema=_schema)

if __name__ == "__main__":
    print("generating imagenet-1k dataset in lance format...")
    _dump_to_lance()
