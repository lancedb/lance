import pathlib
import shutil

import lance
import pyarrow as pa
import torch
import torchvision.models as M
from torchvision.models.feature_extraction import create_feature_extractor

from lance.pytorch.data import Dataset


def _run_inference(uri, model, transforms):
    dataset = Dataset(uri, columns=["image_id", "image"], mode="batch", batch_size=1)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    )
    m = create_feature_extractor(model(weights="DEFAULT"), {"avgpool": "features"}).to(device)
    m.eval()

    image_ids = []
    embeddings = []
    with torch.no_grad():
        for batch in dataset:
            imgs = torch.stack([transforms(img) for img in batch[1]]).to(device)
            resnet_out = m(imgs)["features"].squeeze()
            features = torch.softmax(resnet_out, dim=0).cpu()
            embeddings.append(features.tolist())
            image_ids.extend(batch[0])

    emb_type = pa.list_(pa.float32(), list_size=2048)
    arrays = [pa.array(image_ids), pa.array(embeddings, type=emb_type)]
    schema = pa.schema([pa.field("image_id", pa.string()),
                        pa.field("embedding", emb_type)])

    tbl = pa.Table.from_arrays(arrays, schema=schema)
    return tbl


def compute_embeddings(dataset, model_name, uri, use_cache=True, cache_dir_base="/tmp"):
    cache_dir = pathlib.Path(cache_dir_base) / f"{model_name}_{dataset}_embeddings.lance"
    if not use_cache:
        shutil.rmtree(str(cache_dir), ignore_errors=True)

    model, weights = {
        "resnet": (M.resnet50, M.ResNet50_Weights),
        "vit": (M.vit_b_16, M.ViT_L_16_Weights),
    }[model_name]

    if not use_cache or not cache_dir.exists():
        result_table = _run_inference(uri, model, weights.DEFAULT.transforms())
        lance.write_dataset(result_table, str(cache_dir))
    else:
        result_table = lance.dataset(str(cache_dir)).to_table()
    return result_table

