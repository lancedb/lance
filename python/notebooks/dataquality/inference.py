import pathlib
import shutil

import lance
import pandas as pd
import pyarrow as pa
import torch
import torchvision.models as M

from lance.pytorch import Dataset


def _run_inference(uri, model, transform, col_name) -> pa.Table:
    dataset = Dataset(uri, columns=["image_id", "image"], mode="batch", batch_size=1)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    )
    m = model(weights="DEFAULT").to(device)
    m.eval()
    results = []
    with torch.no_grad():
        for batch in dataset:
            imgs = [transform(img).to(device) for img in batch[1]]
            prediction = m(torch.stack(imgs)).squeeze(0).softmax(0)
            topk = torch.topk(prediction, 2)
            for pk, scores, indices in zip(
                    batch[0], [topk.values.tolist()], [topk.indices.tolist()]
            ):
                results.append({
                    "image_id": pk.item(),
                    col_name: {
                        "score": scores[0],
                        "label": indices[0],
                        "second_score": scores[1],
                        "second_label": indices[1]
                    }
                })
    df = pd.DataFrame(data=results)
    return pa.Table.from_pandas(df)


def run_model(name, uri, use_cache=True, cache_dir_base="/tmp"):
    cache_dir = pathlib.Path(cache_dir_base) / f"{name}.lance"
    if not use_cache:
        shutil.rmtree(str(cache_dir), ignore_errors=True)

    model, weights = {
        "resnet": (M.resnet50, M.ResNet50_Weights),
        "vit": (M.vit_b_16, M.ViT_L_16_Weights),
    }[name]

    if not use_cache or not cache_dir.exists():
        result_table = _run_inference(
            uri, model, weights.DEFAULT.transforms(), name
        )
        lance.write_dataset(result_table, str(cache_dir))
    else:
        result_table = lance.dataset(str(cache_dir)).to_table()
    return result_table
