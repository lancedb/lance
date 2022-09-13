#!/usr/bin/env python3

"""Generate embeddings"""

import io
import os

import click
import pandas as pd
import pyarrow
import torch
import torchvision
from common import Classification, RawOxfordPetDataset, raw_collate_fn
import PIL
from torchvision.models.feature_extraction import create_feature_extractor

import lance

transform = torchvision.models.ResNet50_Weights.DEFAULT.transforms()


def collate_fn(batch):
    pk = batch[1]
    images = [
        transform(PIL.Image.open(io.BytesIO(data)).convert("RGB")) for data in batch[0]
    ]
    return torch.stack(images), pk


@click.command()
@click.argument("checkpoint")
@click.argument("dataset")
@click.option(
    "-f", "--format", "data_format", type=click.Choice(["lance", "raw", "parquet"])
)
@click.option("-b", "--batch_size", type=int, default=128)
@click.option(
    "-w",
    "--num_workers",
    default=os.cpu_count(),
    help="set pytorch DataLoader number of workers",
    show_default=True,
)
@click.option(
    "-o", "--output", default="embeddings.lance", help="Output path", show_default=True
)
def gen_embeddings(checkpoint, dataset, output, batch_size, num_workers, data_format):
    model = Classification.load_from_checkpoint(checkpoint)

    if data_format == "lance":
        dataset = lance.pytorch.data.LanceDataset(
            dataset,
            columns=["image", "filename"],
            batch_size=batch_size,
            # filter=(pc.field("split") == "train")
        )
        train_loader = torch.utils.data.DataLoader(
            dataset, num_workers=num_workers, batch_size=None, collate_fn=collate_fn
        )
    elif data_format == "raw":
        dataset = RawOxfordPetDataset(dataset, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=raw_collate_fn,
        )
    else:
        raise ValueError("Unsupported data format")
    model.eval()

    extractor = create_feature_extractor(model.backbone, {"avgpool": "features"})
    extractor = extractor.to("cuda")
    with torch.no_grad():
        dfs = []
        for batch, pk in train_loader:
            batch = batch.to("cuda")
            features = extractor(batch)["features"].squeeze()
            df = pd.DataFrame(
                {
                    "pk": pk,
                    "features": features.tolist(),
                }
            )
            dfs.append(df)
        df = pd.concat(dfs)
        lance.write_table(pyarrow.Table.from_pandas(df, preserve_index=False), output)


if __name__ == "__main__":
    gen_embeddings()
