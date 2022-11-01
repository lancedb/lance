#!/usr/bin/env python3

from pathlib import Path
import pytest

import pandas as pd
import torch
import pyarrow as pa

from PIL import Image
from duckdb import DuckDBPyConnection
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


def download_model(tmp_path: Path, device: str):
    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    resnet.to(device)
    resnet.eval()
    m = torch.jit.script(resnet)
    model_path = tmp_path / "resnet.pth"
    torch.jit.save(m, str(model_path))
    return resnet, model_path


def create_model(db, model_path, device):
    db.execute(f"CALL create_pytorch_model('resnet', '{str(model_path)}', '{device}');")
    expected_models = pd.DataFrame([{
        "name": "resnet",
        "uri": str(model_path),
        "type": "torchscript",
        "device": device,
    }])
    pd.testing.assert_frame_equal(db.query("SELECT * FROM ml_models()").to_df(), expected_models)


def run_model(db, resnet, device):
    cat_path = Path(__file__).parent / "testdata" / "cat.jpg"
    cat = cat_path.read_bytes()
    tbl = pa.Table.from_pylist([{"img": cat}])

    df = db.query("SELECT predict('resnet', img) as prob FROM tbl").to_df()
    actual_prob = torch.tensor(df["prob"].iloc[0])
    actual_class = torch.argmax(actual_prob)

    probabilities = _get_expected(resnet, device, cat_path)
    assert torch.equal(actual_class, torch.argmax(probabilities))
    if device != 'cuda':
        assert torch.allclose(actual_prob, probabilities)

    argmax = (db.query("SELECT list_argmax(predict('resnet', img)) as pred FROM tbl")
              .to_df().pred)
    assert (argmax.values == torch.argmax(probabilities).numpy()).all()


def _get_expected(resnet, device, cat_path):
    resnet.to(device)
    resnet.eval()

    preprocess = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image: Image = Image.open(cat_path)
    image_tensor = preprocess(image)
    batch: torch.Tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = resnet(batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0).to('cpu')
    return probabilities


def _test_model(db, device, model, model_path):
    try:
        create_model(db, model_path, device)
        run_model(db, model, device)
    finally:
        db.execute("CALL drop_model('resnet')")
        assert db.query("SELECT * FROM ml_models()").to_df().size == 0


def test_resnet(db: DuckDBPyConnection, tmp_path: Path):
    device = 'cpu'
    model, model_path = download_model(tmp_path, device)
    _test_model(db, device, model, model_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_resnet_cuda(db: DuckDBPyConnection, tmp_path: Path):
    device = 'cuda'
    model, model_path = download_model(tmp_path, device)
    _test_model(db, device, model, model_path)

