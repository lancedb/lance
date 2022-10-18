#!/usr/bin/env python3

from pathlib import Path

import pandas as pd
import torch
import pyarrow as pa

from PIL import Image
from duckdb import DuckDBPyConnection
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


def test_resnet(db: DuckDBPyConnection, tmp_path: Path):
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    m = torch.jit.script(resnet)
    model_path = tmp_path / "resnet.pth"
    torch.jit.save(m, str(model_path))

    db.execute(f"CALL create_pytorch_model('resnet', '{str(model_path)}');")

    expected_models = pd.DataFrame([{
        "name": "resnet",
        "uri": str(model_path),
        "type": "torchscript",
    }])
    pd.testing.assert_frame_equal(db.query("SELECT * FROM ml_models()").to_df(), expected_models)

    cat_path = Path(__file__).parent / "testdata" / "cat.jpg"
    cat = cat_path.read_bytes()
    tbl = pa.Table.from_pylist([{"img": cat}])

    df = db.query("SELECT predict('resnet', img) as prob FROM tbl").to_df()
    actual_prob = torch.tensor(df["prob"].iloc[0])
    actual_class = torch.argmax(actual_prob)

    resnet.eval()

    preprocess = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image: Image = Image.open(cat_path)
    image_tensor = preprocess(image)
    batch: torch.Tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        output = resnet(batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    assert torch.equal(actual_class, torch.argmax(probabilities))
    assert torch.allclose(actual_prob, probabilities)

    argmax = (db.query("SELECT list_argmax(predict('resnet', img)) as pred FROM tbl")
              .to_df().pred)
    assert (argmax.values == torch.argmax(probabilities).numpy()).all()

    db.execute("CALL drop_model('resnet')")
    assert db.query("SELECT * FROM ml_models()").to_df().size == 0
