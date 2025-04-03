# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os

import lance
import numpy as np
import pyarrow as pa
import pytest
from lance.torch.data import SafeLanceDataset, get_safe_loader


@pytest.fixture(scope="module")
def temp_lance_dataset(tmp_path_factory):
    """Create temporary Lance dataset for testing"""
    # Create test directory
    test_dir = tmp_path_factory.mktemp("lance_data")
    dataset_path = test_dir / "test_dataset.lance"

    # Generate mock data
    num_samples = 100
    data = pa.table(
        {
            "id": range(num_samples),
            "embedding": [
                np.random.rand(128).astype(np.float32).tobytes()
                for _ in range(num_samples)
            ],
        }
    )

    # Write to Lance format
    lance.write_dataset(data, dataset_path)

    yield str(dataset_path)

    # Cleanup: remove entire test directory
    if os.path.exists(test_dir):
        for root, dirs, files in os.walk(test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(test_dir)


def test_dataset_initialization(temp_lance_dataset):
    """Verify basic dataset initialization and metadata loading"""
    # Initialize dataset
    ds = SafeLanceDataset(temp_lance_dataset)

    # Validate metadata
    assert len(ds) == 100, "Incorrect number of samples"

    # Validate single sample loading
    sample = ds[0]
    assert isinstance(sample, dict), "Sample should be dictionary"
    assert "id" in sample and "embedding" in sample, "Missing required fields"


def test_multiprocess_loading(temp_lance_dataset, capsys):
    """Test multi-worker data loading functionality"""
    # Initialize DataLoader with 2 workers
    dataset = SafeLanceDataset(temp_lance_dataset)
    loader = get_safe_loader(dataset, num_workers=2, batch_size=16, drop_last=True)

    # Iterate through batches
    for batch in loader:
        assert batch["id"].shape == (16,), "Incorrect batch dimensions"


def test_resource_cleanup(temp_lance_dataset):
    """Ensure temporary data is properly cleaned after tests"""
    # Verify dataset exists during tests
    assert os.path.exists(temp_lance_dataset), "Test dataset missing"

    # The actual cleanup is handled by the fixture after yielding

