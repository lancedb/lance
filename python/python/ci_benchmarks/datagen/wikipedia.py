# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

# Creates a Wikipedia dataset for Full Text Search (FTS) benchmarking.
#
# Downloads Wikipedia data from HuggingFace, creates a Lance dataset, and builds
# FTS indices to support various query types.

import re

import lance
import pyarrow as pa
from datasets import load_dataset
from lance.log import LOGGER

from ci_benchmarks.datasets import get_dataset_uri

# HuggingFace dataset configuration
HF_DATASET = "wikimedia/wikipedia"
HF_SUBSET = "20231101.en"
HF_SPLIT = "train"
NUM_ROWS = 100_000

SCHEMA = pa.schema(
    {
        "id": pa.string(),
        "text": pa.large_string(),
    }
)


def _download_and_process_wikipedia(batch_size: int = 5000):
    """Download Wikipedia data from HuggingFace and yield batches.

    Downloads the first NUM_ROWS from the wikimedia/wikipedia dataset
    and yields PyArrow RecordBatches.

    Args:
        batch_size: Number of rows per batch

    Yields:
        PyArrow RecordBatch
    """
    LOGGER.info(
        "Downloading Wikipedia dataset from HuggingFace: %s (subset: %s, split: %s)",
        HF_DATASET,
        HF_SUBSET,
        HF_SPLIT,
    )
    LOGGER.info("Will download first %s rows", f"{NUM_ROWS:,}")

    # Load dataset from HuggingFace with streaming to avoid loading all into memory
    LOGGER.info("Loading dataset in streaming mode...")
    dataset = load_dataset(
        HF_DATASET,
        HF_SUBSET,
        split=HF_SPLIT,
        streaming=True,
    )

    LOGGER.info("Dataset initialized, starting to download and process rows...")

    batch_data = {"id": [], "text": []}
    total_rows = 0

    for idx, row in enumerate(dataset):
        if total_rows >= NUM_ROWS:
            break

        # Extract fields
        # HuggingFace wikipedia dataset has: id, url, title, text
        row_id = row.get("url", f"row_{idx}")
        text = row.get("text", "")

        # Skip empty text
        if not text or text.strip() == "":
            continue

        # Transform text (lowercase and keep only letters)
        batch_data["id"].append(row_id)
        batch_data["text"].append(transform(text))

        # Yield batch when we reach batch_size
        if len(batch_data["id"]) >= batch_size:
            batch = pa.record_batch(
                [
                    pa.array(batch_data["id"], type=pa.string()),
                    pa.array(batch_data["text"], type=pa.large_string()),
                ],
                names=["id", "text"],
            )
            yield batch
            total_rows += len(batch_data["id"])
            progress_pct = (total_rows / NUM_ROWS) * 100
            LOGGER.info(
                "Processed %s / %s rows (%.1f%%)",
                f"{total_rows:,}",
                f"{NUM_ROWS:,}",
                progress_pct,
            )

            # Clear batch data
            batch_data = {"id": [], "text": []}

    # Yield remaining data
    if batch_data["id"]:
        batch = pa.record_batch(
            [
                pa.array(batch_data["id"], type=pa.string()),
                pa.array(batch_data["text"], type=pa.large_string()),
            ],
            names=["id", "text"],
        )
        yield batch
        total_rows += len(batch_data["id"])

    LOGGER.info("Finished processing %s total rows", f"{total_rows:,}")


PTN = re.compile("[^a-zA-Z]+")


def transform(text):
    return PTN.sub(" ", text.lower())


def _create_indices(ds: lance.LanceDataset):
    """Create FTS indices on the dataset.

    Creates indices to support different query types:
    1. Inverted index with position for phrase queries

    Args:
        ds: Lance dataset to create indices on
    """
    existing_indices = [idx.name for idx in ds.list_indices()]

    # Create inverted index with position support for phrase queries
    # This index supports both match and phrase queries
    if "text_fts_idx" not in existing_indices:
        LOGGER.info("Creating FTS index on 'text' column with position support")
        ds.create_scalar_index(
            "text",
            index_type="INVERTED",
            with_position=True,
            name="text_fts_idx",
        )
        LOGGER.info("FTS index 'text_fts_idx' created successfully")
    else:
        LOGGER.info("FTS index 'text_fts_idx' already exists")


def _create(dataset_uri: str):
    """Create Wikipedia dataset and indices (idempotent).

    Args:
        dataset_uri: URI where the dataset should be created
    """
    LOGGER.info("Checking if Wikipedia dataset exists at %s", dataset_uri)

    try:
        ds = lance.dataset(dataset_uri)
        row_count = ds.count_rows()
        LOGGER.info("Dataset exists with %s rows", f"{row_count:,}")

        # Check if indices exist
        existing_indices = [idx.name for idx in ds.list_indices()]
        if "text_fts_idx" in existing_indices:
            LOGGER.info("Dataset and indices already exist, skipping generation")
            return
        else:
            LOGGER.info("Dataset exists but indices are missing, creating indices...")
            _create_indices(ds)
            return

    except ValueError:
        # Dataset doesn't exist, create it
        LOGGER.info("Dataset does not exist, will create from HuggingFace source")

    # Download and create dataset
    LOGGER.info("Starting Wikipedia dataset creation at %s", dataset_uri)
    ds = lance.write_dataset(
        _download_and_process_wikipedia(),
        dataset_uri,
        schema=SCHEMA,
        mode="create",
        use_legacy_format=False,
    )

    row_count = ds.count_rows()
    LOGGER.info("Dataset created successfully with %s rows", f"{row_count:,}")

    # Create FTS indices
    LOGGER.info("Creating FTS indices...")
    _create_indices(ds)

    LOGGER.info("Wikipedia dataset generation complete!")


def gen_wikipedia():
    """Generate Wikipedia dataset for FTS benchmarks.

    This is the main entry point for dataset generation.
    Downloads the first 1 million rows from the wikimedia/wikipedia dataset
    (20231101.en subset) from HuggingFace, creates a Lance dataset,
    and builds FTS indices.
    """
    dataset_uri = get_dataset_uri("wikipedia")
    _create(dataset_uri)


if __name__ == "__main__":
    gen_wikipedia()
