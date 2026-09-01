# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
from typing import TYPE_CHECKING, Any
from urllib.request import urlopen

import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from pathlib import Path

    import psutil

DEFAULT_DATASET_URI = "tos://test-lance/mj-lance/LAION_100M_lance_2.1.lance"
DEFAULT_VECTOR_COLUMN = "emb"
DEFAULT_ID_COLUMN = "id"
DEFAULT_BASELINE_BRANCH = "ivf-rq-reuse-off"
DEFAULT_OPTIMIZED_BRANCH = "ivf-rq-reuse-on"
DEFAULT_DIMENSION = 768
DEFAULT_TARGET_PARTITION_SIZE = 4096
DEFAULT_NUM_PARTITIONS = 100_000_000 // DEFAULT_TARGET_PARTITION_SIZE
DEFAULT_SEGMENT_ROWS = DEFAULT_TARGET_PARTITION_SIZE**2


class ResourceSampler:
    def __init__(self, process: psutil.Process, interval_seconds: float = 0.05):
        self.process = process
        self.interval_seconds = interval_seconds
        self.peak_rss = process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join()
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        return self.peak_rss


def process_resource_snapshot(process: psutil.Process) -> tuple[float, int]:
    cpu = process.cpu_times()
    return cpu.user + cpu.system, process.memory_info().rss


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean, got {value!r}")


def tos_storage_options() -> dict[str, str]:
    """Build TOS storage options without logging credential values."""
    access_key = os.environ.get("TOS_ACCESS_KEY_ID") or os.environ.get("EMR_QA_AK")
    secret_key = os.environ.get("TOS_SECRET_ACCESS_KEY")
    if secret_key is None:
        secret_key = os.environ.get("EMR_QA_SK")
        if secret_key and parse_bool(os.environ.get("EMR_QA_SK_BASE64", "true")):
            try:
                secret_key = base64.b64decode(secret_key, validate=True).decode("utf-8")
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError(
                    "EMR_QA_SK is not valid base64-encoded UTF-8"
                ) from error

    endpoint = os.environ.get("TOS_ENDPOINT")
    missing = [
        name
        for name, value in (
            ("TOS_ACCESS_KEY_ID or EMR_QA_AK", access_key),
            ("TOS_SECRET_ACCESS_KEY or EMR_QA_SK", secret_key),
            ("TOS_ENDPOINT", endpoint),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Missing TOS configuration: {', '.join(missing)}")

    options = {
        "tos_access_key_id": access_key,
        "tos_secret_access_key": secret_key,
        "tos_endpoint": endpoint,
    }
    if region := os.environ.get("TOS_REGION"):
        options["tos_region"] = region
    if token := os.environ.get("TOS_SECURITY_TOKEN"):
        options["tos_security_token"] = token
    return options


def storage_options_for(uri: str) -> dict[str, str] | None:
    return tos_storage_options() if uri.startswith("tos://") else None


def open_branch(uri: str, branch: str) -> lance.LanceDataset:
    options = storage_options_for(uri)
    base = lance.dataset(uri, storage_options=options)
    if branch not in base.branches.list():
        raise ValueError(
            f"Lance branch {branch!r} does not exist. "
            "Create it before running this tool."
        )
    return base.checkout_version((branch, None))


def find_index(dataset: lance.LanceDataset, name: str):
    matches = [index for index in dataset.describe_indices() if index.name == name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one index named {name!r}, found {len(matches)}"
        )
    return matches[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True, default=str) + "\n")


def _read_parquet(uri: str) -> pa.Table:
    if uri.startswith(("http://", "https://")):
        with urlopen(uri) as response:  # noqa: S310 - benchmark input is explicit
            return pq.read_table(pa.BufferReader(response.read()))
    if "://" not in uri or uri.startswith("file://"):
        return pq.read_table(uri.removeprefix("file://"))
    raise ValueError(
        f"Parquet URI {uri!r} is not directly readable. Stage this small file "
        "locally, expose it over HTTP, or convert it to a Lance dataset."
    )


def read_table(uri: str, columns: list[str]) -> pa.Table:
    if uri.endswith(".parquet"):
        table = _read_parquet(uri)
        return table.select(columns)
    return lance.dataset(uri, storage_options=storage_options_for(uri)).to_table(
        columns=columns
    )


def load_evaluation_data(
    query_uri: str,
    ground_truth_uri: str,
    *,
    query_id_column: str = "id",
    query_vector_column: str = "emb",
    ground_truth_id_column: str = "id",
    neighbors_column: str = "neighbors_id",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    query_table = read_table(query_uri, [query_id_column, query_vector_column])
    truth_table = read_table(
        ground_truth_uri, [ground_truth_id_column, neighbors_column]
    )
    truth_by_id = {
        int(query_id): np.asarray(neighbors, dtype=np.int64)
        for query_id, neighbors in zip(
            truth_table[ground_truth_id_column].to_pylist(),
            truth_table[neighbors_column].to_pylist(),
        )
    }

    queries: list[np.ndarray] = []
    ground_truth: list[np.ndarray] = []
    for query_id, vector in zip(
        query_table[query_id_column].to_pylist(),
        query_table[query_vector_column].to_pylist(),
    ):
        normalized_id = int(query_id)
        if normalized_id not in truth_by_id:
            raise ValueError(f"Ground truth is missing query id {normalized_id}")
        queries.append(np.asarray(vector, dtype=np.float32))
        ground_truth.append(truth_by_id[normalized_id])
    if not queries:
        raise ValueError("The evaluation query dataset is empty")
    return queries, ground_truth
