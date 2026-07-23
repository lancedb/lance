# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Shared helpers for the data-overlay benchmark suite.

Data overlay files supply replacement values for a subset of (row offset, field)
cells in a fragment, merged on read, without rewriting the base data file. These
helpers build synthetic base datasets, commit overlay layers through the public
``lance.LanceOperation.DataOverlay`` operation, and measure their cost.
"""

import os

# Data overlay support is gated off in release builds unless this is set (it is
# always on in debug builds). Benchmarks usually run against a release build, so
# enable it here, before lance is imported, or reading overlay datasets fails.
os.environ.setdefault("LANCE_ENABLE_UNSTABLE_DATA_OVERLAY_FILES", "1")

from typing import List, Optional, Tuple  # noqa: E402
from urllib.parse import urlparse  # noqa: E402

import lance  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
from lance.file import LanceFileWriter  # noqa: E402

# Default width for the ``embedding`` dtype. Individual benchmarks override it
# via ``embedding_dim`` to model narrow vs. wide (e.g. 3072-d) value columns.
EMBEDDING_DIM = 128


def _value_type(dtype: str, embedding_dim: int = EMBEDDING_DIM) -> pa.DataType:
    if dtype == "int32":
        return pa.int32()
    if dtype == "embedding":
        return pa.list_(pa.float32(), embedding_dim)
    raise ValueError(f"unknown overlay benchmark dtype {dtype!r}")


def _gen_values(
    dtype: str,
    n: int,
    rng: np.random.Generator,
    embedding_dim: int = EMBEDDING_DIM,
) -> pa.Array:
    if dtype == "int32":
        return pa.array(rng.integers(0, 1 << 30, size=n, dtype=np.int32))
    if dtype == "embedding":
        flat = rng.random(n * embedding_dim, dtype=np.float32)
        return pa.FixedSizeListArray.from_arrays(pa.array(flat), embedding_dim)
    raise ValueError(f"unknown overlay benchmark dtype {dtype!r}")


def make_base_dataset(
    base_path: str,
    num_rows: int,
    rows_per_file: int,
    dtype: str,
    version: str,
    payload_dim: int = 0,
    embedding_dim: int = EMBEDDING_DIM,
) -> lance.LanceDataset:
    """Create a base dataset with an ``id`` key and a ``val`` payload column.

    ``val`` is the column overlays target. ``rows_per_file`` controls the number
    of fragments (``num_rows // rows_per_file``). ``embedding_dim`` sets the width
    of the ``val`` column when ``dtype`` is ``embedding`` (e.g. 3072 for a wide
    embedding). When ``payload_dim`` > 0 an extra ``payload`` fixed-size-list
    float32 column of that width is added, making rows wider so the cost of
    rewriting whole rows (update / merge_insert) can be compared against
    overlaying just ``val``.
    """
    vtype = _value_type(dtype, embedding_dim)
    fields = {"id": pa.int64(), "val": vtype}
    if payload_dim:
        fields["payload"] = pa.list_(pa.float32(), payload_dim)
    schema = pa.schema(fields)
    rng = np.random.default_rng(0)

    def batches():
        written = 0
        while written < num_rows:
            n = min(rows_per_file, num_rows - written)
            ids = pa.array(range(written, written + n), pa.int64())
            cols = [ids, _gen_values(dtype, n, rng, embedding_dim)]
            if payload_dim:
                flat = rng.random(n * payload_dim, dtype=np.float32)
                cols.append(
                    pa.FixedSizeListArray.from_arrays(pa.array(flat), payload_dim)
                )
            yield pa.record_batch(cols, schema=schema)
            written += n

    reader = pa.RecordBatchReader.from_batches(schema, batches())
    return lance.write_dataset(
        reader,
        base_path,
        schema=schema,
        max_rows_per_file=rows_per_file,
        data_storage_version=version,
    )


def coverage_offsets(num_rows: int, fraction: float, pattern: str) -> List[int]:
    """Offsets within a fragment covered by an overlay.

    ``contiguous`` packs the covered cells into a single leading run (few pages
    touched); ``stride`` spreads them evenly across the fragment (many pages
    touched). Both cover ``round(num_rows * fraction)`` cells.
    """
    count = max(1, int(round(num_rows * fraction)))
    if pattern == "contiguous":
        return list(range(count))
    if pattern == "stride":
        step = max(1, num_rows // count)
        return list(range(0, num_rows, step))[:count]
    raise ValueError(f"unknown coverage pattern {pattern!r}")


def _val_field_id(ds: lance.LanceDataset) -> int:
    base_df = ds.get_fragments()[0].metadata.files[0]
    names = [f.name for f in ds.schema]
    return base_df.fields[names.index("val")]


def commit_overlay_layers(
    ds: lance.LanceDataset,
    num_layers: int,
    fraction: float,
    pattern: str,
    dtype: str,
    *,
    seed: int = 0,
    embedding_dim: int = EMBEDDING_DIM,
) -> lance.LanceDataset:
    """Commit ``num_layers`` overlays on ``val``, each covering the same offsets
    in every fragment so that all layers must be consulted on read (the case
    that motivates compaction). Returns the updated dataset.
    """
    base_df = ds.get_fragments()[0].metadata.files[0]
    field_id = _val_field_id(ds)
    data_dir = os.path.join(_local_path(ds), "data")
    for layer in range(num_layers):
        rng = np.random.default_rng(seed + layer + 1)
        groups = []
        for frag in ds.get_fragments():
            offsets = coverage_offsets(frag.count_rows(), fraction, pattern)
            values = _gen_values(dtype, len(offsets), rng, embedding_dim)
            batch = pa.record_batch([values], names=["val"])
            name = f"overlay_l{layer}_f{frag.fragment_id}.lance"
            path = os.path.join(data_dir, name)
            with LanceFileWriter(path) as writer:
                writer.write_batch(batch)
            df = lance.fragment.DataFile(
                path=name,
                fields=[field_id],
                column_indices=[0],
                file_major_version=base_df.file_major_version,
                file_minor_version=base_df.file_minor_version,
                file_size_bytes=os.path.getsize(path),
            )
            groups.append(
                lance.LanceOperation.DataOverlayGroup(
                    fragment_id=frag.fragment_id,
                    overlays=[
                        lance.LanceOperation.DataOverlayFile(
                            data_file=df, offsets=offsets
                        )
                    ],
                )
            )
        op = lance.LanceOperation.DataOverlay(groups=groups)
        ds = lance.LanceDataset.commit(ds, op, read_version=ds.version)
    return ds


# --- Measurement helpers ----------------------------------------------------


def _local_path(ds: lance.LanceDataset) -> str:
    parsed = urlparse(ds.uri)
    return parsed.path if parsed.scheme == "file" else ds.uri


def proc_io_counters() -> Optional[dict]:
    """Process-wide /proc/self/io counters, or None if unavailable.

    Captures IO regardless of which object store issued it, which is what makes
    it the right tool for comparing mechanisms whose IO does not all flow through
    a single dataset handle. Use ``rchar``/``wchar`` (bytes moved by read/write
    syscalls) rather than ``read_bytes``/``write_bytes`` (physical storage IO):
    freshly written data sits in the page cache, so physical reads under-report
    what an operation actually read.
    """
    try:
        stats = {}
        with open("/proc/self/io") as f:
            for line in f:
                key, _, value = line.partition(":")
                stats[key.strip()] = int(value)
        return stats
    except OSError:
        return None


def file_sizes(path: str) -> dict:
    """Map of relative-path -> size in bytes for every file under ``path``."""
    out = {}
    for root, _, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            if os.path.isfile(fp):
                out[os.path.relpath(fp, path)] = os.path.getsize(fp)
    return out


def written_breakdown(before: dict, after: dict) -> Tuple[int, int]:
    """Split newly written/grown bytes (``after`` vs ``before`` snapshots) into
    (data_bytes, metadata_bytes). Data is everything under ``data/``; metadata is
    manifests, transactions, and deletion files."""
    data = meta = 0
    for path, size in after.items():
        delta = size - before.get(path, 0)
        if delta <= 0:
            continue
        if path.startswith("data" + os.sep):
            data += delta
        else:
            meta += delta
    return data, meta


def manifest_size(ds: lance.LanceDataset) -> int:
    """Size in bytes of the manifest for the dataset's current version."""
    # Manifests are named `{u64::MAX - version}.manifest` so that a plain
    # lexicographic directory listing yields newest-version-first.
    name = f"{(1 << 64) - 1 - ds.version}.manifest"
    return os.path.getsize(os.path.join(_local_path(ds), "_versions", name))
