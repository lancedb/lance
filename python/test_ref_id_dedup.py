#!/usr/bin/env python3
"""Verify ref_id dedup works on all three blob storage paths.

For each size class (Inline, Packed, Dedicated), writes 20 rows that all
share the same ref_id. The physical storage should hold 1 copy, not 20.
"""
import os
import shutil
from pathlib import Path

import lance
import pyarrow as pa
from lance.blob import Blob, blob_array, blob_field

WORK = Path("./ref_id_dedup_test")
N_ROWS = 20


def make_batch(n_rows: int, payload: bytes, ref_id: int) -> pa.RecordBatch:
    images = blob_array([Blob(data=payload, ref_id=ref_id) for _ in range(n_rows)])
    schema = pa.schema([
        pa.field("row_id", pa.int64()),
        blob_field("blob"),
    ])
    return pa.RecordBatch.from_arrays(
        [pa.array(range(n_rows), type=pa.int64()), images],
        schema=schema,
    )


def dataset_size_bytes(ds_path: Path) -> tuple[int, int, int]:
    """Return (main_lance_bytes, sidecar_bytes, sidecar_count)."""
    main_bytes = 0
    sidecar_bytes = 0
    sidecar_count = 0
    for p in ds_path.rglob("*"):
        if p.is_file():
            if p.suffix == ".blob":
                sidecar_bytes += p.stat().st_size
                sidecar_count += 1
            elif p.suffix == ".lance":
                main_bytes += p.stat().st_size
    return main_bytes, sidecar_bytes, sidecar_count


def run_case(label: str, payload_size: int, ref_id: int) -> None:
    case_dir = WORK / label
    case_dir.mkdir(parents=True, exist_ok=True)
    ds_path = case_dir / "ds.lance"

    payload = os.urandom(payload_size)
    batch = make_batch(N_ROWS, payload, ref_id)

    lance.write_dataset(batch, str(ds_path), mode="create", data_storage_version="2.2")

    main_b, side_b, side_n = dataset_size_bytes(ds_path)
    ideal = payload_size

    print(f"\n=== {label} (payload={payload_size:,}B, ref_id={ref_id}, rows={N_ROWS}) ===")
    print(f"  main .lance:     {main_b:>13,} B")
    print(f"  sidecar total:   {side_b:>13,} B  ({side_n} files)")
    print(f"  ideal (1 copy):  {ideal:>13,} B")
    print(f"  naive (20 copies): {ideal * N_ROWS:>11,} B")

    # Read back to verify correctness
    ds = lance.dataset(str(ds_path))
    blobs = ds.take_blobs("blob", indices=list(range(N_ROWS)))
    read = [bytes(b.read()) for b in blobs]
    ok = all(b == payload for b in read)
    unique = len({hash(b) for b in read})
    print(f"  read-back OK:    {ok} ({unique} unique contents out of {N_ROWS} rows)")

    # Dedup verdict
    total_storage = main_b + side_b
    amp = total_storage / ideal if ideal else 0
    dedup_works = amp < 3.0  # allow some overhead for descriptors, headers, etc
    verdict = "✓ DEDUP" if dedup_works else "✗ DUPLICATED"
    print(f"  amplification:   {amp:.2f}x   [{verdict}]")


def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    # Inline path: 32 KB (< 64 KB)
    run_case("inline_32kb", 32 * 1024, ref_id=101)

    # Packed path: 1 MB (> 64 KB, < 4 MB)
    run_case("packed_1mb", 1 * 1024 * 1024, ref_id=102)

    # Dedicated path: 6 MB (> 4 MB)
    run_case("dedicated_6mb", 6 * 1024 * 1024, ref_id=103)


if __name__ == "__main__":
    main()
