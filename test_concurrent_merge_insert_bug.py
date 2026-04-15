#!/usr/bin/env python3
"""
Regression test for lancedb/lance#4585: concurrent merge_insert duplicate rows.

This script demonstrates that concurrent merge_insert("id") operations
without unenforced-primary-key schema metadata silently create duplicate
rows in stock lancedb releases (tested against PyPI lancedb).

Usage:
    # Test against stock PyPI lancedb (should show the bug):
    uv run --with lancedb test_concurrent_merge_insert_bug.py

    # Test against a specific version:
    uv run --with 'lancedb==0.20.0' test_concurrent_merge_insert_bug.py

    # Test against a local build with the fix:
    uv run --with ./python test_concurrent_merge_insert_bug.py
"""

import concurrent.futures
import shutil
import sys
import tempfile

import lancedb
import pyarrow as pa


def test_concurrent_merge_insert_creates_duplicates():
    """
    5 concurrent workers each merge_insert 20 rows with overlapping ids.
    Expected: exactly 20 unique rows (ids 0..19) + 1 seed row.
    Bug: without the fix, all 5 workers succeed without conflict detection,
    producing up to 101 rows (5*20 + 1 seed).
    """
    tmpdir = tempfile.mkdtemp(prefix="lance_bug_4585_")
    try:
        db = lancedb.connect(tmpdir)

        # Create table with a seed row. Schema has NO primary key metadata.
        seed = pa.table({"id": [9999], "value": ["seed"]})
        tbl = db.create_table("test", seed)

        num_workers = 5
        rows_per_worker = 20

        def worker(worker_id: int):
            data = pa.table(
                {
                    "id": list(range(rows_per_worker)),
                    "value": [f"w{worker_id}_r{i}" for i in range(rows_per_worker)],
                }
            )
            # merge_insert on "id" — no PK metadata on schema
            tbl.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(
                data
            )

        # Run all workers concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(worker, i) for i in range(num_workers)]
            for f in futures:
                f.result()  # propagate exceptions

        # Check results
        result = tbl.to_arrow()
        total_rows = len(result)
        unique_ids = set(result.column("id").to_pylist())
        expected_unique = rows_per_worker + 1  # +1 for seed row 9999

        return total_rows, len(unique_ids), expected_unique

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_concurrent_identical_writes():
    """
    8 workers all write the exact same 100 rows.
    Expected: exactly 100 rows.
    Bug: without the fix, produces up to 800 rows.
    """
    tmpdir = tempfile.mkdtemp(prefix="lance_bug_4585_identical_")
    try:
        db = lancedb.connect(tmpdir)
        seed = pa.table({"id": [9999], "value": ["seed"]})
        tbl = db.create_table("test", seed)

        num_workers = 8
        num_rows = 100

        shared_data = pa.table(
            {
                "id": list(range(num_rows)),
                "value": [f"row_{i}" for i in range(num_rows)],
            }
        )

        def worker(_worker_id: int):
            tbl.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(
                shared_data
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(worker, i) for i in range(num_workers)]
            for f in futures:
                f.result()

        result = tbl.to_arrow()
        total_rows = len(result)
        unique_ids = set(result.column("id").to_pylist())
        expected_unique = num_rows + 1  # +1 for seed

        return total_rows, len(unique_ids), expected_unique

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    import lancedb as _lb

    print(f"lancedb version: {_lb.__version__}")
    print()

    all_passed = True

    # Test 1: overlapping ranges
    print("Test 1: 5 concurrent workers, overlapping key ranges (ids 0..19)")
    total, unique, expected = test_concurrent_merge_insert_creates_duplicates()
    duplicates = total - unique
    if total == expected and unique == expected:
        print(f"  PASS: {total} rows, {unique} unique ids (no duplicates)")
    else:
        print(f"  FAIL (BUG): {total} rows, {unique} unique ids, expected {expected}")
        print(f"  {duplicates} duplicate rows detected!")
        all_passed = False

    print()

    # Test 2: identical writes
    print("Test 2: 8 concurrent workers, identical data (100 rows)")
    total, unique, expected = test_concurrent_identical_writes()
    duplicates = total - unique
    if total == expected and unique == expected:
        print(f"  PASS: {total} rows, {unique} unique ids (no duplicates)")
    else:
        print(f"  FAIL (BUG): {total} rows, {unique} unique ids, expected {expected}")
        print(f"  {duplicates} duplicate rows detected!")
        all_passed = False

    print()
    if all_passed:
        print("All tests passed — no duplicate rows detected.")
        print("The fix for lancedb/lance#4585 is working correctly.")
    else:
        print("DUPLICATE ROWS DETECTED — this confirms the bug (lancedb/lance#4585).")
        print("The bloom filter is not being included in the transaction for")
        print("schemas without unenforced-primary-key metadata, so concurrent")
        print("merge_insert operations silently create duplicates.")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
