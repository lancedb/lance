#!/usr/bin/env python3
"""
Bug #5130 Reproduction: Zonemap Index Scanning Too Many Fragments

This script demonstrates that zonemap scalar indexes are not properly filtering
fragments at the query planning stage, resulting in scanning all fragments
instead of only the relevant ones.

Setup:
- 10 fragments with 100 rows each
- Each fragment contains a UNIQUE test_id value
- When querying for a specific test_id, only 1 fragment should be scanned

Expected: num_fragments=1 (scan only the relevant fragment)
Actual:   num_fragments=10 (scans ALL fragments - BUG!)

Usage:
    LANCE_LOG=debug python reproduce_bug_5130.py 2>&1 | grep "Executing plan"

The key metric is in the execution plan output:
    LanceRead: ... num_fragments=??? ...
      ScalarIndexQuery: query=[test_id = test_id_X]@test_id_idx

If num_fragments=10, the bug is present (should be 1).
"""

import lance
import pyarrow as pa

# Use scanner with stats callback to see actual fragment counts
scan_stats = None


def stats_callback(stats):
    global scan_stats
    scan_stats = stats


def main():
    # ==========================================================================
    # STEP 1: Create dataset with 10 fragments, each with unique test_id
    # ==========================================================================
    print("=" * 80)
    print("STEP 1: CREATING DATASET WITH 10 FRAGMENTS")
    print("=" * 80)

    fragments_data = []
    num_fragments = 10
    rows_per_fragment = 100

    for fragment_idx in range(num_fragments):
        # Each fragment gets a unique test_id
        test_id = f"test_id_{fragment_idx}"

        # Create rows for this fragment - all rows have the SAME test_id
        fragment_data = pa.table(
            {
                "id": range(
                    fragment_idx * rows_per_fragment,
                    (fragment_idx + 1) * rows_per_fragment,
                ),
                "test_id": [test_id] * rows_per_fragment,
                "value": [fragment_idx] * rows_per_fragment,
            }
        )
        fragments_data.append(fragment_data)

    # Combine all fragments into one table
    full_data = pa.concat_tables(fragments_data)

    print(f"Creating dataset with {len(full_data)} rows")
    print(f"Sample data:")
    print(full_data.slice(0, 5).to_pandas())
    print("...")
    print(full_data.slice(95, 10).to_pandas())  # Show boundary between fragments

    # Write dataset with max_rows_per_file to create multiple fragments
    ds = lance.write_dataset(
        full_data,
        "./zonemap_bug.lance",
        max_rows_per_file=rows_per_fragment,  # One fragment per batch
        max_rows_per_group=10,  # Smaller row groups for better zonemap granularity
        mode="overwrite",
    )

    # Verify fragment count
    fragments = ds.get_fragments()
    print(f"\n✓ Created {len(fragments)} fragments")

    # Show what test_id values are in each fragment
    # print("\nFragment distribution:")
    # for i, frag in enumerate(fragments):
    #     sample = frag.head(1).to_pydict()
    #     print(f"  Fragment {i}: test_id = {sample['test_id'][0]}")

    # ==========================================================================
    # STEP 2: Query WITHOUT index (baseline)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: QUERY WITHOUT INDEX (BASELINE)")
    print("=" * 80)

    scanner_no_index = ds.scanner(
        filter="test_id = 'test_id_5'", scan_stats_callback=stats_callback
    )
    result_no_index = scanner_no_index.to_table()

    print(f"✓ Query returned {len(result_no_index)} rows (expected: 100)")

    if scan_stats:
        print(scan_stats)
        fragments_scanned_no_index = scan_stats.all_counts.get(
            "fragments_scanned", "N/A"
        )
        print(f"\n📊 Scan Statistics (WITHOUT index):")
        print(f"  Fragments scanned: {fragments_scanned_no_index}")
        print(f"  Rows scanned: {scan_stats.all_counts.get('rows_scanned', 'N/A')}")
        print(f"  Bytes read: {scan_stats.bytes_read:,}")
    else:
        print("\n⚠️  Check logs for: num_fragments=10 (full table scan expected)")

    # ==========================================================================
    # STEP 3: Create zonemap index
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: CREATING ZONEMAP INDEX")
    print("=" * 80)

    ds.create_scalar_index("test_id", index_type="ZONEMAP")

    indices = ds.list_indices()
    print(f"✓ Created index: {[idx['name'] for idx in indices]}")

    # ==========================================================================
    # STEP 4: Query WITH zonemap index - THIS IS WHERE THE BUG MANIFESTS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: QUERY WITH ZONEMAP INDEX (BUG CHECK)")
    print("=" * 80)
    print("Querying for: test_id = 'test_id_5'")
    print("Expected: num_fragments=1 (should scan ONLY fragment 5)")
    print("-" * 80)

    scanner = ds.scanner(
        filter="test_id = 'test_id_5'", scan_stats_callback=stats_callback
    )
    result_with_index = scanner.to_table()

    print(f"\n✓ Query returned {len(result_with_index)} rows")

    if scan_stats:
        print(scan_stats)
        fragments_scanned = scan_stats.all_counts.get("fragments_scanned", "N/A")
        rows_scanned = scan_stats.all_counts.get("rows_scanned", "N/A")
        print(f"\n📊 Scan Statistics:")
        print(f"  Fragments scanned: {fragments_scanned}")
        print(f"  Rows scanned: {rows_scanned}")
        print(f"  Bytes read: {scan_stats.bytes_read:,}")
        print(f"  IOPs: {scan_stats.iops:,}")

        print("\n🐛 BUG #5130 CHECK:")
        if fragments_scanned != "N/A":
            if int(fragments_scanned) == 1:
                print(f"   ✅ Only 1 fragment scanned!")
            else:
                print(
                    f"   ⚠️  BUG PRESENT: {fragments_scanned} fragments scanned (expected 1)"
                )
                print(f"   This is a {int(fragments_scanned)}x overhead!")
    else:
        print("\n⚠️  Scan stats not available")
        print("\n🐛 BUG #5130 CHECK:")
        print("   Look in logs for the execution plan:")
        print("   ⚠️  If num_fragments=10, the BUG IS PRESENT")
        print("   ✓  If num_fragments=1, the bug is FIXED")

    # ==========================================================================
    # STEP 5: Verification
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: VERIFICATION")
    print("=" * 80)

    print("\n📋 Dataset Configuration:")
    print(f"  - Total fragments: {len(ds.get_fragments())}")
    print(f"  - Rows per fragment: {rows_per_fragment}")
    print(f"  - Each fragment has UNIQUE test_id values")

    print("\n🔍 Query: test_id = 'test_id_5'")
    print(f"  - Expected: Scan ONLY 1 fragment (fragment 5)")
    print(f"  - Actual result: {len(result_with_index)} rows ✓")

    # Verify the query correctness
    print("\n✅ Query Correctness:")
    sample_rows = result_with_index.to_pydict()
    unique_test_ids = set(sample_rows["test_id"])
    print(f"  - Unique test_ids in result: {unique_test_ids}")
    print(f"  - All rows have correct test_id: {unique_test_ids == {'test_id_5'}}")
    print(f"  - Returns correct count: {len(result_with_index) == 100}")

    # ==========================================================================
    # STEP 6: Test multiple queries
    # ==========================================================================
    # print("\n" + "=" * 80)
    # print("STEP 6: TESTING MULTIPLE QUERIES")
    # print("=" * 80)

    # test_queries = ["test_id_0", "test_id_3", "test_id_7", "test_id_9"]

    # print("Running queries for different test_ids...")
    # for test_id_query in test_queries:
    #     scanner = ds.scanner(
    #         filter=f"test_id = '{test_id_query}'", scan_stats_callback=stats_callback
    #     )
    #     scanner.scan_in_order = False
    #     result = scanner.to_table()

    #     fragments_scanned = (
    #         scan_stats.all_counts.get("fragments_scanned", "?") if scan_stats else "?"
    #     )
    #     status = "✅" if fragments_scanned == 1 else "⚠️ "
    #     print(
    #         f"  {status} Query '{test_id_query}': returned {len(result)} rows, scanned {fragments_scanned} fragments"
    #     )


if __name__ == "__main__":
    main()
