#!/usr/bin/env python3
"""
Test script for GeoArrow Point geo index functionality in Lance.

This script tests:
1. Creating GeoArrow Point data
2. Writing to Lance dataset
3. Creating a geo index on GeoArrow Point column
4. Querying with spatial filters
5. Verifying the geo index is used
"""


import numpy as np
import pyarrow as pa
import lance
import os
import shutil
from geoarrow.pyarrow import point


def main():
    print("🌍 Testing GeoArrow Point Geo Index in Lance")
    print("=" * 50)


    # Clean slate
    dataset_path = "/Users/jay.narale/work/Uber/geo_index_test"
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
        print(f"✅ Cleaned up existing dataset: {dataset_path}")


    # Step 1: Create GeoArrow Point data
    print("\n🔵 Step 1: Creating GeoArrow Point data")
    lat_np = np.array([37.7749, 34.0522, 40.7128], dtype="float64")  # SF, LA, NYC
    lng_np = np.array([-122.4194, -118.2437, -74.0060], dtype="float64")


    start_location = point().from_geobuffers(None, lng_np, lat_np)


    table = pa.table({
        "id": [1, 2, 3],
        "city": ["San Francisco", "Los Angeles", "New York"],
        "start_location": start_location,
        "population": [883305, 3898747, 8336817]
    })


    print("✅ Created GeoArrow Point data")
    print("📊 Table schema:")
    print(table.schema)
    print(f"📍 Point column type: {table.schema.field('start_location').type}")
    print(f"📍 Point column metadata: {table.schema.field('start_location').metadata}")


    # Step 2: Write to Lance dataset
    print("\n🔵 Step 2: Writing to Lance dataset")
    try:
        geo_ds = lance.write_dataset(table, dataset_path)
        print("✅ Successfully wrote GeoArrow data to Lance dataset")


        # Verify data was written correctly
        loaded_table = geo_ds.to_table()
        print(f"📊 Dataset has {len(loaded_table)} rows")
        print("📊 Dataset schema:")
        print(loaded_table.schema)


    except Exception as e:
        print(f"❌ Failed to write dataset: {e}")
        return


    # Step 3: Create geo index
    print("\n🔵 Step 3: Creating geo index on GeoArrow Point column")
    try:
        geo_ds.create_scalar_index(column="start_location", index_type="GEO")
        print("✅ Successfully created geo index")


        # Check what indexes exist
        indexes = geo_ds.list_indices()
        print("📊 Available indexes:")
        for idx in indexes:
            print(f"  - {idx}")


    except Exception as e:
        print(f"❌ Failed to create geo index: {e}")
        return


    # Step 4: Test st_intersects spatial query with broad bbox (both cities)
    print("\n🔵 Step 4: Testing st_intersects spatial query with broad bbox (both cities)")




    # First, run EXPLAIN ANALYZE to see the execution plan
    explain_sql = """
    EXPLAIN ANALYZE SELECT id, city, population 
    FROM dataset 
    WHERE st_intersects(start_location, bbox(-125, 30, -115, 45))
    """


    print("\n📋 Running EXPLAIN ANALYZE...")
    explain_query = geo_ds.sql(explain_sql).build()
    explain_result = explain_query.to_batch_records()


    if explain_result:
        explain_table = pa.Table.from_batches(explain_result)
        print("🔍 EXPLAIN ANALYZE Result:")
        print(f"Schema: {explain_table.schema}")
        print(f"Rows: {len(explain_table)}")


        # Print the execution plan
        for i in range(len(explain_table)):
            for j, column in enumerate(explain_table.columns):
                col_name = explain_table.schema.field(j).name
                value = column.to_pylist()[i]
                print(f"📊 {col_name}: {value}")


        # Check if geo index was used
        if len(explain_table) > 0:
            # Column 1 contains the actual plan, column 0 is just the plan type
            plan_text = str(explain_table.column(1).to_pylist()[0])
            if "ScalarIndexQuery" in plan_text or "start_location_idx" in plan_text:
                print("✅ 🌍 GEO INDEX WAS USED!")
                if "start_location_idx" in plan_text:
                    print("✅ 🌍 Found geo index reference: start_location_idx")
                if "ST_Intersects" in plan_text:
                    print("✅ 🌍 Spatial query detected: ST_Intersects")
                # Extract performance metrics
                import re
                if "output_rows=" in plan_text:
                    rows_match = re.search(r'output_rows=(\d+)', plan_text)
                    if rows_match:
                        print(f"✅ 🌍 Index returned {rows_match.group(1)} rows")
                if "search_time=" in plan_text:
                    time_match = re.search(r'search_time=([^,\]]+)', plan_text)
                    if time_match:
                        print(f"✅ 🌍 Index search time: {time_match.group(1)}")
            else:
                print("⚠️  Geo index was not detected in execution plan")
                print(f"📋 Full plan: {plan_text}")


    # Now run the actual query and get complete results
    print("\n📋 Running actual query...")
    actual_sql = """
    SELECT id, city, population 
    FROM dataset 
    WHERE st_intersects(start_location, bbox(-125, 30, -115, 45))
    """
    query = geo_ds.sql(actual_sql).build()
    result = query.to_batch_records()


    if result:
        table = pa.Table.from_batches(result)
        print("✅ Query Results:")
        print(f"📊 Schema: {table.schema}")
        print(f"📊 Number of rows: {len(table)}")


        # Print complete results
        for i in range(len(table)):
            row_data = {}
            for j, column in enumerate(table.columns):
                col_name = table.schema.field(j).name
                value = column.to_pylist()[i]
                row_data[col_name] = value
            print(f"📍 Row {i}: {row_data}")


        cities = table.column('city').to_pylist()
        print(f"\n✅ Found {len(cities)} cities with broad bbox: {cities}")
        assert len(cities) == 2, f"Expected 2 cities, got {len(cities)}"
        assert 'San Francisco' in cities, "Expected San Francisco in results"
        assert 'Los Angeles' in cities, "Expected Los Angeles in results"
    else:
        print("⚠️  No results returned")


    # Step 4b: Test with tight bbox (only San Francisco)
    print("\n🔵 Step 4b: Testing st_intersects with tight bbox (only San Francisco)")
    # SF is at (-122.4194, 37.7749), so use a tight box around it
    tight_sql = """
    SELECT id, city, population 
    FROM dataset 
    WHERE st_intersects(start_location, bbox(-123, 37, -122, 38))
    """
    tight_query = geo_ds.sql(tight_sql).build()
    tight_result = tight_query.to_batch_records()

    if tight_result:
        tight_table = pa.Table.from_batches(tight_result)
        print("✅ Query Results:")
        print(f"📊 Number of rows: {len(tight_table)}")

        for i in range(len(tight_table)):
            row_data = {}
            for j, column in enumerate(tight_table.columns):
                col_name = tight_table.schema.field(j).name
                value = column.to_pylist()[i]
                row_data[col_name] = value
            print(f"📍 Row {i}: {row_data}")

        cities = tight_table.column('city').to_pylist()
        print(f"\n✅ Found {len(cities)} city with tight bbox: {cities}")
        assert len(cities) == 1, f"Expected 1 city, got {len(cities)}"
        assert cities[0] == 'San Francisco', f"Expected San Francisco, got {cities[0]}"
    else:
        print("⚠️  No results returned")






    # Step 5: Check index files
    print("\n🔵 Step 5: Verifying index files")
    try:
        import glob
        index_files = glob.glob(f"{dataset_path}/_indices/*")
        print(f"📂 Index directories: {len(index_files)}")


        for idx_dir in index_files:
            files = glob.glob(f"{idx_dir}/*")
            print(f"📂 Files in {idx_dir}:")
            for f in files:
                file_size = os.path.getsize(f)
                print(f"  - {os.path.basename(f)} ({file_size} bytes)")


    except Exception as e:
        print(f"❌ Failed to check index files: {e}")


    print("\n🎉 Test completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()