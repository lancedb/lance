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
import logging
from geoarrow.pyarrow import point

# Enable Rust logging
os.environ['RUST_LOG'] = 'lance_index=debug'
logging.basicConfig(level=logging.DEBUG)


def main():
    print("🌍 Testing GeoArrow Point Geo Index in Lance")
    print("=" * 50)


    # Clean slate
    dataset_path = "/Users/jay.narale/work/Uber/geo_index_test"
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
        print(f"✅ Cleaned up existing dataset: {dataset_path}")


    # Step 1: Create GeoArrow Point data with enough points to test tree structure
    print("\n🔵 Step 1: Creating GeoArrow Point data (5000+ points)")
    
    # Generate random points across the US
    # US bounding box approximately: lng [-125, -65], lat [25, 50]
    np.random.seed(42)  # For reproducibility
    num_points = 5000
    
    lng_vals = np.random.uniform(-125, -65, num_points)
    lat_vals = np.random.uniform(25, 50, num_points)
    
    # Add some known cities at the beginning for testing
    known_cities = [
        {"id": 1, "city": "San Francisco", "lng": -122.4194, "lat": 37.7749, "population": 883305},
        {"id": 2, "city": "Los Angeles", "lng": -118.2437, "lat": 34.0522, "population": 3898747},
        {"id": 3, "city": "New York", "lng": -74.0060, "lat": 40.7128, "population": 8336817},
        {"id": 4, "city": "Chicago", "lng": -87.6298, "lat": 41.8781, "population": 2746388},
        {"id": 5, "city": "Houston", "lng": -95.3698, "lat": 29.7604, "population": 2304580},
    ]
    
    # Replace first 5 points with known cities
    for i, city in enumerate(known_cities):
        lng_vals[i] = city["lng"]
        lat_vals[i] = city["lat"]
    
    start_location = point().from_geobuffers(None, lng_vals, lat_vals)
    
    # Create IDs and city names
    ids = list(range(1, num_points + 1))
    cities = [known_cities[i]["city"] if i < len(known_cities) else f"Point_{i+1}" 
              for i in range(num_points)]
    populations = [known_cities[i]["population"] if i < len(known_cities) else np.random.randint(10000, 1000000)
                   for i in range(num_points)]
    
    table = pa.table({
        "id": ids,
        "city": cities,
        "start_location": start_location,
        "population": populations
    })


    print(f"✅ Created GeoArrow Point data with {num_points} points")
    print("📊 Table schema:")
    print(table.schema)
    print(f"📍 Point column type: {table.schema.field('start_location').type}")
    print(f"📍 Point column metadata: {table.schema.field('start_location').metadata}")
    print(f"📍 Known cities: {[c['city'] for c in known_cities]}")


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


        # Print first few results
        max_rows_to_print = min(10, len(table))
        for i in range(max_rows_to_print):
            row_data = {}
            for j, column in enumerate(table.columns):
                col_name = table.schema.field(j).name
                value = column.to_pylist()[i]
                row_data[col_name] = value
            print(f"📍 Row {i}: {row_data}")
        if len(table) > max_rows_to_print:
            print(f"... and {len(table) - max_rows_to_print} more rows")


        cities = table.column('city').to_pylist()
        print(f"\n✅ Found {len(cities)} results with broad bbox")
        print(f"📊 Known cities in results: {[c for c in cities if c in ['San Francisco', 'Los Angeles', 'New York', 'Chicago', 'Houston']]}")
        
        # With 5000 random points and a broad western US bbox, we should get hundreds/thousands of results
        assert len(cities) > 100, f"Expected many results (>100) from broad bbox, got {len(cities)}"
        assert 'San Francisco' in cities, "Expected San Francisco in results"
        assert 'Los Angeles' in cities, "Expected Los Angeles in results"
        print(f"✅ Verified SF and LA are in the {len(cities)} results")
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

        # Print first few results
        max_rows_to_print = min(10, len(tight_table))
        for i in range(max_rows_to_print):
            row_data = {}
            for j, column in enumerate(tight_table.columns):
                col_name = tight_table.schema.field(j).name
                value = column.to_pylist()[i]
                row_data[col_name] = value
            print(f"📍 Row {i}: {row_data}")
        if len(tight_table) > max_rows_to_print:
            print(f"... and {len(tight_table) - max_rows_to_print} more rows")

        cities = tight_table.column('city').to_pylist()
        print(f"\n✅ Found {len(cities)} results with tight bbox")
        known_cities_found = [c for c in cities if c in ['San Francisco', 'Los Angeles', 'New York', 'Chicago', 'Houston']]
        print(f"📊 Known cities in results: {known_cities_found}")
        
        # The tight bbox around SF should include SF, and might include some random points
        assert 'San Francisco' in cities, "Expected San Francisco in results"
        assert len(known_cities_found) == 1 and known_cities_found[0] == 'San Francisco', \
            f"Expected only San Francisco from known cities, got {known_cities_found}"
        print(f"✅ Verified only SF is in the known cities, total results: {len(cities)}")
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