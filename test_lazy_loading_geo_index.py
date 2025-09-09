#!/usr/bin/env python3
"""
Test script for lazy loading functionality in Lance GeoArrow Point geo index.

This script tests lazy loading by:
1. Creating a large dataset (>1000 entries to force multiple leaf pages)
2. Creating a geo index with multiple leaf pages
3. Running queries that only touch some pages to verify lazy loading
4. Monitoring which pages are actually loaded vs available
"""

import numpy as np
import pyarrow as pa
import lance
import os
import shutil
from geoarrow.pyarrow import point
import random

def generate_large_geo_dataset(num_points=2500):
    """Generate a large dataset with points distributed across different regions"""
    print(f"🔢 Generating {num_points} geo points...")
    
    # Generate points in different geographic regions to ensure spatial distribution
    # Making sure our query regions will have matches!
    regions = [
        {"name": "West Coast", "lat_range": (32, 49), "lng_range": (-125, -115), "count": num_points // 5},
        {"name": "East Coast", "lat_range": (25, 47), "lng_range": (-80, -70), "count": num_points // 5},
        {"name": "Midwest", "lat_range": (35, 48), "lng_range": (-105, -85), "count": num_points // 5},
        {"name": "Southwest", "lat_range": (25, 40), "lng_range": (-115, -95), "count": num_points // 5},
        {"name": "California_Bay_Area", "lat_range": (37.6, 37.9), "lng_range": (-122.6, -122.2), "count": num_points // 5},
    ]
    
    all_lats = []
    all_lngs = []
    all_ids = []
    all_cities = []
    all_populations = []
    
    point_id = 1
    
    # First add some specific well-known cities to ensure predictable matches
    known_cities = [
        {"name": "San Francisco", "lat": 37.7749, "lng": -122.4194, "pop": 883305},
        {"name": "Los Angeles", "lat": 34.0522, "lng": -118.2437, "pop": 3898747},
        {"name": "New York", "lat": 40.7128, "lng": -74.0060, "pop": 8336817},
        {"name": "Boston", "lat": 42.3601, "lng": -71.0589, "pop": 685094},
        {"name": "Oakland", "lat": 37.8044, "lng": -122.2711, "pop": 440646},
        {"name": "San Jose", "lat": 37.3382, "lng": -121.8863, "pop": 1035317},
    ]
    
    for city in known_cities:
        all_lats.append(city["lat"])
        all_lngs.append(city["lng"])
        all_ids.append(point_id)
        all_cities.append(city["name"])
        all_populations.append(city["pop"])
        point_id += 1
    
    # Then add random points in each region
    for region in regions:
        for i in range(region["count"]):
            # Generate random point within region bounds
            lat = random.uniform(region["lat_range"][0], region["lat_range"][1])
            lng = random.uniform(region["lng_range"][0], region["lng_range"][1])
            
            all_lats.append(lat)
            all_lngs.append(lng)
            all_ids.append(point_id)
            all_cities.append(f"{region['name']}_City_{i+1}")
            all_populations.append(random.randint(10000, 5000000))
            
            point_id += 1
    
    # Fill remaining points if needed
    remaining = num_points - len(all_lats)
    for i in range(remaining):
        lat = random.uniform(20, 50)
        lng = random.uniform(-130, -65)
        
        all_lats.append(lat)
        all_lngs.append(lng)
        all_ids.append(point_id)
        all_cities.append(f"Extra_City_{i+1}")
        all_populations.append(random.randint(10000, 5000000))
        
        point_id += 1
    
    # Convert to numpy arrays
    lat_np = np.array(all_lats, dtype="float64")
    lng_np = np.array(all_lngs, dtype="float64")
    
    # Create GeoArrow points
    start_location = point().from_geobuffers(None, lng_np, lat_np)
    
    table = pa.table({
        "id": all_ids,
        "city": all_cities,
        "start_location": start_location,
        "population": all_populations
    })
    
    print(f"✅ Generated {len(table)} points across {len(regions)} regions")
    return table

def main():
    print("🌍 Testing Lazy Loading in GeoArrow Point Geo Index")
    print("=" * 60)
    
    # Clean slate
    dataset_path = "../../test_lazy_loading_geo"
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
        print(f"✅ Cleaned up existing dataset: {dataset_path}")
    
    # Step 1: Create large GeoArrow Point dataset
    print("\n🔵 Step 1: Creating large GeoArrow Point dataset")
    
    # Generate enough points to create multiple leaf pages (default is 1000 per page)
    num_points = 2500  # This should create ~3 leaf pages
    table = generate_large_geo_dataset(num_points)
    
    print("📊 Dataset schema:")
    print(table.schema)
    print(f"📍 Point column type: {table.schema.field('start_location').type}")
    
    # Step 2: Write to Lance dataset
    print("\n🔵 Step 2: Writing large dataset to Lance")
    try:
        geo_ds = lance.write_dataset(table, dataset_path)
        print("✅ Successfully wrote large GeoArrow data to Lance dataset")
        
        # Verify data was written correctly
        loaded_table = geo_ds.to_table()
        print(f"📊 Dataset has {len(loaded_table)} rows")
        
    except Exception as e:
        print(f"❌ Failed to write dataset: {e}")
        return
    
    # Step 3: Create geo index (should create multiple leaf pages)
    print("\n🔵 Step 3: Creating geo index (expecting multiple leaf pages)")
    try:
        geo_ds.create_scalar_index(column="start_location", index_type="RTREE")
        print("✅ Successfully created geo index")
        
        # Check what indexes exist
        indexes = geo_ds.list_indices()
        print("📊 Available indexes:")
        for idx in indexes:
            print(f"  - {idx}")
            
    except Exception as e:
        print(f"❌ Failed to create geo index: {e}")
        return
    
    # Step 4: Test queries that should only touch specific pages (lazy loading)
    print("\n🔵 Step 4: Testing lazy loading with targeted queries")
    
    # Query 1: West Coast region (should only load relevant pages)
    print("\n🔍 Query 1: West Coast region (lng: -125 to -115, lat: 32 to 49)")
    west_coast_sql = """
    SELECT id, city, population 
    FROM dataset 
    WHERE st_intersects(start_location, bbox(-125, 32, -115, 49))
    """
    
    print("📋 Running West Coast query...")
    query1 = geo_ds.sql(west_coast_sql).build()
    result1 = query1.to_batch_records()
    
    if result1:
        table1 = pa.Table.from_batches(result1)
        print(f"✅ West Coast query returned {len(table1)} results")
        
        # Show first few results
        for i in range(min(3, len(table1))):
            row_data = {}
            for j, column in enumerate(table1.columns):
                col_name = table1.schema.field(j).name
                value = column.to_pylist()[i]
                row_data[col_name] = value
            print(f"📍 Sample {i+1}: {row_data}")
    
    # Query 2: East Coast region (should load different pages)
    print("\n🔍 Query 2: East Coast region (lng: -80 to -70, lat: 25 to 47)")
    east_coast_sql = """
    SELECT id, city, population 
    FROM dataset 
    WHERE st_intersects(start_location, bbox(-80, 25, -70, 47))
    """
    
    print("📋 Running East Coast query...")
    query2 = geo_ds.sql(east_coast_sql).build()
    result2 = query2.to_batch_records()
    
    if result2:
        table2 = pa.Table.from_batches(result2)
        print(f"✅ East Coast query returned {len(table2)} results")
        
        # Show first few results
        for i in range(min(3, len(table2))):
            row_data = {}
            for j, column in enumerate(table2.columns):
                col_name = table2.schema.field(j).name
                value = column.to_pylist()[i]
                row_data[col_name] = value
            print(f"📍 Sample {i+1}: {row_data}")
    
    # Query 3: Small region that should have matches in Bay Area
    print("\n🔍 Query 3: Bay Area region (lng: -122.6 to -122.2, lat: 37.6 to 37.9)")
    small_region_sql = """
    SELECT id, city, population 
    FROM dataset 
    WHERE st_intersects(start_location, bbox(-122.6, 37.6, -122.2, 37.9))
    """
    
    print("📋 Running small region query...")
    query3 = geo_ds.sql(small_region_sql).build()
    result3 = query3.to_batch_records()
    
    if result3:
        table3 = pa.Table.from_batches(result3)
        print(f"✅ Small region query returned {len(table3)} results")
        
        # Show all results for small query
        for i in range(len(table3)):
            row_data = {}
            for j, column in enumerate(table3.columns):
                col_name = table3.schema.field(j).name
                value = column.to_pylist()[i]
                row_data[col_name] = value
            print(f"📍 Result {i+1}: {row_data}")
    else:
        print("📍 No results in Bay Area region")
    
    # Query 4: Large query that should touch all pages
    print("\n🔍 Query 4: Continental US (should touch all pages)")
    continental_sql = """
    SELECT COUNT(*) as total_count
    FROM dataset 
    WHERE st_intersects(start_location, bbox(-130, 20, -65, 50))
    """
    
    print("📋 Running continental US query...")
    query4 = geo_ds.sql(continental_sql).build()
    result4 = query4.to_batch_records()
    
    if result4:
        table4 = pa.Table.from_batches(result4)
        total_count = table4.column('total_count').to_pylist()[0]
        print(f"✅ Continental US query returned {total_count} total points")
    
    # Step 5: Examine index structure
    print("\n🔵 Step 5: Examining index structure")
    try:
        import glob
        index_files = glob.glob(f"{dataset_path}/_indices/*")
        print(f"📂 Index directories: {len(index_files)}")
        
        for idx_dir in index_files:
            files = glob.glob(f"{idx_dir}/*")
            print(f"📂 Files in {idx_dir}:")
            leaf_page_count = 0
            for f in files:
                file_size = os.path.getsize(f)
                filename = os.path.basename(f)
                if filename.startswith("leaf_page_"):
                    leaf_page_count += 1
                print(f"  - {filename} ({file_size} bytes)")
            
            print(f"🍃 Total leaf pages created: {leaf_page_count}")
            
            if leaf_page_count > 1:
                print("✅ 🎉 MULTIPLE LEAF PAGES DETECTED! Lazy loading can be tested.")
            else:
                print("⚠️  Only one leaf page created. Consider increasing data size.")
                
    except Exception as e:
        print(f"❌ Failed to check index files: {e}")
    
    # Step 6: Run explain analyze to see query plans
    print("\n🔵 Step 6: Analyzing query execution plans")
    
    explain_sql = """
    EXPLAIN ANALYZE SELECT COUNT(*) 
    FROM dataset 
    WHERE st_intersects(start_location, bbox(-122.6, 37.6, -122.2, 37.9))
    """
    
    print("📋 Running EXPLAIN ANALYZE on Bay Area query...")
    explain_query = geo_ds.sql(explain_sql).build()
    explain_result = explain_query.to_batch_records()
    
    if explain_result:
        explain_table = pa.Table.from_batches(explain_result)
        print("🔍 EXPLAIN ANALYZE Result:")
        
        for i in range(len(explain_table)):
            for j, column in enumerate(explain_table.columns):
                col_name = explain_table.schema.field(j).name
                value = column.to_pylist()[i]
                print(f"📊 {col_name}: {value}")
    
    print("\n🎉 Lazy loading test completed!")
    print("=" * 60)
    print("🔍 Key observations to look for in logs:")
    print("  - Multiple leaf_page_*.lance files created")
    print("  - 'LAZY LOADING page X/Y' messages during queries")  
    print("  - 'Cache HIT' vs 'Cache MISS' for different queries")
    print("  - Different pages loaded for different geographic regions")

if __name__ == "__main__":
    main()
