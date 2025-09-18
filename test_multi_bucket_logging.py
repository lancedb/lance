#!/usr/bin/env python3
"""
Test script to demonstrate multi-bucket logging functionality.
"""

import os
import shutil
import pandas as pd
import lance

def test_multi_bucket_with_logging():
    """Test multi-bucket dataset creation with detailed logging."""
    
    print("🚀 Starting multi-bucket logging test...")
    
    # Use fixed test directory
    base_test_dir = "/Users/jay.narale/work/Uber/test"
    
    # Primary dataset location
    primary_uri = os.path.join(base_test_dir, "primary_bucket")
    
    # Additional data bucket locations  
    bucket2_uri = os.path.join(base_test_dir, "bucket2")
    bucket3_uri = os.path.join(base_test_dir, "bucket3")
    
    # Clear existing directories
    print("🧹 Clearing existing test directories...")
    for directory in [primary_uri, bucket2_uri, bucket3_uri]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"🧹 Removed: {directory}")
    
    # Create the bucket directories
    os.makedirs(primary_uri, exist_ok=True)
    os.makedirs(bucket2_uri, exist_ok=True) 
    os.makedirs(bucket3_uri, exist_ok=True)
    
    print(f"📁 Primary URI: {primary_uri}")
    print(f"📁 Bucket 2 URI: {bucket2_uri}")
    print(f"📁 Bucket 3 URI: {bucket3_uri}")
    
    # Create test data that will generate multiple fragments
    data = pd.DataFrame({
        'id': range(500),  # 500 rows
        'value': [f'value_{i}' for i in range(500)],
        'score': [i * 0.1 for i in range(500)]
    })
    
    print(f"📊 Created test data with {len(data)} rows")
    
    try:
        print("\n" + "="*50)
        print("🪣 CREATING MULTI-BUCKET DATASET...")
        print("="*50)
        
        # Create dataset with multi-bucket layout
        dataset = lance.write_dataset(
            data, 
            primary_uri,
            data_bucket_uris=[bucket2_uri, bucket3_uri],
            max_rows_per_file=100  # Force multiple fragments (5 fragments total)
        )
        
        print("\n" + "="*50)
        print("✅ DATASET CREATION COMPLETE!")
        print("="*50)
        
        print(f"Dataset URI: {dataset.uri}")
        print(f"Dataset version: {dataset.version}")
        print(f"Schema: {dataset.schema}")
        
        # Verify we can read the data back
        result = dataset.to_table().to_pandas()
        print(f"Successfully read back {len(result)} rows")
        
        # Check file distribution across buckets
        print(f"\n📁 File distribution:")
        for bucket_name, bucket_uri in [
            ("Primary", primary_uri),
            ("Bucket 2", bucket2_uri), 
            ("Bucket 3", bucket3_uri)
        ]:
            # Check both the bucket root and data subdirectory
            data_dir = os.path.join(bucket_uri, "data")
            if os.path.exists(data_dir):
                files = [f for f in os.listdir(data_dir) if f.endswith('.lance')]
                print(f"  {bucket_name} (data/): {len(files)} files")
                for file in files:
                    print(f"    - {file}")
            elif os.path.exists(bucket_uri):
                files = [f for f in os.listdir(bucket_uri) if f.endswith('.lance')]
                print(f"  {bucket_name}: {len(files)} files")
                for file in files:
                    print(f"    - {file}")
            else:
                print(f"  {bucket_name}: directory not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_bucket_logging():
    """Test single-bucket mode for comparison."""
    
    print("\n" + "="*50)
    print("🪣 TESTING SINGLE-BUCKET MODE...")  
    print("="*50)
    
    base_test_dir = "/Users/jay.narale/work/Uber/test"
    dataset_uri = os.path.join(base_test_dir, "single_bucket")
    
    # Clear existing directory
    if os.path.exists(dataset_uri):
        shutil.rmtree(dataset_uri)
        print(f"🧹 Removed: {dataset_uri}")
    
    os.makedirs(dataset_uri, exist_ok=True)
    
    # Create smaller test data
    data = pd.DataFrame({
        'id': range(200),
        'value': [f'value_{i}' for i in range(200)]
    })
    
    try:
        dataset = lance.write_dataset(
            data, 
            dataset_uri,
            max_rows_per_file=100  # 2 fragments
        )
        
        print("✅ Single-bucket dataset created successfully!")
        print(f"Records: {len(dataset.to_table().to_pandas())}")
        return True
        
    except Exception as e:
        print(f"❌ Single-bucket error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Multi-Bucket Logging Test")
    print("=" * 60)
    
    success = True
    
    # Test multi-bucket with logging
    success &= test_multi_bucket_with_logging()
    
    # # Test single-bucket for comparison
    success &= test_single_bucket_logging()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All logging tests completed!")
    else:
        print("💥 Some tests failed!")
        exit(1)
