#!/usr/bin/env python3
"""
Test script that reproduces the exact case from issue #4373.
"""

import lance
import pyarrow as pa
import sys

def test_original_reproduction():
    """Reproduce the exact case from issue #4373."""
    
    # Create a simple dataset as in the original issue
    df = pa.table({'id': [1, 2]})
    dataset = lance.write_dataset(df, "memory://")
    
    # Create the exact filter from the issue
    filter_conditions = []
    for i in range(100):  # As in the original issue
        filter_conditions.append(f"id = {i}")
    
    filter_expr = ' OR '.join(filter_conditions)
    print(f"Testing original reproduction case with {len(filter_conditions)} OR conditions...")
    
    try:
        # This should not cause a stack overflow
        result = dataset.to_table(filter=filter_expr)
        print("✅ Success! Original reproduction case works without stack overflow.")
        print(f"Result has {len(result)} rows")
        return True
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        return False

if __name__ == "__main__":
    print("Testing original reproduction case from issue #4373...")
    print("=" * 60)
    
    success = test_original_reproduction()
    
    if success:
        print("🎉 Original reproduction case passed! The fix is working.")
        sys.exit(0)
    else:
        print("💥 Original reproduction case failed.")
        sys.exit(1) 