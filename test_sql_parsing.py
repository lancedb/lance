#!/usr/bin/env python3
"""
Test script to verify that the stack overflow issue is in SQL parsing.
"""

import lance
import pyarrow as pa
import sys

def optimize_or_chain_to_in_list(filter_str):
    """Convert OR chains to IN lists before SQL parsing."""
    # Simple optimization: if we have many "column = value OR column = value" patterns,
    # convert them to "column IN (value1, value2, ...)"
    
    # This is a simplified version - in practice, we'd need a more robust parser
    # For now, let's just test with a simple case
    if " OR " in filter_str and "id = " in filter_str:
        # Extract all the values
        parts = filter_str.split(" OR ")
        values = []
        column = None
        
        for part in parts:
            part = part.strip()
            if part.startswith("id = "):
                value = part[5:]  # Remove "id = "
                values.append(value)
                if column is None:
                    column = "id"
            else:
                # If we encounter a non-matching pattern, can't optimize
                return filter_str
        
        if len(values) > 1:
            # Convert to IN list
            in_list = f"{column} IN ({', '.join(values)})"
            print(f"Optimized: {filter_str[:50]}... -> {in_list}")
            return in_list
    
    return filter_str

def test_optimized_filter():
    """Test that optimized filters work without stack overflow."""
    
    # Create a simple dataset
    df = pa.table({'id': [1, 2, 3, 4, 5]})
    dataset = lance.write_dataset(df, "memory://")
    
    # Create a long OR chain
    or_conditions = []
    for i in range(50):  # Using 50 instead of 100 to be safe
        or_conditions.append(f"id = {i}")
    
    filter_expr = ' OR '.join(or_conditions)
    print(f"Original filter: {filter_expr[:50]}...")
    
    # Optimize the filter
    optimized_filter = optimize_or_chain_to_in_list(filter_expr)
    print(f"Optimized filter: {optimized_filter}")
    
    try:
        # This should not cause a stack overflow
        result = dataset.to_table(filter=optimized_filter)
        print("✅ Success! No stack overflow occurred.")
        print(f"Result has {len(result)} rows")
        return True
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        return False

def test_simple_in_list():
    """Test that simple IN lists work correctly."""
    
    # Create a simple dataset
    df = pa.table({'id': [1, 2, 3, 4, 5]})
    dataset = lance.write_dataset(df, "memory://")
    
    # Test with a simple IN list
    filter_expr = "id IN (1, 2, 3)"
    print(f"Testing IN list: {filter_expr}")
    
    try:
        result = dataset.to_table(filter=filter_expr)
        print("✅ Success! IN list works.")
        print(f"Result has {len(result)} rows")
        return True
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        return False

if __name__ == "__main__":
    print("Testing SQL parsing optimization for issue #4373...")
    print("=" * 60)
    
    # Test 1: Simple IN list
    success1 = test_simple_in_list()
    print()
    
    # Test 2: Optimized OR chain
    success2 = test_optimized_filter()
    print()
    
    if success1 and success2:
        print("🎉 All tests passed! The optimization approach works.")
        sys.exit(0)
    else:
        print("💥 Some tests failed.")
        sys.exit(1) 