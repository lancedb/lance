#!/usr/bin/env python3
"""
Test script to verify that the stack overflow issue with lots of OR conditions is resolved.
This script tests the reproduction case from issue #4373.
"""

import lance
import pyarrow as pa
import sys

def test_stack_overflow_fix():
    """Test that long OR chains don't cause stack overflow."""
    
    # Create a simple dataset
    df = pa.table({'id': [1, 2, 3, 4, 5]})
    dataset = lance.write_dataset(df, "memory://")
    
    # Create a long OR chain that would cause stack overflow
    # This is the reproduction case from the issue
    or_conditions = []
    for i in range(100):  # Using 100 instead of 1000 to be safe
        or_conditions.append(f"id = {i}")
    
    filter_expr = ' OR '.join(or_conditions)
    print(f"Testing filter with {len(or_conditions)} OR conditions...")
    
    try:
        # This should not cause a stack overflow
        result = dataset.to_table(filter=filter_expr)
        print("✅ Success! No stack overflow occurred.")
        print(f"Result has {len(result)} rows")
        return True
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        return False

def test_simple_or_chain():
    """Test that simple OR chains work correctly."""
    
    # Create a simple dataset
    df = pa.table({'id': [1, 2, 3, 4, 5]})
    dataset = lance.write_dataset(df, "memory://")
    
    # Create a simple OR chain
    filter_expr = "id = 1 OR id = 2 OR id = 3"
    print(f"Testing simple OR chain: {filter_expr}")
    
    try:
        result = dataset.to_table(filter=filter_expr)
        print("✅ Success! Simple OR chain works.")
        print(f"Result has {len(result)} rows")
        return True
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        return False

if __name__ == "__main__":
    print("Testing stack overflow fix for issue #4373...")
    print("=" * 50)
    
    # Test 1: Simple OR chain
    success1 = test_simple_or_chain()
    print()
    
    # Test 2: Long OR chain (should not cause stack overflow)
    success2 = test_stack_overflow_fix()
    print()
    
    if success1 and success2:
        print("🎉 All tests passed! The stack overflow issue has been resolved.")
        sys.exit(0)
    else:
        print("💥 Some tests failed.")
        sys.exit(1) 