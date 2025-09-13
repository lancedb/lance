#!/usr/bin/env python3
"""
DataFusion Plan Stats Tracing Example

This example demonstrates how to use Lance's DataFusion plan stats tracing
to monitor query execution performance and resource usage.

To enable DataFusion tracing, set the environment variables:
    export LANCE_DATAFUSION_TRACING=1
    export LANCE_LOG=trace  # trace level required to see detailed events

The tracer collects comprehensive metrics including:
- Execution time
- I/O statistics (IOPS, bytes read, request patterns)  
- Scan efficiency (rows, fragments, ranges scanned)
- Memory usage and spill statistics
- Complete execution plan structure

Trace events are emitted to the "lance::datafusion" target with type="plan_run"
"""

import lance
import lance.tracing
from lance.dataset import write_dataset
from lance import __version__
import pyarrow as pa
import pyarrow.compute as pc
import tempfile
import os
from typing import Dict, Any
import datafusion as df

def create_sample_dataset(uri: str, num_rows: int = 10000):
    """Create a sample dataset for demonstration."""
    print(f"Creating sample dataset with {num_rows} rows at {uri}")
    
    # Create sample data with various column types
    data = {
        "id": range(num_rows),
        "name": [f"user_{i}" for i in range(num_rows)],
        "age": [20 + (i % 60) for i in range(num_rows)],
        "score": [i * 0.1 for i in range(num_rows)],
        "category": [f"cat_{i % 10}" for i in range(num_rows)],
        "vector": [[float(i + j) for j in range(128)] for i in range(num_rows)],
    }
    
    table = pa.table(data)
    dataset = write_dataset(table, uri)
    print(f"Dataset created with {dataset.count_rows()} rows")
    return dataset


def setup_tracing():
    """Setup tracing to capture DataFusion events."""
    print("Setting up DataFusion tracing...")
    
    # Enable DataFusion tracing via environment variable
    os.environ["LANCE_DATAFUSION_TRACING"] = "1"
    
    # Enable Lance logging to see trace events in console
    # Set to "trace" to see detailed trace events
    if "LANCE_LOG" not in os.environ:
        os.environ["LANCE_LOG"] = "trace"
        print("Set LANCE_LOG=trace to see detailed trace events in console")
    
    # List to store captured trace events
    trace_events = []
    
    def trace_callback(event):
        """Callback to process trace events."""
        # Look for DataFusion-specific execution events
        if (event.target == "lance::datafusion" and 
            event.args.get("type") == "datafusion_plan_run"):
            trace_events.append({
                "target": event.target,
                "args": dict(event.args)
            })
            print(f"\n🔍 DataFusion Execution Event Captured:")
            print(f"   Target: {event.target}")
            print(f"   Type: {event.args.get('type')}")
            print(f"   Args: {dict(event.args)}")
    
    # Start capturing trace events
    lance.tracing.capture_trace_events(trace_callback)
    
    return trace_events


def demonstrate_basic_query(dataset, trace_events: list):
    """Demonstrate basic query with tracing."""
    print("\n" + "="*60)
    print("🚀 Executing Basic Query with NATIVE Lance Tracing")
    print("="*60)
    
    # Clear previous events
    trace_events.clear()
    
    # Execute a simple query
    print("Query: SELECT * FROM dataset WHERE age > 30 LIMIT 100")
    
    result = (dataset
             .to_table(filter=pc.greater(pc.field("age"), 30))
             .slice(0, 100))
    
    print(f"✅ Query completed. Result has {len(result)} rows")
    
    # Display any captured trace events
    if trace_events:
        print(f"\n📊 Captured {len(trace_events)} trace events")
        for i, event in enumerate(trace_events):
            print(f"\nEvent {i+1}:")
            args = event["args"]
            if "execution_time_ms" in args:
                print(f"  ⏱️  Execution Time: {args['execution_time_ms']}ms")
            if "bytes_read" in args:
                mb_read = int(args['bytes_read']) / (1024 * 1024)
                print(f"  💾 Data Read: {mb_read:.2f} MB")
            if "rows_scanned" in args:
                print(f"  📋 Rows Scanned: {args['rows_scanned']}")
            if "iops" in args:
                print(f"  🔄 I/O Operations: {args['iops']}")
    else:
        print("⚠️  No DataFusion trace events captured")
        print("    Make sure LANCE_DATAFUSION_TRACING environment variable is set")


def demonstrate_complex_query(dataset, trace_events: list):
    """Demonstrate complex query with aggregations using DataFusion."""
    print("\n" + "="*60)
    print("🔍 Executing Complex Query with DataFusion Integration")
    print("="*60)
    
    # Clear previous events
    trace_events.clear()
    
    # Create a scalar index on age column for better performance
    print("Creating scalar index on 'age' column...")
    try:
        dataset.create_scalar_index("age", index_type="BTREE")
        print("✅ Created BTREE scalar index on 'age' column")
    except Exception as e:
        print(f"⚠️  Index creation failed (may already exist): {e}")
    
    # Execute a more complex query with filtering and multiple operations
    print("Query: Complex filtering and projection with multiple conditions (using scalar index)")
    
    
        
        # Create DataFusion session context
    ctx = df.SessionContext()
    
    # Create a table provider for DataFusion integration
    table_provider = lance.FFILanceTableProvider(dataset)
    print(f"✅ Created DataFusion table provider")
    
    # Register the Lance dataset as a table in DataFusion
    ctx.register_table_provider("lance_dataset", table_provider)
    print(f"✅ Registered Lance table in DataFusion context")
    
    # Execute SQL query directly through DataFusion
    # This will trigger our DataFusion tracing infrastructure and should use the scalar index
    sql_query = """
    SELECT category, AVG(score) as avg_score, COUNT(*) as row_count
    FROM lance_dataset 
    WHERE age >= 25 AND age <= 45 
    GROUP BY category
    ORDER BY avg_score DESC
    LIMIT 5
    """
    
    print("📋 Expected trace metrics with scalar index:")
    print("  - Lower bytes_read (index should filter efficiently)")
    print("  - indices_loaded > 0 (scalar index loaded)")
    print("  - index_comparisons > 0 (index search operations)")
    print("  - Reduced rows_scanned vs total dataset size")
    
    print(f"🔍 Executing SQL query: {sql_query.strip()}")
    df1 = ctx.sql(sql_query)
    result = df1.collect()
    
    print(f"✅ DataFusion SQL query completed. Result has {len(result)} batches")
        
    
    # Display captured trace events
    if trace_events:
        print(f"\n📊 Captured {len(trace_events)} trace events from complex operations")
        for i, event in enumerate(trace_events):
            print(f"\nEvent {i+1}:")
            args = event["args"]
            if "execution_time_ms" in args:
                print(f"  ⏱️  Execution Time: {args['execution_time_ms']}ms")
            if "bytes_read" in args:
                mb_read = int(args['bytes_read']) / (1024 * 1024)
                print(f"  💾 Data Read: {mb_read:.2f} MB")
            if "rows_scanned" in args:
                print(f"  📋 Rows Scanned: {args['rows_scanned']}")
            if "indices_loaded" in args:
                print(f"  🗃️  Indices Loaded: {args['indices_loaded']}")
            if "index_comparisons" in args:
                print(f"  🔍 Index Comparisons: {args['index_comparisons']}")
            if "fragments_scanned" in args:
                print(f"  🗂️  Fragments Scanned: {args['fragments_scanned']}")
            if "iops" in args:
                print(f"  🔄 I/O Operations: {args['iops']}")
    else:
        print("ℹ️  No trace events captured - this may indicate the operations didn't use DataFusion")
        print("    Make sure LANCE_DATAFUSION_TRACING environment variable is set")


def demonstrate_vector_search(dataset, trace_events: list):
    """Demonstrate vector search with tracing."""
    print("\n" + "="*60)
    print("🎯 Executing Vector Search with DataFusion Tracing")
    print("="*60)
    
    # Clear previous events  
    trace_events.clear()
    
    # Create a vector index for better performance
    print("Creating vector index...")
    try:
        dataset.create_index(
            "vector",
            index_type="IVF_PQ",
            num_partitions=8,
            num_sub_vectors=16
        )
        print("✅ Vector index created")
    except Exception as e:
        print(f"⚠️  Index creation failed (may already exist): {e}")
    
    # Perform vector search
    query_vector = [1.0] * 128
    print(f"Query: Vector search for top 10 similar vectors")
    
    result = (dataset
             .to_table(nearest={"column": "vector", "q": query_vector, "k": 10}))
    
    print(f"✅ Vector search completed. Found {len(result)} results")
    
    # Display trace events
    if trace_events:
        print(f"\n📊 Captured {len(trace_events)} trace events from vector search")
        for i, event in enumerate(trace_events):
            print(f"\nEvent {i+1}:")
            args = event["args"]
            if "execution_time_ms" in args:
                print(f"  ⏱️  Execution Time: {args['execution_time_ms']}ms")
            if "ranges_scanned" in args:
                print(f"  🎯 Ranges Scanned: {args['ranges_scanned']}")


def demonstrate_chrome_tracing(dataset):
    """Demonstrate Chrome tracing output."""
    print("\n" + "="*60)
    print("🌐 Demonstrating Chrome Tracing Output")
    print("="*60)
    
    # Create a temporary file for chrome trace
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        trace_file = f.name
    
    try:
        print(f"Starting Chrome tracing to: {trace_file}")
        
        # Start chrome tracing
        lance.tracing.trace_to_chrome(file=trace_file)
        
        # Execute some queries
        print("Executing queries for Chrome trace...")
        
        # Simple scan
        result1 = dataset.to_table(limit=1000)
        print(f"✅ Scanned {len(result1)} rows")
        
        # Filtered scan
        result2 = (dataset
                  .to_table(filter=pc.greater(pc.field("score"), 500.0))
                  .slice(0, 50))
        print(f"✅ Filtered scan returned {len(result2)} rows")
        
        print(f"\n📊 Chrome trace written to: {trace_file}")
        print("🌐 Open this file in Chrome at chrome://tracing or https://ui.perfetto.dev/")
        print("   The trace will show detailed execution timelines and performance data")
        
    finally:
        # Clean up trace file after demo
        if os.path.exists(trace_file):
            print(f"\n🧹 Cleaning up trace file: {trace_file}")
            os.unlink(trace_file)


def main():
    """Main demonstration function."""
    print("🚀 Lance DataFusion Plan Stats Tracing Example")
    print("=" * 60)
    print(f"📦 Lance version: {lance.__version__}")
    print("=" * 60)
    
    # Create temporary dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_uri = os.path.join(temp_dir, "sample_dataset")
        
        # Create sample dataset
        dataset = create_sample_dataset(dataset_uri, num_rows=5000)
        
        # Setup tracing
        trace_events = setup_tracing()
        print(f"Trace events: starting query")
        # Demonstrate different query types
       # demonstrate_basic_query(dataset, trace_events)
        demonstrate_complex_query(dataset, trace_events)
        # demonstrate_vector_search(dataset, trace_events)
        
        # # Demonstrate Chrome tracing
        # demonstrate_chrome_tracing(dataset)
    
    print("\n" + "="*60)
    print("✅ DataFusion Tracing Example Completed!")
    print("="*60)
    
    print("\n📋 Summary of DataFusion Tracing Features:")
    print("• Comprehensive execution metrics (time, I/O, memory)")
    print("• Structured plan statistics in JSON format")
    print("• Real-time trace event capture with callbacks")
    print("• Chrome tracing integration for visual analysis")
    print("• Automatic resource usage monitoring")
    print("• Support for all query types (scans, filters, aggregations, vector search)")
    
    print("\n🚀 To run this example with full tracing enabled:")
    print("   export LANCE_DATAFUSION_TRACING=1")
    print("   export LANCE_LOG=trace")
    print("   python examples/datafusion_tracing_example.py")


if __name__ == "__main__":
    main()
