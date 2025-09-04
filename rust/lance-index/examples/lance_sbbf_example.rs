// Example showing how to use the Lance SBBF (Split Block Bloom Filter)
// This is a standalone implementation based on Arrow-rs Parquet bloom filter
// but with all public APIs for use in Lance

use arrow::array::*;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use lance_index::scalar::bloomfilter::{Sbbf, SbbfBuilder};
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Lance SBBF (Split Block Bloom Filter) Example\n");

    // Example 1: Basic SBBF usage with different types
    demonstrate_basic_sbbf_usage()?;

    // Example 2: SBBF with Arrow RecordBatch
    demonstrate_sbbf_with_arrow_data()?;

    // Example 3: SBBF serialization and deserialization
    demonstrate_sbbf_serialization()?;

    // Example 4: Performance comparison with different configurations
    demonstrate_sbbf_performance()?;

    println!("✅ All Lance SBBF examples completed!");
    Ok(())
}

/// Example 1: Basic SBBF usage with different data types
fn demonstrate_basic_sbbf_usage() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Example 1: Basic SBBF Usage ===");

    // Create SBBF using builder pattern
    let mut sbbf = SbbfBuilder::new()
        .expected_items(1000) // Expected number of distinct values
        .false_positive_probability(0.01) // 1% false positive rate
        .build()?;

    println!(
        "Created SBBF with {} blocks ({} bytes)",
        sbbf.num_blocks(),
        sbbf.size_bytes()
    );

    // Insert different types of data
    println!("\nInserting various data types:");

    // Strings
    let strings = ["alice", "bob", "charlie", "diana", "eve"];
    for &name in &strings {
        sbbf.insert(name);
        println!("  ✓ Inserted string: '{}'", name);
    }

    // Integers
    let integers = [42i32, 100, -5, 999, 2024];
    for &num in &integers {
        sbbf.insert(&num);
        println!("  ✓ Inserted i32: {}", num);
    }

    // Floats (need to convert to bytes since f64 doesn't implement Hash directly)
    let floats = [3.14159f64, 2.71828, 1.41421];
    for &val in &floats {
        sbbf.insert(&val);
        println!("  ✓ Inserted f64: {}", val);
    }

    // Boolean values
    let bools = [true, false];
    for &val in &bools {
        sbbf.insert(&val);
        println!("  ✓ Inserted bool: {}", val);
    }

    // Bytes
    let bytes_data = b"binary_data";
    sbbf.insert(&bytes_data[..]);
    println!("  ✓ Inserted bytes: {:?}", bytes_data);

    // Test membership queries
    println!("\n--- Membership Tests ---");

    // Test strings
    assert!(sbbf.check("alice"));
    assert!(sbbf.check("charlie"));
    assert!(!sbbf.check("unknown"));
    println!("  ✓ 'alice': {}", sbbf.check("alice"));
    println!("  ✓ 'unknown': {}", sbbf.check("unknown"));

    // Test integers
    assert!(sbbf.check(&42i32));
    assert!(sbbf.check(&999i32));
    assert!(!sbbf.check(&777i32));
    println!("  ✓ 42: {}", sbbf.check(&42i32));
    println!("  ✓ 777: {}", sbbf.check(&777i32));

    // Test floats
    assert!(sbbf.check(&3.14159f64));
    assert!(!sbbf.check(&6.28318f64));
    println!("  ✓ 3.14159: {}", sbbf.check(&3.14159f64));
    println!("  ✓ 6.28318: {}", sbbf.check(&6.28318f64));

    // Test booleans
    assert!(sbbf.check(&true));
    assert!(sbbf.check(&false));
    println!("  ✓ true: {}", sbbf.check(&true));

    // Test bytes
    assert!(sbbf.check(&bytes_data[..]));
    assert!(!sbbf.check(&b"other_data"[..]));
    println!("  ✓ binary_data: {}", sbbf.check(&bytes_data[..]));

    println!("\n✅ Basic SBBF usage works perfectly!\n");
    Ok(())
}

/// Example 2: SBBF with Arrow RecordBatch data
fn demonstrate_sbbf_with_arrow_data() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Example 2: SBBF with Arrow Data ===");

    // Create a RecordBatch with employee data
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("department", DataType::Utf8, false),
        Field::new("salary", DataType::Float64, false),
        Field::new("active", DataType::Boolean, false),
    ]);

    let id_array = Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let name_array = StringArray::from(vec![
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack",
    ]);
    let dept_array = StringArray::from(vec![
        "Engineering",
        "Sales",
        "Engineering",
        "Marketing",
        "Sales",
        "Engineering",
        "HR",
        "Sales",
        "Marketing",
        "Engineering",
    ]);
    let salary_array = Float64Array::from(vec![
        75000.0, 65000.0, 85000.0, 60000.0, 70000.0, 90000.0, 55000.0, 68000.0, 62000.0, 80000.0,
    ]);
    let active_array = BooleanArray::from(vec![
        true, true, false, true, true, true, false, true, true, false,
    ]);

    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(id_array),
            Arc::new(name_array),
            Arc::new(dept_array),
            Arc::new(salary_array),
            Arc::new(active_array),
        ],
    )?;

    println!(
        "Processing RecordBatch with {} rows, {} columns",
        batch.num_rows(),
        batch.num_columns()
    );

    // Create separate SBBF filters for different columns
    let mut id_filter = SbbfBuilder::new()
        .expected_items(20)
        .false_positive_probability(0.01)
        .build()?;
    let mut name_filter = SbbfBuilder::new()
        .expected_items(20)
        .false_positive_probability(0.01)
        .build()?;
    let mut dept_filter = SbbfBuilder::new()
        .expected_items(10)
        .false_positive_probability(0.01)
        .build()?;
    let mut salary_filter = SbbfBuilder::new()
        .expected_items(20)
        .false_positive_probability(0.01)
        .build()?;

    println!("\nBuilding bloom filters for each column:");

    // Process ID column
    let id_column = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    for i in 0..id_column.len() {
        let value = id_column.value(i);
        id_filter.insert(&value);
    }
    println!("  ✓ Built ID filter");

    // Process Name column
    let name_column = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for i in 0..name_column.len() {
        let value = name_column.value(i);
        name_filter.insert(value);
    }
    println!("  ✓ Built Name filter");

    // Process Department column
    let dept_column = batch
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for i in 0..dept_column.len() {
        let value = dept_column.value(i);
        dept_filter.insert(value);
    }
    println!("  ✓ Built Department filter");

    // Process Salary column
    let salary_column = batch
        .column(3)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    for i in 0..salary_column.len() {
        let value = salary_column.value(i);
        salary_filter.insert(&value);
    }
    println!("  ✓ Built Salary filter");

    // Test queries
    println!("\n--- Arrow Data Query Tests ---");

    println!("ID queries:");
    assert!(id_filter.check(&5i32));
    assert!(id_filter.check(&10i32));
    assert!(!id_filter.check(&15i32));
    println!("  ✓ ID 5 exists: {}", id_filter.check(&5i32));
    println!("  ✓ ID 15 exists: {}", id_filter.check(&15i32));

    println!("Name queries:");
    assert!(name_filter.check("Alice"));
    assert!(name_filter.check("Jack"));
    assert!(!name_filter.check("Unknown"));
    println!("  ✓ 'Alice' exists: {}", name_filter.check("Alice"));
    println!("  ✓ 'Unknown' exists: {}", name_filter.check("Unknown"));

    println!("Department queries:");
    assert!(dept_filter.check("Engineering"));
    assert!(dept_filter.check("Marketing"));
    assert!(!dept_filter.check("Finance"));
    println!(
        "  ✓ 'Engineering' exists: {}",
        dept_filter.check("Engineering")
    );
    println!("  ✓ 'Finance' exists: {}", dept_filter.check("Finance"));

    println!("Salary queries:");
    assert!(salary_filter.check(&75000.0f64));
    assert!(salary_filter.check(&90000.0f64));
    assert!(!salary_filter.check(&100000.0f64));
    println!(
        "  ✓ Salary 75000 exists: {}",
        salary_filter.check(&75000.0f64)
    );
    println!(
        "  ✓ Salary 100000 exists: {}",
        salary_filter.check(&100000.0f64)
    );

    println!("\n✅ Arrow data integration works perfectly!\n");
    Ok(())
}

/// Example 3: SBBF serialization and deserialization
fn demonstrate_sbbf_serialization() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Example 3: SBBF Serialization ===");

    // Create and populate an SBBF
    let mut original_sbbf = SbbfBuilder::new()
        .expected_items(100)
        .false_positive_probability(0.01)
        .build()?;

    println!(
        "Original SBBF: {} blocks, {} bytes",
        original_sbbf.num_blocks(),
        original_sbbf.size_bytes()
    );

    // Insert test data
    let test_data = ["apple", "banana", "cherry", "date", "elderberry"];
    for &item in &test_data {
        original_sbbf.insert(item);
    }

    // Serialize to bytes
    let serialized_bytes = original_sbbf.to_bytes();
    println!("Serialized to {} bytes", serialized_bytes.len());

    // Deserialize from bytes
    let deserialized_sbbf = Sbbf::new(&serialized_bytes)?;
    println!(
        "Deserialized SBBF: {} blocks, {} bytes",
        deserialized_sbbf.num_blocks(),
        deserialized_sbbf.size_bytes()
    );

    // Verify the deserialized filter works correctly
    println!("\n--- Serialization Verification ---");
    for &item in &test_data {
        assert!(deserialized_sbbf.check(item));
        println!(
            "  ✓ '{}' exists in deserialized filter: {}",
            item,
            deserialized_sbbf.check(item)
        );
    }

    assert!(!deserialized_sbbf.check("not_inserted"));
    println!(
        "  ✓ 'not_inserted' exists: {}",
        deserialized_sbbf.check("not_inserted")
    );

    // Test write to writer
    let mut buffer = Vec::new();
    deserialized_sbbf.write_bitset(&mut buffer)?;
    println!("Written {} bytes to buffer", buffer.len());

    println!("\n✅ Serialization/deserialization works perfectly!\n");
    Ok(())
}

/// Example 4: Performance comparison with different configurations
fn demonstrate_sbbf_performance() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Example 4: SBBF Performance ===");

    let test_size = 10000;
    let test_data: Vec<String> = (0..test_size)
        .map(|i| format!("test_item_{:06}", i))
        .collect();

    println!("Performance test with {} items", test_size);

    // Test different FPP configurations
    let fpp_configs = [0.1, 0.01, 0.001];

    for &fpp in &fpp_configs {
        println!("\n--- FPP: {} ---", fpp);

        let mut sbbf = SbbfBuilder::new()
            .expected_items(test_size as u64)
            .false_positive_probability(fpp)
            .build()?;

        println!(
            "  Filter size: {} blocks, {} bytes",
            sbbf.num_blocks(),
            sbbf.size_bytes()
        );

        // Measure insert time
        let start = Instant::now();
        for item in &test_data {
            sbbf.insert(item.as_str());
        }
        let insert_duration = start.elapsed();

        // Measure query time (positive queries)
        let start = Instant::now();
        let mut hits = 0;
        for i in 0..1000 {
            let query = format!("test_item_{:06}", i);
            if sbbf.check(query.as_str()) {
                hits += 1;
            }
        }
        let query_duration = start.elapsed();

        // Measure query time (negative queries)
        let start = Instant::now();
        let mut false_positives = 0;
        for i in test_size..(test_size + 1000) {
            let query = format!("test_item_{:06}", i);
            if sbbf.check(query.as_str()) {
                false_positives += 1;
            }
        }
        let negative_query_duration = start.elapsed();

        println!(
            "  Insert time: {:?} ({:.2} ns/item)",
            insert_duration,
            insert_duration.as_nanos() as f64 / test_size as f64
        );
        println!(
            "  Positive query time: {:?} ({:.2} ns/query)",
            query_duration,
            query_duration.as_nanos() as f64 / 1000.0
        );
        println!(
            "  Negative query time: {:?} ({:.2} ns/query)",
            negative_query_duration,
            negative_query_duration.as_nanos() as f64 / 1000.0
        );
        println!("  Positive hits: {}/1000 ({}%)", hits, hits as f64 / 10.0);
        println!(
            "  False positives: {}/1000 ({:.2}%)",
            false_positives,
            false_positives as f64 / 10.0
        );
        println!("  Memory usage: {} bytes", sbbf.estimated_memory_size());
    }

    println!("\n🚀 KEY SBBF ADVANTAGES:");
    println!("  ✅ Compatible with Parquet SBBF specification");
    println!("  ✅ SIMD-optimized block operations (8 hash functions)");
    println!("  ✅ All public APIs - no pub(crate) restrictions");
    println!("  ✅ Comprehensive type support via AsBytes trait");
    println!("  ✅ Configurable false positive rates");
    println!("  ✅ Efficient serialization/deserialization");
    println!("  ✅ Perfect for Lance columnar indexing");

    println!("\n✅ Performance testing completed!\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lance_sbbf_with_arrow_strings() {
        let mut sbbf = SbbfBuilder::new()
            .expected_items(100)
            .false_positive_probability(0.01)
            .build()
            .unwrap();

        let string_array = StringArray::from(vec!["test1", "test2", "test3"]);

        // Insert all values
        for i in 0..string_array.len() {
            sbbf.insert(string_array.value(i));
        }

        // Test membership
        for i in 0..string_array.len() {
            assert!(sbbf.check(string_array.value(i)));
        }

        assert!(!sbbf.check("not_inserted"));
    }

    #[test]
    fn test_lance_sbbf_serialization() {
        let mut sbbf1 = SbbfBuilder::new()
            .expected_items(50)
            .false_positive_probability(0.01)
            .build()
            .unwrap();

        // Insert test data
        for i in 0..20 {
            sbbf1.insert(&i);
        }

        // Serialize and deserialize
        let bytes = sbbf1.to_bytes();
        let sbbf2 = Sbbf::new(&bytes).unwrap();

        // Verify deserialized filter works
        for i in 0..20 {
            assert!(sbbf2.check(&i));
        }
        assert!(!sbbf2.check(&999));
    }
}
