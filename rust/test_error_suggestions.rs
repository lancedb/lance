// Test file for error suggestions
use lance_core::{levenshtein::find_best_suggestion, DataType, Field, Schema};
use lance_index::IndexType;
use lance_linalg::DistanceType;

fn test_column_not_found() {
    println!("=== Testing Column Not Found Errors ===");

    // Create a schema with some fields
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("vector", DataType::Float32, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "metadata",
            DataType::Struct(
                vec![
                    Field::new("name", DataType::Utf8, false),
                    Field::new("value", DataType::Float32, false),
                ]
                .into(),
            ),
            false,
        ),
    ]);

    // Test 1: Incorrect field name with suggestion
    let result = schema.field_id("vacter");
    println!("Test 1 - Field 'vacter': {}", result.unwrap_err());

    // Test 2: Incorrect nested field name
    let result = schema.field_id("metadata.name");
    println!("Test 2 - Field 'metadata.name': {}", result.unwrap_err());

    // Test 3: Completely wrong name
    let result = schema.field_id("completely_wrong");
    println!("Test 3 - Field 'completely_wrong': {}", result.unwrap_err());
}

fn test_distance_type_not_found() {
    println!("\n=== Testing Distance Type Not Found Errors ===");

    // Test 1: Misspelled distance type with suggestion
    let result = DistanceType::try_from("l1");
    println!("Test 1 - Distance 'l1': {}", result.unwrap_err());

    // Test 2: Another misspelled distance type
    let result = DistanceType::try_from("cosin");
    println!("Test 2 - Distance 'cosin': {}", result.unwrap_err());

    // Test 3: Completely wrong distance type
    let result = DistanceType::try_from("wrong_distance");
    println!(
        "Test 3 - Distance 'wrong_distance': {}",
        result.unwrap_err()
    );
}

fn test_index_type_not_found() {
    println!("\n=== Testing Index Type Not Found Errors ===");

    // Test 1: Misspelled index type with suggestion
    let result = IndexType::try_from("Btree");
    println!("Test 1 - Index 'Btree': {}", result.unwrap_err());

    // Test 2: Another misspelled index type
    let result = IndexType::try_from("Vectr");
    println!("Test 2 - Index 'Vectr': {}", result.unwrap_err());

    // Test 3: Completely wrong index type
    let result = IndexType::try_from("wrong_index");
    println!("Test 3 - Index 'wrong_index': {}", result.unwrap_err());
}

fn test_levenshtein_suggestion() {
    println!("\n=== Testing Levenshtein Suggestion Function ===");

    let options = vec!["vector", "id", "text", "metadata.name"];

    // Test with 1 character difference
    let suggestion = find_best_suggestion("vacter", &options);
    println!("Test 1 - 'vacter' -> {:?}", suggestion);

    // Test with 2 character differences
    let suggestion = find_best_suggestion("vecor", &options);
    println!("Test 2 - 'vecor' -> {:?}", suggestion);

    // Test with more than 1/3 characters different
    let suggestion = find_best_suggestion("vctr", &options);
    println!("Test 3 - 'vctr' -> {:?}", suggestion);

    // Test exact match
    let suggestion = find_best_suggestion("vector", &options);
    println!("Test 4 - 'vector' -> {:?}", suggestion);
}

fn main() {
    println!("Testing Enhanced Error Messages with Suggestions");
    println!("==============================================");

    test_levenshtein_suggestion();
    test_column_not_found();
    test_distance_type_not_found();
    test_index_type_not_found();

    println!("\nAll tests completed!");
}
