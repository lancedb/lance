use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::displayable;
use datafusion::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // Create test data
    let ctx = SessionContext::new();
    
    // Create large table
    let large_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("data", DataType::Utf8, false),
    ]));
    
    let large_batch = RecordBatch::try_new(
        large_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])),
            Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])),
        ],
    )?;
    
    ctx.register_batch("large", large_batch)?;
    
    // Create small exclusion table
    let small_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
    ]));
    
    let small_batch = RecordBatch::try_new(
        small_schema.clone(),
        vec![Arc::new(Int32Array::from(vec![2, 4, 6]))],
    )?;
    
    ctx.register_batch("small", small_batch)?;
    
    // Test query: NOT EXISTS with LIMIT
    let sql = "SELECT * FROM large WHERE NOT EXISTS (SELECT 1 FROM small WHERE small.id = large.id) LIMIT 5";
    
    println!("=== SQL Query ===");
    println!("{}\n", sql);
    
    // Get logical plan
    let df = ctx.sql(sql).await?;
    let logical_plan = df.logical_plan();
    println!("=== Logical Plan ===");
    println!("{}\n", logical_plan.display_indent());
    
    // Get initial physical plan (before optimization)
    let physical_plan = df.create_physical_plan().await?;
    println!("=== Initial Physical Plan ===");
    println!("{}\n", displayable(&*physical_plan).indent(true));
    
    // Check what methods are available
    println!("=== Checking fetch() on physical plan ===");
    println!("Root plan type: {:?}", physical_plan.name());
    println!("Root plan has fetch(): {:?}", physical_plan.fetch());
    
    // Check children
    let children = physical_plan.children();
    if !children.is_empty() {
        println!("First child type: {:?}", children[0].name());
        println!("First child has fetch(): {:?}", children[0].fetch());
    }
    
    Ok(())
}