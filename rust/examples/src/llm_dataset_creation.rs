use anyhow::Result;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use lance::dataset::Dataset;
use reqwest::Client;
use std::sync::Arc;
use tantivy::tokenizer::{LowerCaser, SimpleTokenizer, TextAnalyzer};

#[tokio::main]
async fn main() -> Result<()> {
    // Download Wikitext dataset
    println!("Downloading Wikitext dataset...");
    let client = Client::new();
    let response = client
        .get("https://raw.githubusercontent.com/huggingface/datasets/master/datasets/wikitext/wikitext-2-raw-v1/train.txt")
        .send()
        .await?;

    let text = response.text().await?;
    println!("Downloaded {} bytes of text", text.len());

    // Create a tokenizer using tantivy
    let mut tokenizer = TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(LowerCaser)
        .build();

    println!("Tokenizing text...");
    // Tokenize text line by line
    let mut tokenized_lines = Vec::new();
    let mut total_tokens = 0;

    for line in text.lines() {
        if !line.trim().is_empty() {
            let mut token_stream = tokenizer.token_stream(line);
            let mut tokens = Vec::new();

            while token_stream.advance() {
                tokens.push(token_stream.token().text.clone());
            }

            if !tokens.is_empty() {
                total_tokens += tokens.len();
                tokenized_lines.push(tokens);
            }
        }
    }

    println!(
        "Tokenized {} lines with {} total tokens",
        tokenized_lines.len(),
        total_tokens
    );

    // Create Arrow arrays and record batch
    let mut token_builder = arrow::array::ListBuilder::new(arrow::array::StringBuilder::new());

    for tokens in &tokenized_lines {
        let values = token_builder.values();
        for token in tokens {
            values.append_value(token);
        }
        token_builder.append(true);
    }

    let token_array = token_builder.finish();

    // Create schema
    let schema = Schema::new(vec![Field::new(
        "tokens",
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        false,
    )]);

    // Create record batch
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(token_array)])?;
    println!("Writing dataset to disk...");

    // Create a reader for the batch
    let batches = vec![batch];
    let reader = arrow::record_batch::RecordBatchIterator::new(
        batches.into_iter().map(Ok),
        Arc::new(schema.clone()),
    );

    // Write the dataset
    Dataset::write(reader, "wikitext_tokenized.lance", None).await?;

    // Read and display dataset info
    let dataset = Dataset::open("wikitext_tokenized.lance").await?;
    println!("Dataset schema: {:?}", dataset.schema());

    // Count rows using the dataset's count method
    let count = dataset.count_rows(None).await?;
    println!("Number of rows: {}", count);

    Ok(())
}
