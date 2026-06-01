// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! End-to-end FM-Index vs N-Gram benchmark through the Lance dataset API.
//!
//! Creates a Lance dataset, builds FM-Index and N-Gram indices, then queries
//! using `dataset.scan().filter("contains(content, 'pattern')")`.
//!
//! Usage:
//!   SOURCE_ROOT=/path/to/code cargo run --release -p lance --example fmindex_lance_e2e
//!   FMINDEX_DATA_PATH=/path/to/data.txt cargo run --release -p lance --example fmindex_lance_e2e

use std::sync::Arc;
use std::time::Instant;

use arrow_array::{LargeStringArray, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use lance::dataset::Dataset;
use lance::index::DatasetIndexExt;
use lance_index::IndexType;
use lance_index::scalar::ScalarIndexParams;

async fn count_rows(dataset: &Dataset) -> usize {
    dataset.count_rows(None).await.unwrap()
}
fn load_texts_from_file(path: &str) -> Vec<String> {
    let text = std::fs::read_to_string(path).expect("Cannot read file");
    text.lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect()
}

fn load_source_files(root: &str, max: usize) -> Vec<String> {
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().map_or(false, |e| {
            e == "rs" || e == "py" || e == "toml" || e == "md"
        }) {
            if let Ok(content) = std::fs::read_to_string(path) {
                if !content.is_empty() {
                    files.push(content);
                    if files.len() >= max {
                        break;
                    }
                }
            }
        }
    }
    files
}

async fn run_query(dataset: &Dataset, pattern: &str) -> usize {
    let filter = format!("contains(content, '{}')", pattern.replace('\'', "''"));
    dataset.count_rows(Some(filter)).await.unwrap_or(0)
}

#[tokio::main]
async fn main() {
    let texts = if let Ok(path) = std::env::var("FMINDEX_DATA_PATH") {
        println!("Loading from text file: {path}");
        let max: usize = std::env::var("MAX_ROWS")
            .unwrap_or("100000".into())
            .parse()
            .unwrap();
        let mut t = load_texts_from_file(&path);
        t.truncate(max);
        t
    } else {
        let root = std::env::var("SOURCE_ROOT").unwrap_or_else(|_| ".".to_string());
        let max: usize = std::env::var("MAX_FILES")
            .unwrap_or("5000".into())
            .parse()
            .unwrap();
        println!("Loading source files from {root}...");
        load_source_files(&root, max)
    };

    let total = texts.len();
    let total_bytes: usize = texts.iter().map(|t| t.len()).sum();
    println!(
        "Loaded {} documents, {:.1} MB total, avg {} bytes/doc",
        total,
        total_bytes as f64 / 1e6,
        total_bytes / total.max(1)
    );

    // Create Lance dataset
    let schema = Arc::new(Schema::new(vec![Field::new(
        "content",
        DataType::LargeUtf8,
        false,
    )]));

    let dataset_dir =
        std::env::var("DATASET_DIR").unwrap_or_else(|_| "/tmp/fmindex_bench".to_string());
    std::fs::create_dir_all(&dataset_dir).ok();
    let fm_path = format!("{}/fm_dataset.lance", dataset_dir);
    let ngram_path = format!("{}/ngram_dataset.lance", dataset_dir);

    println!("Writing Lance dataset to {fm_path}...");
    let t0 = Instant::now();
    let batch_size = 10000;
    let mut batches: Vec<RecordBatch> = Vec::new();
    for chunk in texts.chunks(batch_size) {
        let col = Arc::new(LargeStringArray::from(
            chunk.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ));
        batches.push(RecordBatch::try_new(schema.clone(), vec![col]).unwrap());
    }
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(reader, &fm_path, None).await.unwrap();
    println!(
        "Dataset created: {} rows in {:.2}s",
        count_rows(&dataset).await,
        t0.elapsed().as_secs_f64()
    );

    // Queries
    let short_queries = vec![
        "fn ", "use ", "self.", "impl ", "pub ", "async ", "Result", "let mut ", "struct ",
        "#[test]",
    ];
    let medium_queries = vec![
        "fn main()",
        "use std::sync::Arc",
        "impl Default for",
        "async fn search",
        "#[derive(Debug)]",
    ];
    let long_queries: Vec<String> = texts
        .iter()
        .take(200)
        .filter_map(|t| {
            let chars: Vec<char> = t.chars().collect();
            if chars.len() > 100 {
                let start = chars.len() / 4;
                let end = (start + 80).min(chars.len());
                let q: String = chars[start..end].iter().collect();
                if !q.contains('\n') && !q.contains('\r') && !q.contains('\'') {
                    Some(q)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .take(20)
        .collect();

    println!(
        "\nQueries: {} short, {} medium, {} long",
        short_queries.len(),
        medium_queries.len(),
        long_queries.len()
    );

    // ── Baseline: no index (full scan) ──
    println!("\n============================================================");
    println!("=== No Index (full scan) ===");
    for (label, queries) in [("Short", &short_queries)] {
        let t0 = Instant::now();
        for q in queries.iter().take(3) {
            let count = run_query(&dataset, q).await;
            println!("  \"{}\" -> {} matches", q, count);
        }
        let elapsed = t0.elapsed();
        println!(
            "{label}: {:.2}ms for {} queries",
            elapsed.as_secs_f64() * 1000.0,
            3
        );
    }

    // ── FM-Index ──
    println!("\n============================================================");
    println!("=== FM-Index ===");
    let t0 = Instant::now();
    let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::FMIndex);
    dataset
        .create_index(&["content"], IndexType::FMIndex, None, &params, true)
        .await
        .unwrap();
    println!("Index build: {:.2}s", t0.elapsed().as_secs_f64());

    // Reload to pick up the index
    let dataset = Dataset::open(&fm_path).await.unwrap();

    for (label, queries) in [
        (
            "Short",
            short_queries
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        ),
        (
            "Medium",
            medium_queries.iter().map(|s| s.to_string()).collect(),
        ),
        ("Long", long_queries.clone()),
    ] {
        println!("\n--- {label} queries ---");
        let t0 = Instant::now();
        let mut results = Vec::new();
        for q in &queries {
            let count = run_query(&dataset, q).await;
            results.push((q.clone(), count));
        }
        let elapsed = t0.elapsed();
        println!(
            "{} queries: total {:.2}ms, avg {:.3}ms",
            queries.len(),
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() * 1000.0 / queries.len() as f64
        );
        for (q, count) in results.iter().take(10) {
            let display = if q.len() > 70 {
                format!("{}...", &q[..67])
            } else {
                q.clone()
            };
            println!("  \"{}\" -> {} matches", display, count);
        }
    }

    // ── N-Gram ──
    println!("\n============================================================");
    println!("=== N-Gram ===");
    // Create a fresh dataset for N-Gram
    let mut batches2: Vec<RecordBatch> = Vec::new();
    for chunk in texts.chunks(batch_size) {
        let col = Arc::new(LargeStringArray::from(
            chunk.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ));
        batches2.push(RecordBatch::try_new(schema.clone(), vec![col]).unwrap());
    }
    let reader2 = RecordBatchIterator::new(batches2.into_iter().map(Ok), schema.clone());
    let mut dataset2 = Dataset::write(reader2, &ngram_path, None).await.unwrap();

    let t0 = Instant::now();
    let params2 = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::NGram);
    dataset2
        .create_index(&["content"], IndexType::NGram, None, &params2, true)
        .await
        .unwrap();
    println!("Index build: {:.2}s", t0.elapsed().as_secs_f64());

    let dataset2 = Dataset::open(&ngram_path).await.unwrap();

    for (label, queries) in [
        (
            "Short",
            short_queries
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        ),
        (
            "Medium",
            medium_queries.iter().map(|s| s.to_string()).collect(),
        ),
        ("Long", long_queries.clone()),
    ] {
        println!("\n--- {label} queries ---");
        let t0 = Instant::now();
        let mut results = Vec::new();
        for q in &queries {
            let count = run_query(&dataset2, q).await;
            results.push((q.clone(), count));
        }
        let elapsed = t0.elapsed();
        println!(
            "{} queries: total {:.2}ms, avg {:.3}ms",
            queries.len(),
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() * 1000.0 / queries.len() as f64
        );
        for (q, count) in results.iter().take(10) {
            let display = if q.len() > 70 {
                format!("{}...", &q[..67])
            } else {
                q.clone()
            };
            println!("  \"{}\" -> {} matches", display, count);
        }
    }

    // Print directory structure and sizes
    println!("\n============================================================");
    println!("=== Storage Layout ===");
    for (label, path) in [("FM-Index", &fm_path), ("N-Gram", &ngram_path)] {
        println!("\n--- {label}: {path} ---");
        let mut total = 0u64;
        for entry in walkdir::WalkDir::new(path)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let meta = entry.metadata().unwrap();
            if meta.is_file() {
                let size = meta.len();
                total += size;
                let rel = entry.path().strip_prefix(path).unwrap_or(entry.path());
                if size > 1024 * 1024 {
                    println!("  {}: {:.1} MB", rel.display(), size as f64 / 1e6);
                } else if size > 1024 {
                    println!("  {}: {:.1} KB", rel.display(), size as f64 / 1e3);
                } else {
                    println!("  {}: {} B", rel.display(), size);
                }
            }
        }
        println!("  TOTAL: {:.1} MB", total as f64 / 1e6);
    }
}
