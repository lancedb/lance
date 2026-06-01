// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One-shot FM-Index + N-Gram benchmark for large datasets.
//! Usage: FMINDEX_DATA_PATH=/path/to/data.txt cargo run --release --example fmindex_oneshot

use std::sync::Arc;
use std::time::Instant;

use arrow_array::{LargeStringArray, RecordBatch, UInt64Array};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::stream;
use itertools::Itertools;
use lance_core::ROW_ID;
use lance_core::cache::LanceCache;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::progress::NoopIndexBuildProgress;
use lance_index::scalar::fmindex::FMIndexPlugin;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::ngram::{NGramIndexBuilder, NGramIndexBuilderOptions, NGramIndexPlugin};
use lance_index::scalar::registry::{ScalarIndexPlugin, VALUE_COLUMN_NAME};
use lance_index::scalar::{IndexStore, TextQuery};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;

#[tokio::main]
async fn main() {
    let path = std::env::var("FMINDEX_DATA_PATH").expect("Set FMINDEX_DATA_PATH");
    let text = std::fs::read_to_string(&path).expect("Cannot read file");
    let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
    let total = lines.len();
    println!("Loaded {} lines from {}", total, path);

    let row_id_col = Arc::new(UInt64Array::from(
        (0..total).map(|i| i as u64).collect_vec(),
    ));
    let doc_col = Arc::new(LargeStringArray::from(
        lines.iter().map(|l| l.to_string()).collect_vec(),
    ));
    let batch = RecordBatch::try_new(
        arrow_schema::Schema::new(vec![
            arrow_schema::Field::new(VALUE_COLUMN_NAME, arrow_schema::DataType::LargeUtf8, false),
            arrow_schema::Field::new(ROW_ID, arrow_schema::DataType::UInt64, false),
        ])
        .into(),
        vec![doc_col, row_id_col],
    )
    .unwrap();

    let batch_size = 1000.min(total);
    let num_batches = total / batch_size;
    let batches: Vec<RecordBatch> = (0..num_batches)
        .map(|i| batch.slice(i * batch_size, batch_size))
        .collect_vec();

    // Generate queries
    let mut short_queries = Vec::new();
    let mut long_queries = Vec::new();
    let mut very_long_queries = Vec::new();
    for i in (0..total).step_by(total / 200) {
        let s = lines[i];
        let chars: Vec<char> = s.chars().collect();
        if chars.len() > 10 {
            let start = chars.len() / 4;
            let end = (start + 8).min(chars.len());
            short_queries.push(chars[start..end].iter().collect::<String>());
        }
        if chars.len() > 50 {
            let start = chars.len() / 4;
            let end = (start + 50).min(chars.len());
            long_queries.push(chars[start..end].iter().collect::<String>());
        }
        if chars.len() > 120 {
            let start = chars.len() / 4;
            let end = (start + 100).min(chars.len());
            very_long_queries.push(chars[start..end].iter().collect::<String>());
        }
    }
    let short_queries: Vec<String> = short_queries.into_iter().take(100).collect();
    let long_queries: Vec<String> = long_queries.into_iter().take(100).collect();
    let very_long_queries: Vec<String> = very_long_queries.into_iter().take(100).collect();
    println!(
        "Prepared {} short, {} long, {} very_long queries",
        short_queries.len(),
        long_queries.len(),
        very_long_queries.len()
    );

    // ── FM-Index ──
    println!("\n=== FM-Index ===");
    let tempdir = tempfile::tempdir().unwrap();
    let index_dir = Path::from_filesystem_path(tempdir.path()).unwrap();
    let store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        index_dir,
        Arc::new(LanceCache::no_cache()),
    ));

    let t0 = Instant::now();
    let stream = RecordBatchStreamAdapter::new(
        batch.schema(),
        stream::iter(batches.clone().into_iter().map(Ok)),
    );
    let req = FMIndexPlugin
        .new_training_request("", batch.schema().field(0))
        .unwrap();
    let created = FMIndexPlugin
        .train_index(
            Box::pin(stream),
            store.as_ref(),
            req,
            None,
            Arc::new(NoopIndexBuildProgress),
        )
        .await
        .unwrap();
    let build_time = t0.elapsed();
    println!("Build time: {:.2}s", build_time.as_secs_f64());

    if let Some(ref files) = created.files {
        for file in files {
            println!(
                "  FILE: {} ({} bytes, {:.1} MB)",
                file.path,
                file.size_bytes,
                file.size_bytes as f64 / 1e6
            );
        }
    }

    let t0 = Instant::now();
    let index = FMIndexPlugin
        .load_index(
            store.clone(),
            &created.index_details,
            None,
            &LanceCache::no_cache(),
        )
        .await
        .unwrap();
    let load_time = t0.elapsed();
    println!("Index load time: {:.2}ms", load_time.as_secs_f64() * 1000.0);

    // Short queries
    let t0 = Instant::now();
    let mut short_counts: Vec<String> = Vec::new();
    for q in &short_queries {
        let r = index
            .search(&TextQuery::StringContains(q.clone()), &NoOpMetricsCollector)
            .await
            .unwrap();
        let count = match &r {
            lance_index::scalar::SearchResult::Exact(s) => {
                s.len().map_or("?".to_string(), |n| n.to_string())
            }
            _ => "?".to_string(),
        };
        short_counts.push(count);
    }
    let elapsed = t0.elapsed();
    println!(
        "Short queries ({} queries): total {:.2}ms, avg {:.3}ms",
        short_queries.len(),
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / short_queries.len() as f64
    );
    for i in 0..5.min(short_queries.len()) {
        println!(
            "  q[{i}]: \"{}\" -> {} matches",
            &short_queries[i], short_counts[i]
        );
    }

    // Long queries
    let t0 = Instant::now();
    let mut long_counts: Vec<String> = Vec::new();
    for q in &long_queries {
        let r = index
            .search(&TextQuery::StringContains(q.clone()), &NoOpMetricsCollector)
            .await
            .unwrap();
        let count = match &r {
            lance_index::scalar::SearchResult::Exact(s) => {
                s.len().map_or("?".to_string(), |n| n.to_string())
            }
            _ => "?".to_string(),
        };
        long_counts.push(count);
    }
    let elapsed = t0.elapsed();
    println!(
        "Long queries ({} queries): total {:.2}ms, avg {:.3}ms",
        long_queries.len(),
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / long_queries.len() as f64
    );
    for i in 0..5.min(long_queries.len()) {
        println!(
            "  q[{i}]: \"{}\" -> {} matches",
            &long_queries[i], long_counts[i]
        );
    }

    // Very long queries (100 chars)
    let t0 = Instant::now();
    let mut vlong_counts: Vec<String> = Vec::new();
    for q in &very_long_queries {
        let r = index
            .search(&TextQuery::StringContains(q.clone()), &NoOpMetricsCollector)
            .await
            .unwrap();
        let count = match &r {
            lance_index::scalar::SearchResult::Exact(s) => {
                s.len().map_or("?".to_string(), |n| n.to_string())
            }
            _ => "?".to_string(),
        };
        vlong_counts.push(count);
    }
    let elapsed = t0.elapsed();
    println!(
        "Very long queries ({} queries, ~100 chars): total {:.2}ms, avg {:.3}ms",
        very_long_queries.len(),
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / very_long_queries.len() as f64
    );
    for i in 0..5.min(very_long_queries.len()) {
        let preview = if very_long_queries[i].len() > 40 {
            &very_long_queries[i][..40]
        } else {
            &very_long_queries[i]
        };
        println!("  q[{i}]: \"{preview}...\" -> {} matches", vlong_counts[i]);
    }

    drop(index);
    drop(store);

    // ── N-Gram ──
    println!("\n=== N-Gram ===");
    let tempdir2 = tempfile::tempdir().unwrap();
    let index_dir2 = Path::from_filesystem_path(tempdir2.path()).unwrap();
    let store2 = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        index_dir2,
        Arc::new(LanceCache::no_cache()),
    ));

    let t0 = Instant::now();
    let stream = RecordBatchStreamAdapter::new(
        batch.schema(),
        stream::iter(batches.clone().into_iter().map(Ok)),
    );
    let mut builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions::default()).unwrap();
    let num_spill_files = builder.train(Box::pin(stream)).await.unwrap();
    builder
        .write_index(store2.as_ref(), num_spill_files, None)
        .await
        .unwrap();
    let build_time = t0.elapsed();
    println!("Build time: {:.2}s", build_time.as_secs_f64());

    let files = store2.list_files_with_sizes().await.unwrap();
    for file in &files {
        println!(
            "  FILE: {} ({} bytes, {:.1} MB)",
            file.path,
            file.size_bytes,
            file.size_bytes as f64 / 1e6
        );
    }

    let details =
        prost_types::Any::from_msg(&lance_index::pbold::NGramIndexDetails::default()).unwrap();
    let t0 = Instant::now();
    let ngram_index = NGramIndexPlugin
        .load_index(store2, &details, None, &LanceCache::no_cache())
        .await
        .unwrap();
    let load_time = t0.elapsed();
    println!("Index load time: {:.2}ms", load_time.as_secs_f64() * 1000.0);

    let t0 = Instant::now();
    let mut ng_short_counts: Vec<String> = Vec::new();
    for q in &short_queries {
        let r = ngram_index
            .search(&TextQuery::StringContains(q.clone()), &NoOpMetricsCollector)
            .await
            .unwrap();
        let count = match &r {
            lance_index::scalar::SearchResult::Exact(s) => {
                s.len().map_or("?".to_string(), |n| n.to_string())
            }
            lance_index::scalar::SearchResult::AtMost(s) => {
                format!("≤{}", s.len().map_or("?".to_string(), |n| n.to_string()))
            }
            _ => "?".to_string(),
        };
        ng_short_counts.push(count);
    }
    let elapsed = t0.elapsed();
    println!(
        "Short queries ({} queries): total {:.2}ms, avg {:.3}ms",
        short_queries.len(),
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / short_queries.len() as f64
    );
    for i in 0..5.min(short_queries.len()) {
        println!(
            "  q[{i}]: \"{}\" -> {} matches",
            &short_queries[i], ng_short_counts[i]
        );
    }

    let t0 = Instant::now();
    let mut ng_long_counts: Vec<String> = Vec::new();
    for q in &long_queries {
        let r = ngram_index
            .search(&TextQuery::StringContains(q.clone()), &NoOpMetricsCollector)
            .await
            .unwrap();
        let count = match &r {
            lance_index::scalar::SearchResult::Exact(s) => {
                s.len().map_or("?".to_string(), |n| n.to_string())
            }
            lance_index::scalar::SearchResult::AtMost(s) => {
                format!("≤{}", s.len().map_or("?".to_string(), |n| n.to_string()))
            }
            _ => "?".to_string(),
        };
        ng_long_counts.push(count);
    }
    let elapsed = t0.elapsed();
    println!(
        "Long queries ({} queries): total {:.2}ms, avg {:.3}ms",
        long_queries.len(),
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / long_queries.len() as f64
    );
    for i in 0..5.min(long_queries.len()) {
        println!(
            "  q[{i}]: \"{}\" -> {} matches",
            &long_queries[i], ng_long_counts[i]
        );
    }

    let t0 = Instant::now();
    let mut ng_vlong_counts: Vec<String> = Vec::new();
    for q in &very_long_queries {
        let r = ngram_index
            .search(&TextQuery::StringContains(q.clone()), &NoOpMetricsCollector)
            .await
            .unwrap();
        let count = match &r {
            lance_index::scalar::SearchResult::Exact(s) => {
                s.len().map_or("?".to_string(), |n| n.to_string())
            }
            lance_index::scalar::SearchResult::AtMost(s) => {
                format!("≤{}", s.len().map_or("?".to_string(), |n| n.to_string()))
            }
            _ => "?".to_string(),
        };
        ng_vlong_counts.push(count);
    }
    let elapsed = t0.elapsed();
    println!(
        "Very long queries ({} queries, ~100 chars): total {:.2}ms, avg {:.3}ms",
        very_long_queries.len(),
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / very_long_queries.len() as f64
    );
    for i in 0..5.min(very_long_queries.len()) {
        let preview = if very_long_queries[i].len() > 40 {
            &very_long_queries[i][..40]
        } else {
            &very_long_queries[i]
        };
        println!(
            "  q[{i}]: \"{preview}...\" -> {} matches",
            ng_vlong_counts[i]
        );
    }
}
