// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! End-to-end FM-Index vs N-Gram benchmark using real source code files.
//!
//! Reads Rust source files from the lance repo, stores full content in a Lance table,
//! builds both FM-Index and N-Gram indices, then benchmarks substring queries.
//!
//! Usage: cargo run --release -p lance-index --example fmindex_e2e

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
use walkdir::WalkDir;

fn load_source_files(root: &str) -> Vec<(String, String)> {
    let mut files = Vec::new();
    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.extension().map_or(false, |e| {
            e == "rs" || e == "py" || e == "toml" || e == "md"
        }) {
            if let Ok(content) = std::fs::read_to_string(path) {
                if !content.is_empty() {
                    let rel = path.strip_prefix(root).unwrap_or(path);
                    files.push((rel.to_string_lossy().to_string(), content));
                }
            }
        }
    }
    files
}

#[tokio::main]
async fn main() {
    let root = std::env::var("SOURCE_ROOT").unwrap_or_else(|_| ".".to_string());
    let max_files: usize = std::env::var("MAX_FILES")
        .unwrap_or_else(|_| "5000".to_string())
        .parse()
        .unwrap();

    println!("Loading source files from {root}...");
    let mut files = load_source_files(&root);
    files.truncate(max_files);
    let total = files.len();
    let total_bytes: usize = files.iter().map(|(_, c)| c.len()).sum();
    let avg_len = total_bytes / total.max(1);
    println!(
        "Loaded {total} files, {:.1} MB total, avg {avg_len} bytes/file",
        total_bytes as f64 / 1e6
    );

    // Keep paths for result display
    let paths: Vec<String> = files.iter().map(|(p, _)| p.clone()).collect();
    let contents: Vec<String> = files.iter().map(|(_, c)| c.clone()).collect();

    // Build Arrow batch with full file contents (no truncation!)
    let row_id_col = Arc::new(UInt64Array::from(
        (0..total).map(|i| i as u64).collect_vec(),
    ));
    let doc_col = Arc::new(LargeStringArray::from(
        contents.iter().map(|f| f.as_str()).collect_vec(),
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

    let batch_size = 500.min(total);
    let num_batches = total / batch_size;
    let batches: Vec<RecordBatch> = (0..num_batches)
        .map(|i| batch.slice(i * batch_size, batch_size))
        .collect_vec();

    // Generate queries from actual file contents
    let queries_short: Vec<String> = [
        "fn ", "use ", "self.", "impl ", "pub ", "async ", "Result", "let mut ", "struct ",
        "#[test]",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let queries_medium: Vec<String> = [
        "fn main()",
        "use std::sync::Arc",
        "impl Default for",
        "async fn search",
        "#[derive(Debug)]",
        "pub struct FMIndex",
        "Result<Self>",
        "fn build_suffix_array",
        "cargo test --workspace",
        "pub fn deep_size",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    // Long queries: grab actual substrings from files
    let mut queries_long: Vec<String> = Vec::new();
    for c in contents.iter().take(200) {
        let chars: Vec<char> = c.chars().collect();
        if chars.len() > 100 {
            let start = chars.len() / 4;
            let end = (start + 80).min(chars.len());
            let q: String = chars[start..end].iter().collect();
            // Skip queries with newlines for clean output
            if !q.contains('\n') && !q.contains('\r') {
                queries_long.push(q);
            }
        }
    }
    queries_long.truncate(50);

    println!(
        "\nQueries: {} short, {} medium, {} long",
        queries_short.len(),
        queries_medium.len(),
        queries_long.len()
    );

    // ── FM-Index ──
    println!("\n============================================================");
    println!("=== FM-Index ===");
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
    println!("Build time: {:.2}s", t0.elapsed().as_secs_f64());

    if let Some(ref idx_files) = created.files {
        let total_size: u64 = idx_files.iter().map(|f| f.size_bytes as u64).sum();
        println!(
            "Total index size: {:.1} MB ({} files)",
            total_size as f64 / 1e6,
            idx_files.len()
        );
        for f in idx_files.iter().take(3) {
            println!("  {}: {:.1} MB", f.path, f.size_bytes as f64 / 1e6);
        }
        if idx_files.len() > 3 {
            println!("  ... and {} more", idx_files.len() - 3);
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
    println!(
        "Index load time: {:.2}ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // Helper: extract row IDs from SearchResult and show matching files with context
    let show_matches = |query: &str,
                        r: &lance_index::scalar::SearchResult,
                        paths: &[String],
                        contents: &[String],
                        max_show: usize| {
        let (tree_map, is_exact) = match r {
            lance_index::scalar::SearchResult::Exact(s) => (s.true_rows(), true),
            lance_index::scalar::SearchResult::AtMost(s) => (s.true_rows(), false),
            _ => (lance_select::RowAddrTreeMap::new(), false),
        };
        let row_ids: Vec<u64> = tree_map
            .row_addrs()
            .map(|it| it.map(|addr| addr.into()).collect::<Vec<u64>>())
            .unwrap_or_default();
        let count = row_ids.len();
        let label = if is_exact { "exact" } else { "approx" };
        let display_q = if query.len() > 70 {
            format!("{}...", &query[..67])
        } else {
            query.to_string()
        };
        println!("  \"{}\" -> {} matches ({})", display_q, count, label);
        for &rid in row_ids.iter().take(max_show) {
            let idx = rid as usize;
            if idx < paths.len() {
                let path = &paths[idx];
                let content = &contents[idx];
                if let Some(pos) = content.find(query) {
                    let start = pos.saturating_sub(20);
                    let end = (pos + query.len() + 20).min(content.len());
                    let snippet: String = content[start..end]
                        .chars()
                        .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
                        .collect();
                    println!("    [{idx}] {path}");
                    println!("         ...{snippet}...");
                } else {
                    println!("    [{idx}] {path} (false positive)");
                }
            }
        }
        if count > max_show {
            println!("    ... and {} more", count - max_show);
        }
    };

    for (label, queries) in [
        ("Short", &queries_short),
        ("Medium", &queries_medium),
        ("Long", &queries_long),
    ] {
        println!("\n--- {label} queries ---");
        let t0 = Instant::now();
        let mut search_results = Vec::new();
        for q in queries.iter() {
            let r = index
                .search(&TextQuery::StringContains(q.clone()), &NoOpMetricsCollector)
                .await
                .unwrap();
            search_results.push((q.clone(), r));
        }
        let elapsed = t0.elapsed();
        println!(
            "{} queries: total {:.2}ms, avg {:.3}ms",
            queries.len(),
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() * 1000.0 / queries.len() as f64
        );
        for (q, r) in &search_results {
            show_matches(q, r, &paths, &contents, 3);
        }
    }

    drop(index);
    drop(store);

    // ── N-Gram ──
    println!("\n============================================================");
    println!("=== N-Gram ===");
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
    println!("Build time: {:.2}s", t0.elapsed().as_secs_f64());

    let idx_files = store2.list_files_with_sizes().await.unwrap();
    let total_size: u64 = idx_files.iter().map(|f| f.size_bytes as u64).sum();
    println!(
        "Total index size: {:.1} MB ({} files)",
        total_size as f64 / 1e6,
        idx_files.len()
    );

    let details =
        prost_types::Any::from_msg(&lance_index::pbold::NGramIndexDetails::default()).unwrap();
    let t0 = Instant::now();
    let ngram_index = NGramIndexPlugin
        .load_index(store2, &details, None, &LanceCache::no_cache())
        .await
        .unwrap();
    println!(
        "Index load time: {:.2}ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    for (label, queries) in [
        ("Short", &queries_short),
        ("Medium", &queries_medium),
        ("Long", &queries_long),
    ] {
        println!("\n--- {label} queries ---");
        let t0 = Instant::now();
        let mut search_results = Vec::new();
        for q in queries.iter() {
            let r = ngram_index
                .search(&TextQuery::StringContains(q.clone()), &NoOpMetricsCollector)
                .await
                .unwrap();
            search_results.push((q.clone(), r));
        }
        let elapsed = t0.elapsed();
        println!(
            "{} queries: total {:.2}ms, avg {:.3}ms",
            queries.len(),
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() * 1000.0 / queries.len() as f64
        );
        for (q, r) in &search_results {
            show_matches(q, r, &paths, &contents, 3);
        }
    }
}
