// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
#![allow(clippy::print_stdout)]

//! Example: Configuring tracing to observe Lance operations.
//!
//! Lance emits structured tracing events at various levels. This example
//! shows how to set up a tracing subscriber with env-filter to capture
//! Lance telemetry output.
//!
//! Run with different filter levels:
//! ```bash
//! # Show compaction and IO retry events
//! RUST_LOG="lance::compaction=info,lance_io::retry=debug" cargo run --example tracing_setup
//!
//! # Show everything including buffer operations
//! RUST_LOG="debug" cargo run --example tracing_setup
//!
//! # Show only warnings (retries, connection resets)
//! RUST_LOG="lance=warn,lance_io=warn" cargo run --example tracing_setup
//! ```

use std::sync::Arc;

use arrow::array::{Float32Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use lance::Dataset;
use lance::dataset::WriteParams;
use lance_core::utils::tempfile::TempStrDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing with env-filter.
    // Set RUST_LOG to control which Lance targets are visible.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "lance=info,lance_io=info,lance_index=info".parse().unwrap()),
        )
        .with_target(true)
        .init();

    let tmp_dir = TempStrDir::default();
    let uri: &str = &tmp_dir;

    // Create a small dataset
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Float32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..1000)),
            Arc::new(Float32Array::from_iter_values((0..1000).map(|i| i as f32))),
        ],
    )?;

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let mut ds = Dataset::write(reader, uri, Some(WriteParams::default()))
        .await
        .expect("Failed to write dataset");

    // Fragment distribution stats (in-memory, no IO)
    let frag_stats = ds.fragment_distribution_stats();
    println!(
        "Fragments: count={}, total_rows={}, avg_rows={:.1}",
        frag_stats.count, frag_stats.total_physical_rows, frag_stats.avg_physical_rows,
    );

    // Compact to trigger compaction tracing events
    let metrics =
        lance::dataset::optimize::compact_files(&mut ds, Default::default(), None).await?;
    println!(
        "Compaction: removed={}, added={}, bytes_rewritten={}, elapsed_ms_sum={}",
        metrics.fragments_removed,
        metrics.fragments_added,
        metrics.bytes_rewritten,
        metrics.elapsed_ms_sum,
    );

    Ok(())
}
