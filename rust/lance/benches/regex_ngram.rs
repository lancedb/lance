// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark: regex predicate scans over an ngram-indexed string column.
//!
//! Each query is a `regexp_match(doc, '...')` filter against a dataset that has
//! an NGram index on `doc`. The benchmark builds both the default fixed-trigram
//! index and the experimental sparse n-gram index so the same workload can
//! compare query-time pruning cost directly.

use std::hint::black_box;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use arrow::array::AsArray;
use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use criterion::{Criterion, criterion_group, criterion_main};
use futures::TryStreamExt;
use lance::Dataset;
use lance::index::DatasetIndexExt;
use lance_core::utils::tempfile::TempStrDir;
use lance_datagen::{RowCount, array};
use lance_index::IndexType;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
#[cfg(target_os = "linux")]
use lance_testing::pprof::{Output, PProfProfiler};

const TOTAL: usize = 200_000;
const SPARSE_DECOY_LITERAL: &str = "sparsemarkerabcdefghijklmnopqrstuvwx";

static SPARSE_DECOY_TRIGRAMS: LazyLock<String> = LazyLock::new(|| {
    let mut trigrams = String::new();
    for trigram in SPARSE_DECOY_LITERAL.as_bytes().windows(3) {
        if !trigrams.is_empty() {
            trigrams.push(' ');
        }
        trigrams.push_str(std::str::from_utf8(trigram).unwrap());
    }
    trigrams
});

/// Build the `doc` column: random sentences with rare markers injected into a
/// small fraction of rows so the regex queries have controlled selectivity.
/// The markers are unlikely to appear in the generated English-word sentences.
fn build_docs() -> StringArray {
    let mut sentence_gen = array::random_sentence(1, 30, false);
    let base = sentence_gen
        .generate_default(RowCount::from(TOTAL as u64))
        .unwrap();
    let base = base.as_string::<i32>();
    let docs = (0..TOTAL).map(|i| {
        let sentence = base.value(i);
        let mut doc = if i % 200 == 0 {
            // ~0.5% of rows match `zqxwvu.*needlexyz` and `zqxwvu`.
            format!("{sentence} zqxwvu needlexyz")
        } else if i % 211 == 0 {
            // A second marker for the alternation query.
            format!("{sentence} qwerasdf")
        } else {
            sentence.to_string()
        };
        // Every row contains all trigrams from SPARSE_DECOY_LITERAL, but only a
        // small fraction contains the full literal. This creates a workload where
        // fixed trigrams produce a broad candidate set while sparse longer n-grams
        // can stay selective.
        doc.push(' ');
        doc.push_str(&SPARSE_DECOY_TRIGRAMS);
        if i % 997 == 0 {
            doc.push(' ');
            doc.push_str(SPARSE_DECOY_LITERAL);
        }
        doc
    });
    StringArray::from_iter_values(docs)
}

#[derive(Clone, Copy)]
enum BenchTokenization {
    Trigram,
    Sparse,
}

impl BenchTokenization {
    fn name(self) -> &'static str {
        match self {
            Self::Trigram => "trigram",
            Self::Sparse => "sparse",
        }
    }

    fn params(self) -> ScalarIndexParams {
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::NGram);
        match self {
            Self::Trigram => params,
            Self::Sparse => params.with_params(&serde_json::json!({
                "tokenization": "sparse",
            })),
        }
    }
}

async fn build_dataset(tempdir: &TempStrDir, tokenization: BenchTokenization) -> Arc<Dataset> {
    let schema = Arc::new(Schema::new(vec![Field::new("doc", DataType::Utf8, false)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(build_docs())]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);

    let mut dataset = Dataset::write(reader, tempdir.as_str(), None)
        .await
        .unwrap();
    dataset
        .create_index(
            &["doc"],
            IndexType::NGram,
            None,
            &tokenization.params(),
            true,
        )
        .await
        .unwrap();
    Arc::new(dataset)
}

async fn scan_filter(dataset: &Dataset, filter: &str) -> usize {
    let mut scanner = dataset.scan();
    scanner.filter(filter).unwrap();
    let stream = scanner.try_into_stream().await.unwrap();
    let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
    batches.iter().map(|b| b.num_rows()).sum()
}

fn bench_regex_ngram(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let trigram_tempdir = TempStrDir::default();
    let sparse_tempdir = TempStrDir::default();
    let trigram_dataset = rt.block_on(build_dataset(&trigram_tempdir, BenchTokenization::Trigram));
    let sparse_dataset = rt.block_on(build_dataset(&sparse_tempdir, BenchTokenization::Sparse));

    let queries = [
        ("selective_and", "regexp_match(doc, 'zqxwvu.*needlexyz')"),
        (
            "alternation",
            "regexp_match(doc, '(zqxwvu|qwerasdf|needlexyz)')",
        ),
        ("plain_literal", "regexp_match(doc, 'zqxwvu')"),
        (
            "sparse_decoy_literal",
            "regexp_match(doc, 'sparsemarkerabcdefghijklmnopqrstuvwx')",
        ),
        ("non_accelerable_a_dot_b", "regexp_match(doc, 'a.b')"),
    ];

    let mut group = c.benchmark_group("regex_ngram");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));
    for (tokenization, dataset) in [
        (BenchTokenization::Trigram, trigram_dataset.as_ref()),
        (BenchTokenization::Sparse, sparse_dataset.as_ref()),
    ] {
        for (name, filter) in queries {
            group.bench_function(format!("{}/{name}", tokenization.name()), |b| {
                b.iter(|| black_box(rt.block_on(scan_filter(dataset, filter))));
            });
        }
    }
    group.finish();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name = benches;
    config = Criterion::default()
        .significance_level(0.1)
        .sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_regex_ngram);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_regex_ngram);

criterion_main!(benches);
