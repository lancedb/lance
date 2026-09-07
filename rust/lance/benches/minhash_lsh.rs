// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! End-to-end benchmark of MinHash LSH similarity search: query latency by
//! `k`, by index parameters, by segment count, and with unindexed rows that
//! take the flat path. The corpus is synthetic near-duplicate text generated
//! from a seed, so every query is a perturbed copy of an indexed document.

use std::sync::{Arc, OnceLock};

use arrow_array::{ArrayRef, Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lance::Dataset;
use lance::dataset::scanner::Scanner;
use lance::dataset::{WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance_index::IndexType;
use lance_index::scalar::minhash_lsh::MinHashQuery;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
#[cfg(target_os = "linux")]
use lance_testing::pprof::{Output, PProfProfiler};
use tempfile::TempDir;

const INDEX_NAME: &str = "text_minhash";
const ROWS_PER_FRAGMENT: usize = 50_000;
const INDEXED_FRAGMENT_COUNT: usize = 8;
const UNINDEXED_FRAGMENT_COUNT: usize = 1;
const VOCAB: usize = 50_000;
/// Shared header/footer phrases a third of the texts carry.
const BOILERPLATE_PHRASES: usize = 200;
/// Fraction of texts that are near duplicates of an earlier one.
const DUP_RATE: f64 = 0.2;
const CORPUS_SEED: u64 = 7;
const QUERY_COUNT: usize = 64;

struct Lcg(u64);

impl Lcg {
    fn for_doc(salt: u64, doc: u64) -> Self {
        let mut rng = Self((CORPUS_SEED + salt) ^ doc.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        rng.next();
        rng
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }

    fn unit(&mut self) -> f64 {
        (self.next() % (1 << 24)) as f64 / (1u64 << 24) as f64
    }
}

/// A word whose rank follows Zipf's law (frequency ∝ 1/rank), as in natural
/// text: the most common words appear in most documents.
fn word(rng: &mut Lcg) -> String {
    format!("w{}", (VOCAB as f64).powf(rng.unit()) as usize)
}

/// Text length in words: mostly snippets and paragraphs, a tail of articles.
fn text_len(rng: &mut Lcg) -> usize {
    match rng.below(10) {
        0..=2 => 6 + rng.below(25),
        3..=8 => 30 + rng.below(120),
        _ => 150 + rng.below(450),
    }
}

fn boilerplate(phrase: usize) -> Vec<String> {
    let mut rng = Lcg::for_doc(2, phrase as u64);
    let len = 5 + rng.below(8);
    (0..len).map(|_| word(&mut rng)).collect()
}

/// The original text of cluster `root`.
fn base_text(root: u64) -> Vec<String> {
    let mut rng = Lcg::for_doc(0, root);
    let mut words: Vec<String> = (0..text_len(&mut rng)).map(|_| word(&mut rng)).collect();
    if rng.below(3) == 0 {
        let phrase = boilerplate(rng.below(BOILERPLATE_PHRASES));
        if rng.below(2) == 0 {
            words.splice(0..0, phrase);
        } else {
            words.extend(phrase);
        }
    }
    words
}

/// A near duplicate of `source`: an exact copy a quarter of the time,
/// otherwise 2–30% of the words replaced, sometimes cut short or extended.
fn near_duplicate(rng: &mut Lcg, source: &[String]) -> Vec<String> {
    let rate = if rng.below(4) == 0 {
        0.0
    } else {
        0.02 + 0.28 * rng.unit()
    };
    let mut words: Vec<String> = source
        .iter()
        .map(|w| {
            if rng.unit() < rate {
                word(rng)
            } else {
                w.clone()
            }
        })
        .collect();
    match rng.below(10) {
        0..=1 => {
            let keep = words.len() * (60 + rng.below(30)) / 100;
            words.truncate(keep.max(1));
        }
        2..=3 => {
            let extra = words.len() * (10 + rng.below(30)) / 100;
            words.extend((0..extra).map(|_| word(rng)));
        }
        _ => {}
    }
    words
}

/// Which earlier text document `doc` duplicates, if any. Sources skew
/// towards early documents, so cluster sizes are heavy-tailed: most texts
/// are unique, some have a few copies, a handful have hundreds.
fn duplicate_source(doc: u64) -> Option<u64> {
    let mut rng = Lcg::for_doc(1, doc);
    (doc > 0 && rng.unit() < DUP_RATE).then(|| (doc as f64 * rng.unit().powi(3)) as u64)
}

/// The document whose original text `doc`'s cluster derives from.
fn cluster_root(mut doc: u64) -> u64 {
    while let Some(source) = duplicate_source(doc) {
        doc = source;
    }
    doc
}

fn corpus_text(doc: u64) -> String {
    let words = match duplicate_source(doc) {
        Some(source) => {
            let mut rng = Lcg::for_doc(1, doc);
            rng.next();
            rng.next();
            near_duplicate(&mut rng, &base_text(cluster_root(source)))
        }
        None => base_text(doc),
    };
    words.join(" ")
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("text", DataType::Utf8, true),
    ]))
}

fn fragment_batch(fragment: usize) -> RecordBatch {
    let start = (fragment * ROWS_PER_FRAGMENT) as i64;
    let ids = Int64Array::from_iter_values(start..start + ROWS_PER_FRAGMENT as i64);
    let texts = StringArray::from_iter_values(
        (start..start + ROWS_PER_FRAGMENT as i64).map(|doc| corpus_text(doc as u64)),
    );
    RecordBatch::try_new(
        schema(),
        vec![Arc::new(ids) as ArrayRef, Arc::new(texts) as ArrayRef],
    )
    .unwrap()
}

/// Queries: near duplicates of indexed texts, with the corpus' spread of
/// similarity from exact copies to heavy edits.
fn queries() -> Vec<String> {
    let mut rng = Lcg(99);
    (0..QUERY_COUNT)
        .map(|_| {
            let doc = rng.below(INDEXED_FRAGMENT_COUNT * ROWS_PER_FRAGMENT) as u64;
            let words: Vec<String> = corpus_text(doc).split(' ').map(String::from).collect();
            near_duplicate(&mut rng, &words).join(" ")
        })
        .collect()
}

struct BenchDataset {
    _tmpdir: TempDir,
    dataset: Dataset,
}

fn index_params(num_hashes: u32, num_bands: u32) -> ScalarIndexParams {
    ScalarIndexParams::for_builtin(BuiltinIndexType::MinHashLsh)
        .with_params(&serde_json::json!({ "num_hashes": num_hashes, "num_bands": num_bands }))
}

/// Write the indexed fragments and build the index as `segment_count`
/// segments; append `unindexed_fragments` more fragments afterwards.
async fn build_dataset(
    params: &ScalarIndexParams,
    segment_count: usize,
    unindexed_fragments: usize,
) -> BenchDataset {
    let tmpdir = TempDir::new().unwrap();
    let uri = format!("file://{}", tmpdir.path().display());
    let batches = RecordBatchIterator::new(
        (0..INDEXED_FRAGMENT_COUNT).map(|fragment| Ok(fragment_batch(fragment))),
        schema(),
    );
    let mut dataset = Dataset::write(
        batches,
        &uri,
        Some(WriteParams {
            max_rows_per_file: ROWS_PER_FRAGMENT,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), INDEXED_FRAGMENT_COUNT);

    let fragments_per_segment = INDEXED_FRAGMENT_COUNT / segment_count;
    let mut segments = Vec::with_capacity(segment_count);
    for segment in 0..segment_count {
        let start = (segment * fragments_per_segment) as u32;
        let fragment_ids: Vec<u32> = (start..start + fragments_per_segment as u32).collect();
        let staged = dataset
            .create_index_builder(&["text"], IndexType::MinHashLsh, params)
            .name(INDEX_NAME.to_string())
            .fragments(fragment_ids)
            .execute_uncommitted()
            .await
            .unwrap();
        segments.push(staged);
    }
    dataset
        .commit_existing_index_segments(INDEX_NAME, "text", segments)
        .await
        .unwrap();

    if unindexed_fragments > 0 {
        let batches = RecordBatchIterator::new(
            (INDEXED_FRAGMENT_COUNT..INDEXED_FRAGMENT_COUNT + unindexed_fragments)
                .map(|fragment| Ok(fragment_batch(fragment))),
            schema(),
        );
        dataset = Dataset::write(
            batches,
            &uri,
            Some(WriteParams {
                max_rows_per_file: ROWS_PER_FRAGMENT,
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    }
    dataset.prewarm_index(INDEX_NAME).await.unwrap();

    BenchDataset {
        _tmpdir: tmpdir,
        dataset,
    }
}

async fn search(scanner: &mut Scanner, text: &str, k: usize) -> usize {
    let batch = scanner
        .minhash_search(MinHashQuery::new(text, "text"))
        .unwrap()
        .limit(Some(k as i64), None)
        .unwrap()
        .project(&["_rowid"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert!(batch.num_rows() <= k, "expected at most {k} rows");
    batch.num_rows()
}

/// The default index (k=128, b=16, one segment), shared by the groups that
/// vary something else.
fn default_dataset(rt: &tokio::runtime::Runtime) -> &'static BenchDataset {
    static DATASET: OnceLock<BenchDataset> = OnceLock::new();
    DATASET.get_or_init(|| rt.block_on(build_dataset(&index_params(128, 16), 1, 0)))
}

/// Latency of an indexed search by `k`.
fn bench_minhash_search(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let bench_dataset = default_dataset(&rt);
    let queries = queries();

    let mut group = c.benchmark_group("minhash_lsh_search");
    for k in [1usize, 10, 100] {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            let mut next_query = 0usize;
            b.iter(|| {
                let text = &queries[next_query % queries.len()];
                next_query += 1;
                rt.block_on(search(&mut bench_dataset.dataset.scan(), text, k))
            });
        });
    }
    group.finish();
}

/// Latency by index parameters: fewer hashes shrink the signature reads,
/// more bands widen the candidate set.
fn bench_minhash_search_params(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let queries = queries();
    let others = [(64u32, 8u32), (128, 32)]
        .into_iter()
        .map(|(num_hashes, num_bands)| {
            (
                format!("k{num_hashes}_b{num_bands}"),
                rt.block_on(build_dataset(&index_params(num_hashes, num_bands), 1, 0)),
            )
        })
        .collect::<Vec<_>>();
    let mut bench_datasets = vec![("k128_b16".to_string(), default_dataset(&rt))];
    bench_datasets.extend(
        others
            .iter()
            .map(|(label, dataset)| (label.clone(), dataset)),
    );

    let mut group = c.benchmark_group("minhash_lsh_search_params");
    for (label, bench_dataset) in &bench_datasets {
        group.bench_with_input(BenchmarkId::from_parameter(label), label, |b, _| {
            let mut next_query = 0usize;
            b.iter(|| {
                let text = &queries[next_query % queries.len()];
                next_query += 1;
                rt.block_on(search(&mut bench_dataset.dataset.scan(), text, 10))
            });
        });
    }
    group.finish();
}

/// Latency by segment count: every segment answers the query and the hits
/// are merged by distance.
fn bench_minhash_search_segment_count(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let queries = queries();
    let others = [2usize, 4, 8]
        .into_iter()
        .map(|segment_count| {
            (
                segment_count,
                rt.block_on(build_dataset(&index_params(128, 16), segment_count, 0)),
            )
        })
        .collect::<Vec<_>>();
    let mut bench_datasets = vec![(1usize, default_dataset(&rt))];
    bench_datasets.extend(others.iter().map(|(count, dataset)| (*count, dataset)));

    let mut group = c.benchmark_group("minhash_lsh_search_segment_count");
    for (segment_count, bench_dataset) in &bench_datasets {
        group.bench_with_input(
            BenchmarkId::from_parameter(segment_count),
            segment_count,
            |b, _| {
                let mut next_query = 0usize;
                b.iter(|| {
                    let text = &queries[next_query % queries.len()];
                    next_query += 1;
                    rt.block_on(search(&mut bench_dataset.dataset.scan(), text, 10))
                });
            },
        );
    }
    group.finish();
}

/// Latency with an unindexed fragment: the flat path signs its rows on every
/// query unless the scan opts out with `fast_search`.
fn bench_minhash_search_unindexed_rows(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let queries = queries();
    let bench_dataset = rt.block_on(build_dataset(
        &index_params(128, 16),
        1,
        UNINDEXED_FRAGMENT_COUNT,
    ));

    let mut group = c.benchmark_group("minhash_lsh_search_unindexed_rows");
    for (label, fast_search) in [("index_and_flat", false), ("fast_search", true)] {
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &fast_search,
            |b, &fast| {
                let mut next_query = 0usize;
                b.iter(|| {
                    let text = &queries[next_query % queries.len()];
                    next_query += 1;
                    let mut scanner = bench_dataset.dataset.scan();
                    if fast {
                        scanner.fast_search();
                    }
                    rt.block_on(search(&mut scanner, text, 10))
                });
            },
        );
    }
    group.finish();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_minhash_search, bench_minhash_search_params,
        bench_minhash_search_segment_count, bench_minhash_search_unindexed_rows
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_minhash_search, bench_minhash_search_params,
        bench_minhash_search_segment_count, bench_minhash_search_unindexed_rows
);

criterion_main!(benches);
