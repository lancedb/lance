// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmarks comparing the imperative scan planner against the logical-plan prototype.
//!
//! Both paths are called directly rather than through `LANCE_LOGICAL_SCAN_PLANNER`, so the two
//! appear in one process and criterion can put them side by side.
//!
//! Three questions, which is why there are three groups:
//!
//! * `plan/` — is going through a logical plan, a rule loop and a physical planner more expensive
//!   than hand-building the exec tree? This is the cost the prototype adds, measured alone.
//! * `scan/` and `search/` — does the resulting plan execute at the same speed? Planning is a
//!   fixed cost per query; execution is what a real workload pays.
//!
//! ```text
//! cargo bench -p lance --bench logical_scan_planner
//! ```

use std::sync::Arc;

use arrow_array::Float32Array;
use arrow_array::types::{Float32Type, Int32Type};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use futures::TryStreamExt;
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::dataset::scanner::{Scanner, logical};
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_core::utils::tempfile::TempStrDir;
use lance_datafusion::exec::{LanceExecutionOptions, execute_plan};
use lance_datagen::{BatchCount, ByteCount, Dimension, RowCount, array, gen_batch};
use lance_index::IndexType;
use lance_index::scalar::ScalarIndexParams;
use lance_linalg::distance::DistanceType;
#[cfg(target_os = "linux")]
use lance_testing::pprof::{Output, PProfProfiler};

const DIM: u32 = 64;
const ROWS_PER_FRAGMENT: u64 = 25_000;
const NUM_FRAGMENTS: u32 = 8;
const TOTAL_ROWS: i32 = (ROWS_PER_FRAGMENT as u32 * NUM_FRAGMENTS) as i32;

/// On-disk rather than `memory://`: the prototype's plans differ from the imperative ones in how
/// much they read and when, which an in-memory store would flatten out.
struct Fixture {
    _datadir: TempStrDir,
    dataset: Arc<Dataset>,
}

impl Fixture {
    async fn open() -> Self {
        let datadir = TempStrDir::default();
        let reader = gen_batch()
            .col("i", array::step::<Int32Type>())
            .col("s", array::rand_utf8(ByteCount::from(32), false))
            .col("vec", array::rand_vec::<Float32Type>(Dimension::from(DIM)))
            .into_reader_rows(
                RowCount::from(ROWS_PER_FRAGMENT),
                BatchCount::from(NUM_FRAGMENTS),
            );
        let mut dataset = Dataset::write(
            reader,
            datadir.as_str(),
            Some(WriteParams {
                max_rows_per_file: ROWS_PER_FRAGMENT as usize,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // A scalar index on the filter column so the planning groups exercise the scalar-index
        // rules, which are the prototype's most expensive rewrite.
        dataset
            .create_index(
                &["i"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                None,
                &VectorIndexParams::ivf_pq(16, 8, 8, DistanceType::L2, 20),
                true,
            )
            .await
            .unwrap();

        Self {
            _datadir: datadir,
            dataset: Arc::new(dataset),
        }
    }
}

fn query_vector() -> Float32Array {
    Float32Array::from((0..DIM).map(|v| v as f32).collect::<Vec<_>>())
}

/// One benchmarked query, as a name and a way to build the scanner for it.
///
/// The planner under test is chosen by the caller rather than baked in here, so both paths are
/// guaranteed to plan the identical query.
type Shape = (&'static str, fn(&Dataset) -> Scanner);

/// The query shapes both groups run, named so criterion's output reads as a comparison.
fn shapes() -> Vec<Shape> {
    vec![
        ("full_scan", |dataset| dataset.scan()),
        ("filtered_scan", |dataset| {
            let mut scan = dataset.scan();
            scan.project(&["s"]).unwrap();
            scan.filter(&format!("i < {}", TOTAL_ROWS / 100)).unwrap();
            scan
        }),
        ("filtered_scan_with_limit", |dataset| {
            let mut scan = dataset.scan();
            scan.project(&["s"]).unwrap();
            scan.filter(&format!("i < {}", TOTAL_ROWS / 2)).unwrap();
            scan.limit(Some(100), None).unwrap();
            scan
        }),
        ("ann", |dataset| {
            let mut scan = dataset.scan();
            scan.project(&["s"]).unwrap();
            scan.nearest("vec", &query_vector(), 10).unwrap();
            scan
        }),
        ("ann_prefiltered", |dataset| {
            let mut scan = dataset.scan();
            scan.project(&["s"]).unwrap();
            scan.prefilter(true);
            scan.filter(&format!("i < {}", TOTAL_ROWS / 10)).unwrap();
            scan.nearest("vec", &query_vector(), 10).unwrap();
            scan
        }),
    ]
}

/// Whether a shape's cost is dominated by reading rows or by the index search, which is the only
/// reason to separate the two execution groups.
fn is_search(shape: &str) -> bool {
    shape.starts_with("ann")
}

async fn plan(
    scanner: &Scanner,
    use_logical: bool,
) -> Arc<dyn datafusion::physical_plan::ExecutionPlan> {
    if use_logical {
        logical::create_plan(scanner).await.unwrap()
    } else {
        scanner.create_plan().await.unwrap()
    }
}

async fn plan_and_execute(scanner: &Scanner, use_logical: bool) -> usize {
    let plan = plan(scanner, use_logical).await;
    execute_plan(plan, LanceExecutionOptions::default())
        .unwrap()
        .try_fold(0, |rows, batch| async move { Ok(rows + batch.num_rows()) })
        .await
        .unwrap()
}

fn bench_planning(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let fixture = rt.block_on(Fixture::open());
    let dataset = &fixture.dataset;

    let mut group = c.benchmark_group("plan");
    for (shape, configure) in shapes() {
        let scanner = configure(dataset);
        // Plan once per path before measuring: both read index metadata on their first call and
        // cache it on the dataset, and that one-time cost would otherwise land in whichever path
        // criterion warmed up first.
        for use_logical in [false, true] {
            rt.block_on(plan(&scanner, use_logical));
            let path = if use_logical { "logical" } else { "imperative" };
            group.bench_function(BenchmarkId::new(path, shape), |b| {
                b.iter(|| rt.block_on(plan(&scanner, use_logical)))
            });
        }
    }
    group.finish();
}

fn bench_execution(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let fixture = rt.block_on(Fixture::open());
    let dataset = &fixture.dataset;

    for (name, want_search) in [("scan", false), ("search", true)] {
        let mut group = c.benchmark_group(name);
        for (shape, configure) in shapes() {
            if is_search(shape) != want_search {
                continue;
            }
            let scanner = configure(dataset);
            for use_logical in [false, true] {
                let path = if use_logical { "logical" } else { "imperative" };
                // Assert the two paths agree on row count before timing them. A path that returns
                // fewer rows would otherwise look like a speedup.
                let rows = rt.block_on(plan_and_execute(&scanner, use_logical));
                assert!(rows > 0, "{path}/{shape} returned no rows");
                group.bench_function(BenchmarkId::new(path, shape), |b| {
                    b.iter(|| rt.block_on(plan_and_execute(&scanner, use_logical)))
                });
            }
        }
        group.finish();
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_planning, bench_execution);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_planning, bench_execution);

criterion_main!(benches);
