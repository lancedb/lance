use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use arrow::datatypes::Float32Type;
use futures::{stream, StreamExt};
use lance_core::{Error, Result};
use lance_index::vector::{
    graph::memory::InMemoryVectorStorage,
    hnsw::{builder::HnswBuildParams, HNSWBuilder, HNSW},
};
use lance_linalg::{distance::MetricType, MatrixView};
use lance_testing::datagen::generate_random_array_with_seed;
use snafu::{location, Location};
use tokio::sync::RwLock;

struct ValueHolder<T> {
    value: Arc<T>,
}

struct LeftRight<T> {
    is_left: AtomicBool,
    left: Arc<RwLock<ValueHolder<T>>>,
    right: Arc<RwLock<ValueHolder<T>>>,
}

impl<T> LeftRight<T> {
    fn new(val: Arc<T>) -> Self {
        Self {
            is_left: AtomicBool::new(true),
            left: Arc::new(RwLock::new(ValueHolder { value: val.clone() })),
            right: Arc::new(RwLock::new(ValueHolder { value: val.clone() })),
        }
    }

    async fn get(&self) -> Arc<T> {
        if self.is_left.load(Ordering::Relaxed) {
            self.left.read().await.value.clone()
        } else {
            self.right.read().await.value.clone()
        }
    }

    fn use_left(&self) {
        self.is_left.store(true, Ordering::Relaxed)
    }

    fn use_right(&self) {
        self.is_left.store(false, Ordering::Relaxed)
    }

    async fn set_left(&self, val: Arc<T>) -> Result<()> {
        if self.is_left.load(Ordering::Relaxed) {
            return Err(Error::Internal {
                message: "left is active".into(),
                location: location!(),
            });
        };

        self.left.write().await.value = val;

        Ok(())
    }

    async fn set_right(&self, val: Arc<T>) -> Result<()> {
        if !self.is_left.load(Ordering::Relaxed) {
            return Err(Error::Internal {
                message: "right is active".into(),
                location: location!(),
            });
        };

        self.right.write().await.value = val;

        Ok(())
    }
}

async fn run_query_loop(hnsw_holder: Arc<LeftRight<HNSW>>) {
    let mut s = stream::iter(0..100000)
        .map(move |_| {
            let hnsw_holder = hnsw_holder.clone();
            async move {
                let start = std::time::Instant::now();
                let hnsw = hnsw_holder.get().await;
                let _res = hnsw.search(&[0.0; 512], 10, 150, None);
                start.elapsed()
            }
        })
        .buffered(4);

    let mut counter = 0;
    let mut total_duration = std::time::Duration::new(0, 0);

    while let Some(duration) = s.next().await {
        total_duration += duration;
        counter += 1;
        if counter % 1000 == 0 {
            println!("average duration: {:?}", total_duration / counter);
            total_duration = std::time::Duration::new(0, 0);
            counter = 0;
        }
    }
}

#[tokio::main]
async fn main() {
    test(
        HnswBuildParams::default()
            .use_select_heuristic(false)
            .num_edges(20)
            .max_num_edges(40)
            .max_level(6),
        1024 * 1024 * 10,
        20000,
    )
    .await;
}

async fn test(hnsw_param: HnswBuildParams, base_vecs: usize, new_vecs: usize) {
    const DIMENSION: usize = 512;
    const SEED: [u8; 32] = [42; 32];

    let data = generate_random_array_with_seed::<Float32Type>(base_vecs * DIMENSION, SEED);
    let mat = Arc::new(MatrixView::<Float32Type>::new(
        data.clone().into(),
        DIMENSION,
    ));
    let vectors = Arc::new(InMemoryVectorStorage::new(mat.clone(), MetricType::L2));

    let mut hnsw_builder = HNSWBuilder::with_params(hnsw_param, vectors.clone());

    // build a new hnsw index
    hnsw_builder.index().unwrap();
    // go into diff recording mode
    hnsw_builder.record_diff();

    let mut new_data =
        arrow::array::Float32Array::builder(base_vecs * DIMENSION + DIMENSION * 1000);

    new_data.append_slice(data.values());
    new_data.append_slice(
        generate_random_array_with_seed::<Float32Type>(new_vecs * DIMENSION, [12; 32]).values(),
    );
    let new_data = new_data.finish();
    let mat = Arc::new(MatrixView::<Float32Type>::new(new_data.into(), DIMENSION));
    let vectors = Arc::new(InMemoryVectorStorage::new(mat.clone(), MetricType::L2));

    hnsw_builder.set_vectors(vectors.clone());
    hnsw_builder.set_offset(base_vecs);

    hnsw_builder.index().unwrap();

    let hnsw = Arc::new(hnsw_builder.get_index());
    let hnsw_holder = Arc::new(LeftRight::new(hnsw.clone()));

    let loop_handle = tokio::spawn(run_query_loop(hnsw_holder.clone()));

    let diff = hnsw_builder.diff();
    println!(
        "diff len {}, {:.2}% of the data changed",
        diff.len(),
        diff.len() as f32 / (base_vecs + new_vecs) as f32 * 100.0
    );

    let start = std::time::Instant::now();
    let new_hnsw = Arc::new(hnsw.apply_diff(diff, vectors));
    println!("apply diff takes {:?}", start.elapsed());

    tokio::time::sleep(Duration::from_secs(5)).await;

    println!("setting right to new hnsw");
    hnsw_holder.set_right(new_hnsw).await.unwrap();
    hnsw_holder.use_right();

    println!("right is now active");
    loop_handle.await.unwrap();
}
