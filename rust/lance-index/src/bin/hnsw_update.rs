use std::{sync::Arc, time::Instant};

use arrow::datatypes::Float32Type;
use lance_index::vector::{graph::memory::InMemoryVectorStorage, hnsw::{builder::HnswBuildParams, HNSWBuilder}};
use lance_linalg::{distance::MetricType, MatrixView};
use lance_testing::datagen::generate_random_array_with_seed;

#[tokio::main]
async fn main() {
    const DIMENSION: usize = 512;
    const TOTAL: usize = 1024 * 1024;
    const SEED: [u8; 32] = [42; 32];

    let data = generate_random_array_with_seed::<Float32Type>(TOTAL * DIMENSION, SEED);
    let mat = Arc::new(MatrixView::<Float32Type>::new(data.into(), DIMENSION));
    let vectors = Arc::new(InMemoryVectorStorage::new(mat.clone(), MetricType::L2));

    let start = Instant::now();
    let _hnsw = HNSWBuilder::with_params(
        HnswBuildParams::default().max_level(6).use_select_heuristic(true),
        vectors.clone(),
    )
    .build()
    .unwrap();
    // let uids: HashSet<u32> = hnsw
    //     .search(query, K, 300, None)
    //     .unwrap()
    //     .iter()
    //     .map(|node| node.id)
    //     .collect();

    println!("Time: {:?}", start.elapsed());
}
