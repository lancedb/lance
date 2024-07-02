use std::{env, sync::Arc};

use arrow_array::Float32Array;
use futures::{stream, StreamExt};
use lance::dataset::builder::DatasetBuilder;

fn random_vector(dim: usize) -> Float32Array {
    let mut vec = Vec::with_capacity(dim);
    for _ in 0..dim {
        vec.push(rand::random::<f32>());
    }
    Float32Array::from(vec)
}

#[tokio::main]
async fn main() {
    let args = env::args().collect::<Vec<String>>();
    let uri = args[1].clone();
    let column = args[2].clone();
    let concurrency = args[3].parse::<usize>().unwrap();

    let dataset = Arc::new(DatasetBuilder::from_uri(uri).load().await.unwrap());

    println!("Start benchmarking");
    for _ in 0..3 {
        let _batch = dataset
            .scan()
            .nearest(&column, &random_vector(512), 100)
            .unwrap()
            .nprobs(128)
            .try_into_batch()
            .await
            .unwrap();
    }
    println!("Finished warmup");

    let mut stream = stream::repeat(dataset)
        .map(|d| {
            let col = column.clone();

            async move {
                let start = std::time::Instant::now();
                d.scan()
                    .nearest(&col, &random_vector(512), 100)
                    .unwrap()
                    .nprobs(128)
                    .try_into_batch()
                    .await
                    .unwrap();
                let elapsed = start.elapsed();
                elapsed
            }
        })
        .map(|f| tokio::spawn(f))
        .buffer_unordered(concurrency);

    while let Some(elapsed) = stream.next().await {
        println!("{:?}", elapsed.unwrap());
    }
}
