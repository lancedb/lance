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

    let dataset = Arc::new(DatasetBuilder::from_uri(uri).with_index_cache_size(10000).load().await.unwrap());

    println!("Start benchmarking");
    for _ in 0..5 {
        println!("warmup run...");
        let _batch = dataset
            .scan()
            .nearest(&column, &random_vector(512), 100)
            .unwrap()
            .nprobs(10000)
            .try_into_batch()
            .await
            .unwrap();
    }
    println!("Finished warmup");

    let mut acc = 0.0;
    let batch_size = 100;

    let mut stream = stream::repeat(dataset)
        .map(|d| {
            let col = column.clone();

            async move {
                let start = std::time::Instant::now();
                d.scan()
                    .nearest(&col, &random_vector(512), 1)
                    .unwrap()
                    .nprobs(20)
                    // .filter("year >= 2010 AND year < 2020")
                    // .unwrap()
                    .prefilter(true)
                    .project(&["id"])
                    .unwrap()
                    .try_into_batch()
                    .await
                    .unwrap();
                let elapsed = start.elapsed();
                elapsed
            }
        })
        .map(|f| tokio::spawn(f))
        .buffer_unordered(concurrency);

    let mut counter = 0;
    while let Some(elapsed) = stream.next().await {
        acc += elapsed.unwrap().as_secs_f64();
        counter += 1;
        if counter % batch_size == 0 {
            println!("Avg latency: {:.2} ms", acc / batch_size as f64 * 1000.0);
            acc = 0.0;
        }
        if counter == 10000 * concurrency {
            break;
        }
    }
}
