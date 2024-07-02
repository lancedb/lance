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

    let dataset = Arc::new(
        DatasetBuilder::from_uri(uri)
            .with_index_cache_size(10000)
            .load()
            .await
            .unwrap(),
    );

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

    for concurrency in 1..=32 {
        let mut stream = stream::repeat(dataset.clone())
            .map(|d| {
                let col = column.clone();
                async move {
                    let start = std::time::Instant::now();
                    d.scan()
                        .nearest(&col, &random_vector(512), 10)
                        .unwrap()
                        .nprobs(4)
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

        let mut acc = Vec::with_capacity(10000);
        let mut counter = 0;
        while let Some(elapsed) = stream.next().await {
            counter += 1;
            // skip the first 100 runs
            if counter < 100 * concurrency {
                continue;
            }
            acc.push(elapsed.unwrap().as_micros());
            if acc.len() == 10000 {
                break;
            }
        }

        acc.sort_unstable();
        let p50 = acc[acc.len() / 2];
        let p90 = acc[(acc.len() as f64 * 0.9) as usize];
        let p99 = acc[(acc.len() as f64 * 0.99) as usize];
        let p999 = acc[(acc.len() as f64 * 0.999) as usize];
        let avg = acc.iter().sum::<u128>() / acc.len() as u128;
        let std = (acc.iter().map(|x| (*x - avg)).map(|x| (x * x)).sum::<u128>()
            / acc.len() as u128) as f32;
        let std = std.sqrt();
        println!(
            "{}, {}, {}, {}, {}, {}, {:.2}",
            concurrency, p50, p90, p99, p999, avg, std
        );
    }
}
