// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::hint::black_box;
use std::time::Duration;

use arrow_array::{Array, FixedSizeListArray, Float32Array};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lance_arrow::FixedSizeListArrayExt;
use lance_index::pb;
use lance_index::vector::pca::{PcaModel, PcaParams};

fn model(input_dimension: usize, output_dimension: usize) -> PcaModel {
    let scale = (2.0 / input_dimension as f64).sqrt();
    let matrix: Vec<f32> = (0..output_dimension)
        .flat_map(|row| {
            (0..input_dimension).map(move |column| {
                (scale
                    * (std::f64::consts::PI * (column as f64 + 0.5) * (row + 1) as f64
                        / input_dimension as f64)
                        .cos()) as f32
            })
        })
        .collect();
    let mean: Vec<f32> = (0..input_dimension)
        .map(|column| (column % 17) as f32 / 100.0)
        .collect();
    let tensor = |values: &[f32], shape| pb::Tensor {
        data_type: pb::tensor::DataType::Float32 as i32,
        shape,
        data: values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    };
    PcaModel::try_from(pb::PcaModel {
        model_version: 1,
        transform_matrix: Some(tensor(
            &matrix,
            vec![output_dimension as u32, input_dimension as u32],
        )),
        mean: Some(tensor(&mean, vec![input_dimension as u32])),
    })
    .unwrap()
}

fn values(count: usize) -> Float32Array {
    Float32Array::from(
        (0..count)
            .map(|index| ((index as f32 + 0.5) * 0.173).sin())
            .collect::<Vec<_>>(),
    )
}

fn bench_pca(c: &mut Criterion) {
    let mut query_group = c.benchmark_group("pca_query");
    for (input, output) in [(128, 32), (768, 128), (960, 256), (1024, 512), (1536, 384)] {
        let model = model(input, output);
        let query = values(input);
        let projected = model.transform_query(&query).unwrap();
        assert_eq!(projected.len(), output);
        query_group.bench_with_input(
            BenchmarkId::new(format!("{input}d"), output),
            &query,
            |b, query| b.iter(|| model.transform_query(black_box(query)).unwrap()),
        );
    }
    query_group.finish();

    let mut batch_group = c.benchmark_group("pca_batch");
    for (rows, input, output) in [(64, 768, 128), (256, 960, 256)] {
        let model = model(input, output);
        let vectors =
            FixedSizeListArray::try_new_from_values(values(rows * input), input as i32).unwrap();
        batch_group.bench_with_input(
            BenchmarkId::new(format!("{rows}x{input}d"), output),
            &vectors,
            |b, vectors| b.iter(|| model.transform(black_box(vectors)).unwrap()),
        );
    }
    batch_group.finish();

    let vectors = FixedSizeListArray::try_new_from_values(
        Float32Array::from(
            (0..256 * 32)
                .map(|index| {
                    let mixed = (index as u32)
                        .wrapping_mul(747796405)
                        .wrapping_add(2891336453);
                    ((mixed ^ (mixed >> 16)) % 10007) as f32 / 10007.0
                })
                .collect::<Vec<_>>(),
        ),
        32,
    )
    .unwrap();
    c.bench_function("pca_train/256x32d_to_8", |b| {
        b.iter(|| PcaModel::train(black_box(&vectors), &PcaParams::new(8)).unwrap())
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1));
    targets = bench_pca
}
criterion_main!(benches);
