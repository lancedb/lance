//! Test utilities

//! Testing utilities
//!
//! TODO: How to make this repo tests and bench only?

use std::iter::{repeat, repeat_with};

use arrow_array::{types::Float32Type, FixedSizeListArray, Float32Array};
use rand::Rng;

pub fn generate_random_array(n: usize) -> Float32Array {
    let mut rng = rand::thread_rng();
    Float32Array::from(
        repeat_with(|| rng.gen::<f32>())
            .take(n)
            .collect::<Vec<f32>>(),
    )
}

pub fn generate_random_matrix(dim: i32, rows: i32) -> FixedSizeListArray {
    let mut rng = rand::thread_rng();
    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        repeat(0).take(rows as usize).map(|_| {
            Some(
                repeat_with(|| Some(rng.gen::<f32>()))
                    .take(dim as usize)
                    .collect::<Vec<Option<f32>>>(),
            )
        }),
        dim,
    )
}
