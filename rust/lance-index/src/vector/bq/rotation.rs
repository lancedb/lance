// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use num_traits::AsPrimitive;
use rand::RngCore;

const FAST_ROTATION_ROUNDS: usize = 4;

#[inline]
fn fwht_in_place(values: &mut [f32]) {
    debug_assert!(values.len().is_power_of_two());
    let mut half = 1usize;
    while half < values.len() {
        let step = half * 2;
        for i in (0..values.len()).step_by(step) {
            for j in i..(i + half) {
                let x = values[j];
                let y = values[j + half];
                values[j] = x + y;
                values[j + half] = x - y;
            }
        }
        half = step;
    }
}

#[inline]
fn flip_signs(values: &mut [f32], signs: &[u8]) {
    debug_assert!(signs.len() * 8 >= values.len());
    for (idx, value) in values.iter_mut().enumerate() {
        if (signs[idx / 8] >> (idx % 8)) & 1 == 1 {
            *value = -*value;
        }
    }
}

#[inline]
fn kacs_walk(values: &mut [f32]) {
    let half = values.len() / 2;
    for i in 0..half {
        let x = values[i];
        let y = values[i + half];
        values[i] = x + y;
        values[i + half] = x - y;
    }
}

#[inline]
fn rescale(values: &mut [f32], factor: f32) {
    values.iter_mut().for_each(|v| *v *= factor);
}

#[inline]
fn sign_bytes_per_round(dim: usize) -> usize {
    dim.div_ceil(8)
}

pub fn random_fast_rotation_signs(dim: usize) -> Vec<u8> {
    let mut signs = vec![0u8; FAST_ROTATION_ROUNDS * sign_bytes_per_round(dim)];
    rand::rng().fill_bytes(&mut signs);
    signs
}

pub fn apply_fast_rotation<T: AsPrimitive<f32>>(input: &[T], output: &mut [f32], signs: &[u8]) {
    let dim = output.len();
    let bytes_per_round = sign_bytes_per_round(dim);
    debug_assert_eq!(signs.len(), FAST_ROTATION_ROUNDS * bytes_per_round);
    output.fill(0.0);
    output
        .iter_mut()
        .zip(input.iter())
        .for_each(|(dst, src)| *dst = src.as_());

    if dim == 0 {
        return;
    }

    let trunc_dim = 1usize << dim.ilog2();
    let scale = 1.0f32 / (trunc_dim as f32).sqrt();
    if trunc_dim == dim {
        for round in 0..FAST_ROTATION_ROUNDS {
            let offset = round * bytes_per_round;
            flip_signs(output, &signs[offset..offset + bytes_per_round]);
            fwht_in_place(output);
            rescale(output, scale);
        }
        return;
    }

    let start = dim - trunc_dim;
    for round in 0..FAST_ROTATION_ROUNDS {
        let offset = round * bytes_per_round;
        flip_signs(output, &signs[offset..offset + bytes_per_round]);

        if round % 2 == 0 {
            let head = &mut output[..trunc_dim];
            fwht_in_place(head);
            rescale(head, scale);
        } else {
            let tail = &mut output[start..];
            fwht_in_place(tail);
            rescale(tail, scale);
        }

        kacs_walk(output);
    }

    // Matches RaBitQ-Library FhtKacRotator behavior for non-power-of-two dimensions.
    rescale(output, 0.25);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_rotation_sign_bytes() {
        assert_eq!(random_fast_rotation_signs(128).len(), 64);
        assert_eq!(random_fast_rotation_signs(130).len(), 68);
    }

    #[test]
    fn test_fast_rotation_preserves_shape() {
        let input = vec![1.0f32; 129];
        let mut output = vec![0.0f32; 129];
        let signs = random_fast_rotation_signs(129);
        apply_fast_rotation(&input, &mut output, &signs);
        assert_eq!(output.len(), 129);
    }
}
