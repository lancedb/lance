//! Compute distance
//!
//! Support method:
//!  - Euclidean Distance (L2)

use arrow_arith::aggregate::sum;
use arrow_arith::arithmetic::{multiply, subtract};
use arrow_array::{Array, FixedSizeListArray, Float32Array};
use arrow_schema::ArrowError;

/// Euclidean distance from a point to a list of points.
pub fn euclidean_distance(
    from: &Float32Array,
    to: &FixedSizeListArray,
) -> Result<Float32Array, ArrowError> {
    assert_eq!(from.len(), to.value_length() as usize);

    // Naive implementation.
    // TODO: benchmark and use SIMD or BLAS
    let scores: Float32Array = (0..to.len())
        .map(|idx| to.value(idx))
        .map(|left| {
            let arr = left.as_any().downcast_ref::<Float32Array>().unwrap();
            let mut sub = subtract(arr, from).unwrap();
            sub = multiply(&sub, &sub).unwrap();
            sum(&sub).unwrap().sqrt()
        })
        .collect();
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use arrow_array::types::Float32Type;

    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let mat = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vec![
                Some(vec![Some(1.0), Some(2.0), Some(3.0)]),
                Some(vec![Some(2.0), Some(3.0), Some(4.0)]),
                Some(vec![Some(3.0), Some(4.0), Some(5.0)]),
                Some(vec![Some(4.0), Some(5.0), Some(6.0)]),
            ],
            3,
        );
        let point = Float32Array::from(vec![2.0, 3.0, 4.0]);
        let scores = euclidean_distance(&point, &mat).unwrap();

        assert_eq!(
            scores,
            Float32Array::from(vec![3.0_f32.sqrt(), 0_f32, 3.0_f32.sqrt(), 12.0_f32.sqrt()])
        );
    }
}
