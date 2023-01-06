//! Compute distance
//!
//! Support method:
//!  - Euclidean Distance (L2)

use rayon::prelude::*;

use arrow_arith::aggregate::sum;
use arrow_arith::arithmetic::{multiply, powf_scalar, subtract};
use arrow_array::{Array, FixedSizeListArray, Float32Array};
use arrow_schema::{ArrowError, DataType};

/// Euclidean distance from a point to a list of points.
pub fn euclidean_distance(
    from: &Float32Array,
    to: &FixedSizeListArray,
) -> Result<Vec<f32>, ArrowError> {
    if from.len() != to.value_length() as usize {
        return Err(ArrowError::ComputeError(format!("Index has {} dims but in put has {}", to.value_length(), from.len())));
    }
    if to.value_type() != DataType::Float32 {
        return Err(ArrowError::SchemaError("Index needs to have Float32 values".to_string()));
    }

    // Naive implementation.
    // TODO: benchmark and use SIMD or BLAS
    let scores = (0..to.len())
        .into_par_iter()
        .map(|idx| to.value(idx))
        .map(|left| {
            let arr = left.as_any().downcast_ref::<Float32Array>().unwrap();
            euclidean_distanc(from, arr).unwrap()  // how to handle NAs ?
        })
        .collect();
    Ok(scores)
}


pub fn euclidean_distanc(
    from: &Float32Array,
    to: &Float32Array
) -> Result<f32, ArrowError> {
    let diff = subtract(from, to)?;
    let pow = multiply(&diff, &diff)?;
    sum(&pow).ok_or_else(|| ArrowError::ComputeError("".to_string()))
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
            Float32Array::from(scores),
            Float32Array::from(vec![3.0_f32, 0_f32, 3.0_f32, 12.0_f32])
        );
    }
}
