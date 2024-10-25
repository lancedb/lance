// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{types::ArrowPrimitiveType, ArrayRef, FixedSizeListArray, PrimitiveArray};
use lance_arrow::FixedSizeListArrayExt;
use log::info;
use rand::{seq::IteratorRandom, Rng};
use snafu::{location, Location};

use lance_core::{Error, Result};
use lance_linalg::{
    distance::{DistanceType, Dot, Normalize, L2},
    kmeans::{KMeans, KMeansParams},
};

/// Train KMeans model and returns the centroids of each cluster.
#[allow(clippy::too_many_arguments)]
pub fn train_kmeans<T: ArrowPrimitiveType>(
    array: &[T::Native],
    dimension: usize,
    k: usize,
    max_iterations: u32,
    redos: usize,
    mut rng: impl Rng,
    distance_type: DistanceType,
    sample_rate: usize,
) -> Result<ArrayRef>
where
    T::Native: Dot + L2 + Normalize,
    PrimitiveArray<T>: From<Vec<T::Native>>,
{
    let num_rows = array.len() / dimension;
    if num_rows < k {
        return Err(Error::Index{message: format!(
            "KMeans: can not train {k} centroids with {num_rows} vectors, choose a smaller K (< {num_rows}) instead"
        ),location: location!()});
    }

    // Only sample sample_rate * num_clusters. See Faiss
    let data = if num_rows > sample_rate * k {
        info!(
            "Sample {} out of {} to train kmeans of {} dim, {} clusters",
            sample_rate * k,
            array.len() / dimension,
            dimension,
            k,
        );
        let sample_size = sample_rate * k;
        let chosen = (0..num_rows).choose_multiple(&mut rng, sample_size);
        let mut builder = Vec::with_capacity(sample_size * dimension);
        for idx in chosen.iter() {
            let s = &array[idx * dimension..(idx + 1) * dimension];
            builder.extend_from_slice(s);
        }
        PrimitiveArray::<T>::from(builder)
    } else {
        PrimitiveArray::<T>::from(array.to_vec())
    };

    let params = KMeansParams {
        max_iters: max_iterations,
        distance_type,
        redos,
        ..Default::default()
    };
    let data = FixedSizeListArray::try_new_from_values(data, dimension as i32)?;
    let model = KMeans::new_with_params(&data, k, &params)?;
    Ok(model.centroids.clone())
}
