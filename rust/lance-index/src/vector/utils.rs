// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::{
    array::AsArray,
    datatypes::{Float16Type, Float32Type, Float64Type},
};
use arrow_array::{Array, ArrayRef, BooleanArray, FixedSizeListArray};
use arrow_schema::{DataType, Field};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};
use lance_io::encodings::plain::bytes_to_array;
use lance_linalg::distance::DistanceType;
use prost::bytes;
use snafu::location;
use std::{ops::Range, sync::Arc};

use super::pb;
use crate::pb::Tensor;
use crate::vector::flat::storage::FlatFloatStorage;
use crate::vector::hnsw::builder::HnswBuildParams;
use crate::vector::hnsw::HNSW;
use crate::vector::v3::subindex::IvfSubIndex;

#[derive(Debug)]
pub struct SimpleIndex {
    store: FlatFloatStorage,
    index: HNSW,
}

impl SimpleIndex {
    pub fn try_new(store: FlatFloatStorage) -> Result<Self> {
        let hnsw = HNSW::index_vectors(
            &store,
            HnswBuildParams::default().ef_construction(10).num_edges(12),
        )?;
        Ok(Self { store, index: hnsw })
    }

    // train HNSW over the centroids to speed up finding the nearest clusters,
    // only train if all conditions are met:
    //  - the centroids are float32s or uint8s
    //  - `num_centroids * dimension >= 1_000_000`
    //      we benchmarked that it's 2x faster in the case of 1024 centroids and 1024 dimensions,
    //      so set the threshold to 1_000_000.
    pub fn may_train_index(
        centroids: ArrayRef,
        dimension: usize,
        distance_type: DistanceType,
    ) -> Result<Option<Self>> {
        if centroids.len() < 1_000_000 {
            // the centroids are stored in a flat array,
            // the length of the centroids is `num_centroids * dimension`
            return Ok(None);
        }

        match centroids.data_type() {
            DataType::Float32 => {
                let fsl =
                    FixedSizeListArray::try_new_from_values(centroids.clone(), dimension as i32)?;
                let store = FlatFloatStorage::new(fsl, distance_type);
                Self::try_new(store).map(Some)
            }
            _ => Ok(None),
        }
    }

    pub(crate) fn search(&self, query: ArrayRef) -> Result<(u32, f32)> {
        let res = self.index.search_basic(query, 1, 10, None, &self.store)?;
        Ok((res[0].id, res[0].dist.0))
    }
}

#[inline]
#[allow(dead_code)]
pub(crate) fn prefetch_arrow_array(array: &dyn Array) -> Result<()> {
    match array.data_type() {
        DataType::FixedSizeList(_, _) => {
            let array = array.as_fixed_size_list();
            return prefetch_arrow_array(array.values());
        }
        DataType::Float16 => {
            let array = array.as_primitive::<Float16Type>();
            do_prefetch(array.values().as_ptr_range())
        }
        DataType::Float32 => {
            let array = array.as_primitive::<Float32Type>();
            do_prefetch(array.values().as_ptr_range())
        }
        DataType::Float64 => {
            let array = array.as_primitive::<Float64Type>();
            do_prefetch(array.values().as_ptr_range())
        }
        _ => {
            return Err(Error::io(
                format!("unsupported prefetch on {} type", array.data_type()),
                location!(),
            ));
        }
    }

    Ok(())
}

#[inline]
pub(crate) fn do_prefetch<T>(ptrs: Range<*const T>) {
    // TODO use rust intrinsics instead of x86 intrinsics
    // TODO finish this
    unsafe {
        let (ptr, end_ptr) = (ptrs.start as *const i8, ptrs.end as *const i8);
        let mut current_ptr = ptr;
        while current_ptr < end_ptr {
            const CACHE_LINE_SIZE: usize = 64;
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                _mm_prefetch(current_ptr, _MM_HINT_T0);
            }
            current_ptr = current_ptr.add(CACHE_LINE_SIZE);
        }
    }
}

impl From<pb::tensor::DataType> for DataType {
    fn from(dt: pb::tensor::DataType) -> Self {
        match dt {
            pb::tensor::DataType::Uint8 => Self::UInt8,
            pb::tensor::DataType::Uint16 => Self::UInt16,
            pb::tensor::DataType::Uint32 => Self::UInt32,
            pb::tensor::DataType::Uint64 => Self::UInt64,
            pb::tensor::DataType::Float16 => Self::Float16,
            pb::tensor::DataType::Float32 => Self::Float32,
            pb::tensor::DataType::Float64 => Self::Float64,
            pb::tensor::DataType::Bfloat16 => unimplemented!(),
        }
    }
}

impl TryFrom<&DataType> for pb::tensor::DataType {
    type Error = Error;

    fn try_from(dt: &DataType) -> Result<Self> {
        match dt {
            DataType::UInt8 => Ok(Self::Uint8),
            DataType::UInt16 => Ok(Self::Uint16),
            DataType::UInt32 => Ok(Self::Uint32),
            DataType::UInt64 => Ok(Self::Uint64),
            DataType::Float16 => Ok(Self::Float16),
            DataType::Float32 => Ok(Self::Float32),
            DataType::Float64 => Ok(Self::Float64),
            _ => Err(Error::Index {
                message: format!("pb tensor type not supported: {:?}", dt),
                location: location!(),
            }),
        }
    }
}

impl TryFrom<DataType> for pb::tensor::DataType {
    type Error = Error;

    fn try_from(dt: DataType) -> Result<Self> {
        (&dt).try_into()
    }
}

impl TryFrom<&FixedSizeListArray> for pb::Tensor {
    type Error = Error;

    fn try_from(array: &FixedSizeListArray) -> Result<Self> {
        let mut tensor = Self::default();
        tensor.data_type = pb::tensor::DataType::try_from(array.value_type())? as i32;
        tensor.shape = vec![array.len() as u32, array.value_length() as u32];
        let flat_array = array.values();
        tensor.data = flat_array.into_data().buffers()[0].to_vec();
        Ok(tensor)
    }
}

impl TryFrom<&pb::Tensor> for FixedSizeListArray {
    type Error = Error;

    fn try_from(tensor: &Tensor) -> Result<Self> {
        if tensor.shape.len() != 2 {
            return Err(Error::Index {
                message: format!("only accept 2-D tensor shape, got: {:?}", tensor.shape),
                location: location!(),
            });
        }
        let dim = tensor.shape[1] as usize;
        let num_rows = tensor.shape[0] as usize;

        let data = bytes::Bytes::from(tensor.data.clone());
        let flat_array = bytes_to_array(
            &DataType::from(pb::tensor::DataType::try_from(tensor.data_type).unwrap()),
            data,
            dim * num_rows,
            0,
        )?;

        if flat_array.len() != dim * num_rows {
            return Err(Error::Index {
                message: format!(
                    "Tensor shape {:?} does not match to data len: {}",
                    tensor.shape,
                    flat_array.len()
                ),
                location: location!(),
            });
        }

        let field = Field::new("item", flat_array.data_type().clone(), true);
        Ok(Self::try_new(
            Arc::new(field),
            dim as i32,
            flat_array,
            None,
        )?)
    }
}

/// Check if all vectors in the FixedSizeListArray are finite
/// null values are considered as not finite
/// returns a BooleanArray
/// with the same length as the FixedSizeListArray
/// with true for finite values and false for non-finite values
pub fn is_finite(fsl: &FixedSizeListArray) -> BooleanArray {
    let is_finite = fsl
        .iter()
        .map(|v| match v {
            Some(v) => match v.data_type() {
                DataType::Float16 => {
                    let v = v.as_primitive::<Float16Type>();
                    v.null_count() == 0 && v.values().iter().all(|v| v.is_finite())
                }
                DataType::Float32 => {
                    let v = v.as_primitive::<Float32Type>();
                    v.null_count() == 0 && v.values().iter().all(|v| v.is_finite())
                }
                DataType::Float64 => {
                    let v = v.as_primitive::<Float64Type>();
                    v.null_count() == 0 && v.values().iter().all(|v| v.is_finite())
                }
                _ => v.null_count() == 0,
            },
            None => false,
        })
        .collect::<Vec<_>>();
    BooleanArray::from(is_finite)
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{Float16Array, Float32Array, Float64Array};
    use half::f16;
    use lance_arrow::FixedSizeListArrayExt;
    use num_traits::identities::Zero;

    #[test]
    fn test_fsl_to_tensor() {
        let fsl =
            FixedSizeListArray::try_new_from_values(Float16Array::from(vec![f16::zero(); 20]), 5)
                .unwrap();
        let tensor = pb::Tensor::try_from(&fsl).unwrap();
        assert_eq!(tensor.data_type, pb::tensor::DataType::Float16 as i32);
        assert_eq!(tensor.shape, vec![4, 5]);
        assert_eq!(tensor.data.len(), 20 * 2);

        let fsl =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0; 20]), 5).unwrap();
        let tensor = pb::Tensor::try_from(&fsl).unwrap();
        assert_eq!(tensor.data_type, pb::tensor::DataType::Float32 as i32);
        assert_eq!(tensor.shape, vec![4, 5]);
        assert_eq!(tensor.data.len(), 20 * 4);

        let fsl =
            FixedSizeListArray::try_new_from_values(Float64Array::from(vec![0.0; 20]), 5).unwrap();
        let tensor = pb::Tensor::try_from(&fsl).unwrap();
        assert_eq!(tensor.data_type, pb::tensor::DataType::Float64 as i32);
        assert_eq!(tensor.shape, vec![4, 5]);
        assert_eq!(tensor.data.len(), 20 * 8);
    }
}
