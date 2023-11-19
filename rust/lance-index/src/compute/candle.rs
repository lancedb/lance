use std::{fmt::Debug, sync::Arc};

use arrow::{array::ArrayData, buffer::Buffer};
use arrow_array::{FixedSizeListArray, Float32Array, UInt32Array};
use arrow_schema::{DataType, Field};
use candle_core::{Device, Tensor};
use lance_arrow::FloatArray;
use lance_core::{Error, Result};
use rayon::prelude::*;

pub trait IVFCentroidRouter: Send + Sync + std::fmt::Debug {
    /// Search the index for the given vectors.
    /// Returns the indices of the nearest IVF centroids.
    fn search(&self, vectors: &FixedSizeListArray) -> Result<UInt32Array>;
}

pub struct CandleIVFCentroidRouter {
    device: Device,
    centroids: Tensor,
    nprobes: usize,

    batch_size: usize,
}

impl CandleIVFCentroidRouter {
    pub fn new(device: Device, centroids: Tensor, nprobes: usize, batch_size: usize) -> Self {
        Self {
            device,
            centroids,
            nprobes,
            batch_size,
        }
    }
}

impl Debug for CandleIVFCentroidRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleIVFCentroidRouter")
            .field("device", &self.device)
            .finish()
    }
}

impl IVFCentroidRouter for CandleIVFCentroidRouter {
    fn search(&self, vectors: &FixedSizeListArray) -> Result<UInt32Array> {
        let dimensions = vectors.value_length() as usize;
        let num_vecs = vectors.values().len() / dimensions;

        let data = vectors
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("expected Float32Array");

        let mut ids: Vec<u32> = vec![];
        ids.reserve_exact(self.nprobes * num_vecs - ids.capacity());

        // resulting the logic from our python distance impl
        // part_ids = []
        // distances = []

        // y2 = (y * y).sum(dim=1)
        // for sub_vectors in x.split(split_size):
        //     x2 = (sub_vectors * sub_vectors).sum(dim=1)
        //     xy = sub_vectors @ y.T
        //     dists = (
        //         x2.broadcast_to(y2.shape[0], x2.shape[0]).T
        //         + y2.broadcast_to(x2.shape[0], y2.shape[0])
        //         - 2 * xy
        //     )
        //     idx = torch.argmin(dists, dim=1, keepdim=True)
        //     part_ids.append(idx)
        //     distances.append(dists.take_along_dim(idx, dim=1))

        // return torch.cat(part_ids).reshape(-1), torch.cat(distances).reshape(-1)

        let y2 = self.centroids.mul(&self.centroids)?.sum(1)?;

        for start in (0..num_vecs).step_by(self.batch_size) {
            let range =
                start * dimensions..std::cmp::min(start + self.batch_size, num_vecs) * dimensions;
            let x = Tensor::from_slice(
                &data.as_slice()[range.clone()],
                (range.len() / dimensions, dimensions),
                &self.device,
            )?;

            let x2 = x.mul(&x)?.sum(1)?;
            let xy = x.matmul(&self.centroids.t()?)?;

            let dist = x2
                .broadcast_as((y2.shape().dims()[0], x2.shape().dims()[0]))?
                .t()?
                .add(&y2.broadcast_as((x2.shape().dims()[0], y2.shape().dims()[0]))?)?
                .sub(&xy)?
                .sub(&xy)?;

            // this is not ideal, candle doesn't have topk op
            // so we do it on CPU blocking
            let dist_vecs = dist.to_vec2::<f32>()?;

            let result = dist_vecs
                .par_iter()
                .flat_map(|one_row| {
                    let mut dist_vec: Vec<_> = one_row.iter().enumerate().collect();
                    dist_vec.sort_by(|x, y| x.1.partial_cmp(y.1).unwrap());

                    dist_vec
                        .iter()
                        .map(|x| x.0 as u32)
                        .take(10)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            ids.extend(result);
        }

        Ok(UInt32Array::from_iter(ids))
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow::{array::ArrayData, buffer::Buffer};
    use arrow_schema::{DataType, Field};

    use super::*;

    #[test]
    fn test_l2_partitons() {
        let value_data = ArrayData::builder(DataType::Float32)
            .len(64)
            .add_buffer(Buffer::from_slice_ref(&[0.1; 64]))
            .build()
            .unwrap();
        let list_data_type =
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 8);
        let list_data = ArrayData::builder(list_data_type)
            .len(8)
            .add_child_data(value_data.clone())
            .build()
            .unwrap();
        let list_array = FixedSizeListArray::from(list_data);

        let device = Device::new_cuda(0).unwrap();

        let centroids = Tensor::from_slice(&[0.1_f32; 64], (8, 8), &device).unwrap();

        let router = CandleIVFCentroidRouter::new(device, centroids, 1, 4);

        println!("{:?}", router.search(&list_array).unwrap());
    }
}
