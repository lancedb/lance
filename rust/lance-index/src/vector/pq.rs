// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Product Quantization
//!

use std::sync::Arc;

use arrow::datatypes::{self, ArrowPrimitiveType, Float16Type, Float32Type, Float64Type};
use arrow_array::{cast::AsArray, Array, FixedSizeListArray, UInt8Array};
use arrow_array::{ArrayRef, Float32Array, PrimitiveArray};
use arrow_schema::DataType;
use deepsize::DeepSizeOf;
use lance_arrow::*;
use lance_core::{Error, Result};
use lance_linalg::distance::{dot_distance_batch, DistanceType, Dot, L2};
use lance_linalg::kmeans::compute_partition;
use lance_linalg::MatrixView;
use num_traits::Float;
use prost::Message;
use rayon::prelude::*;
use snafu::{location, Location};
use storage::{ProductQuantizationMetadata, ProductQuantizationStorage, PQ_METADTA_KEY};

pub mod builder;
mod distance;
pub mod storage;
pub mod transform;
pub(crate) mod utils;

use self::distance::{build_distance_table_l2, compute_l2_distance};
pub use self::utils::num_centroids;
use super::quantizer::{Quantization, QuantizationMetadata, QuantizationType, Quantizer};
use super::{pb, PQ_CODE_COLUMN};
pub use builder::PQBuildParams;
use utils::get_sub_vector_centroids;

#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    pub num_sub_vectors: usize,
    pub num_bits: u32,
    pub dimension: usize,
    pub codebook: FixedSizeListArray,
    pub distance_type: DistanceType,
}

impl DeepSizeOf for ProductQuantizer {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.codebook.get_array_memory_size()
            + self.num_sub_vectors.deep_size_of_children(_context)
            + self.num_bits.deep_size_of_children(_context)
            + self.dimension.deep_size_of_children(_context)
            + self.distance_type.deep_size_of_children(_context)
    }
}

impl ProductQuantizer {
    pub fn new(
        num_sub_vectors: usize,
        num_bits: u32,
        dimension: usize,
        codebook: FixedSizeListArray,
        distance_type: DistanceType,
    ) -> Self {
        Self {
            num_bits,
            num_sub_vectors,
            dimension,
            codebook,
            distance_type,
        }
    }

    pub fn from_proto(proto: &pb::Pq, distance_type: DistanceType) -> Result<Self> {
        let codebook = match proto.codebook_tensor.as_ref() {
            Some(tensor) => FixedSizeListArray::try_from(tensor)?,
            None => FixedSizeListArray::try_new_from_values(
                Float32Array::from(proto.codebook.clone()),
                proto.dimension as i32,
            )?,
        };
        Ok(Self {
            num_bits: proto.num_bits,
            num_sub_vectors: proto.num_sub_vectors as usize,
            dimension: proto.dimension as usize,
            codebook,
            distance_type,
        })
    }

    pub fn use_residual(&self) -> bool {
        matches!(self.distance_type, DistanceType::L2 | DistanceType::Cosine)
    }

    fn transform<T: ArrowPrimitiveType>(&self, vectors: &dyn Array) -> Result<ArrayRef>
    where
        T::Native: Float + L2 + Dot,
    {
        let fsl = vectors.as_fixed_size_list_opt().ok_or(Error::Index {
            message: format!(
                "Expect to be a FixedSizeList<float> vector array, got: {:?} array",
                vectors.data_type()
            ),
            location: location!(),
        })?;
        let num_sub_vectors = self.num_sub_vectors;
        let dim = self.dimension;
        let num_bits = self.num_bits;
        let codebook = self.codebook.values().as_primitive::<T>();

        let distance_type = self.distance_type;

        let flatten_data = fsl.values().as_primitive::<T>();
        let sub_dim = dim / num_sub_vectors;
        let values = flatten_data
            .values()
            .par_chunks(dim)
            .map(|vector| {
                vector
                    .chunks_exact(sub_dim)
                    .enumerate()
                    .map(|(sub_idx, sub_vector)| {
                        let centroids = get_sub_vector_centroids(
                            codebook.values(),
                            dim,
                            num_bits,
                            num_sub_vectors,
                            sub_idx,
                        );
                        compute_partition(centroids, sub_vector, distance_type).map(|v| v as u8)
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect::<Vec<_>>();

        Ok(Arc::new(FixedSizeListArray::try_new_from_values(
            UInt8Array::from(values),
            self.num_sub_vectors as i32,
        )?))
    }

    pub fn compute_distances(&self, query: &dyn Array, code: &UInt8Array) -> Result<Float32Array> {
        match self.distance_type {
            DistanceType::L2 => self.l2_distances(query, code),
            DistanceType::Cosine => {
                // L2 over normalized vectors:  ||x - y|| = x^2 + y^2 - 2 * xy = 1 + 1 - 2 * xy = 2 * (1 - xy)
                // Cosine distance: 1 - |xy| / (||x|| * ||y||) = 1 - xy / (x^2 * y^2) = 1 - xy / (1 * 1) = 1 - xy
                // Therefore, Cosine = L2 / 2
                let l2_dists = self.l2_distances(query, code)?;
                Ok(l2_dists.values().iter().map(|v| *v / 2.0).collect())
            }
            DistanceType::Dot => self.dot_distances(query, code),
            _ => panic!(
                "ProductQuantization: distance type {} not supported",
                self.distance_type
            ),
        }
    }

    /// Pre-compute L2 distance from the query to all code.
    ///
    /// It returns the squared L2 distance.
    fn l2_distances(&self, key: &dyn Array, code: &UInt8Array) -> Result<Float32Array> {
        let distance_table = self.build_l2_distance_table(key)?;

        #[cfg(target_feature = "avx512f")]
        {
            Ok(self.compute_l2_distance::<16, 64>(&distance_table, code.values()))
        }
        #[cfg(not(target_feature = "avx512f"))]
        {
            Ok(self.compute_l2_distance::<8, 64>(&distance_table, code.values()))
        }
    }

    /// Parameters
    /// ----------
    ///  - query: the query vector, with shape (dimension, )
    ///  - code: the PQ code in one partition.
    ///
    fn dot_distances(&self, key: &dyn Array, code: &UInt8Array) -> Result<Float32Array> {
        match key.data_type() {
            DataType::Float16 => {
                self.dot_distances_impl::<datatypes::Float16Type>(key.as_primitive(), code)
            }
            DataType::Float32 => {
                self.dot_distances_impl::<datatypes::Float32Type>(key.as_primitive(), code)
            }
            DataType::Float64 => {
                self.dot_distances_impl::<datatypes::Float64Type>(key.as_primitive(), code)
            }
            _ => Err(Error::Index {
                message: format!("unsupported data type: {}", key.data_type()),
                location: location!(),
            }),
        }
    }

    fn dot_distances_impl<T: ArrowPrimitiveType>(
        &self,
        key: &PrimitiveArray<T>,
        code: &UInt8Array,
    ) -> Result<Float32Array>
    where
        T::Native: Dot,
    {
        let capacity = self.num_sub_vectors * num_centroids(self.num_bits);
        let mut distance_table = Vec::with_capacity(capacity);

        let sub_vector_length = self.dimension / self.num_sub_vectors;
        key.values()
            .chunks_exact(sub_vector_length)
            .enumerate()
            .for_each(|(sub_vec_id, sub_vec)| {
                let subvec_centroids = self.centroids::<T>(sub_vec_id);
                let distances = dot_distance_batch(sub_vec, subvec_centroids, sub_vector_length);
                distance_table.extend(distances);
            });

        // Compute distance from the pre-compute table.
        Ok(Float32Array::from_iter_values(
            code.values().chunks_exact(self.num_sub_vectors).map(|c| {
                c.iter()
                    .enumerate()
                    .map(|(sub_vec_idx, centroid)| {
                        distance_table[sub_vec_idx * 256 + *centroid as usize]
                    })
                    .sum::<f32>()
            }),
        ))
    }

    fn build_l2_distance_table(&self, key: &dyn Array) -> Result<Vec<f32>> {
        match key.data_type() {
            DataType::Float16 => {
                Ok(self.build_l2_distance_table_impl::<datatypes::Float16Type>(key.as_primitive()))
            }
            DataType::Float32 => {
                Ok(self.build_l2_distance_table_impl::<datatypes::Float32Type>(key.as_primitive()))
            }
            DataType::Float64 => {
                Ok(self.build_l2_distance_table_impl::<datatypes::Float64Type>(key.as_primitive()))
            }
            _ => Err(Error::Index {
                message: format!("unsupported data type: {}", key.data_type()),
                location: location!(),
            }),
        }
    }

    fn build_l2_distance_table_impl<T: ArrowPrimitiveType>(
        &self,
        key: &PrimitiveArray<T>,
    ) -> Vec<f32>
    where
        T::Native: L2,
    {
        build_distance_table_l2(
            self.codebook.values().as_primitive::<T>().values(),
            self.num_bits,
            self.num_sub_vectors,
            key.values(),
        )
    }

    /// Compute L2 distance from the query to all code.
    ///
    /// Type parameters
    /// ---------------
    /// - C: the tile size of code-book to run at once.
    /// - V: the tile size of PQ code to run at once.
    ///
    /// Parameters
    /// ----------
    /// - distance_table: the pre-computed L2 distance table.
    ///   It is a flatten array of [num_sub_vectors, num_centroids] f32.
    /// - code: the PQ code to be used to compute the distances.
    ///
    /// Returns
    /// -------
    ///  The squared L2 distance.
    #[inline]
    fn compute_l2_distance<const C: usize, const V: usize>(
        &self,
        distance_table: &[f32],
        code: &[u8],
    ) -> Float32Array {
        Float32Array::from(compute_l2_distance::<C, V>(
            distance_table,
            self.num_bits,
            self.num_sub_vectors,
            code,
        ))
    }

    /// Get the centroids for one sub-vector.
    ///
    /// Returns a flatten `num_centroids * sub_vector_width` f32 array.
    pub fn centroids<T: ArrowPrimitiveType>(&self, sub_vector_idx: usize) -> &[T::Native] {
        get_sub_vector_centroids(
            self.codebook.values().as_primitive::<T>().values(),
            self.dimension,
            self.num_bits,
            self.num_sub_vectors,
            sub_vector_idx,
        )
    }
}

impl Quantization for ProductQuantizer {
    type BuildParams = PQBuildParams;
    type Metadata = ProductQuantizationMetadata;
    type Storage = ProductQuantizationStorage;

    fn build(
        data: &dyn Array,
        distance_type: DistanceType,
        params: &Self::BuildParams,
    ) -> Result<Self> {
        assert_eq!(data.null_count(), 0);
        let fsl = data.as_fixed_size_list_opt().ok_or(Error::Index {
            message: format!(
                "PQ builder: input is not a FixedSizeList: {}",
                data.data_type()
            ),
            location: location!(),
        })?;

        if let Some(codebook) = params.codebook.as_ref() {
            return Ok(Self::new(
                params.num_sub_vectors,
                params.num_bits as u32,
                fsl.value_length() as usize,
                FixedSizeListArray::try_new_from_values(codebook.clone(), fsl.value_length())?,
                distance_type,
            ));
        }

        // TODO: support bf16 later.
        match fsl.value_type() {
            DataType::Float16 => {
                let data = MatrixView::<Float16Type>::try_from(fsl)?;
                params.build_from_matrix(&data, distance_type)
            }
            DataType::Float32 => {
                let data = MatrixView::<Float32Type>::try_from(fsl)?;
                params.build_from_matrix(&data, distance_type)
            }
            DataType::Float64 => {
                let data = MatrixView::<Float64Type>::try_from(fsl)?;
                params.build_from_matrix(&data, distance_type)
            }
            _ => Err(Error::Index {
                message: format!("PQ builder: unsupported data type: {}", fsl.value_type()),
                location: location!(),
            }),
        }
    }

    fn code_dim(&self) -> usize {
        self.num_sub_vectors
    }

    fn column(&self) -> &'static str {
        PQ_CODE_COLUMN
    }

    fn quantize(&self, vectors: &dyn Array) -> Result<ArrayRef> {
        let fsl = vectors.as_fixed_size_list_opt().ok_or(Error::Index {
            message: format!(
                "Expect to be a FixedSizeList<float> vector array, got: {:?} array",
                vectors.data_type()
            ),
            location: location!(),
        })?;

        match fsl.value_type() {
            DataType::Float16 => self.transform::<datatypes::Float16Type>(vectors),
            DataType::Float32 => self.transform::<datatypes::Float32Type>(vectors),
            DataType::Float64 => self.transform::<datatypes::Float64Type>(vectors),
            _ => Err(Error::Index {
                message: format!("unsupported data type: {}", fsl.value_type()),
                location: location!(),
            }),
        }
    }

    fn metadata_key() -> &'static str {
        PQ_METADTA_KEY
    }

    fn quantization_type() -> QuantizationType {
        QuantizationType::Product
    }

    fn metadata(&self, args: Option<QuantizationMetadata>) -> Result<serde_json::Value> {
        let codebook_position = match args {
            Some(args) => args.codebook_position,
            None => Some(0),
        };
        let codebook_position = codebook_position.ok_or(Error::Index {
            message: "codebook_position not found".to_owned(),
            location: location!(),
        })?;
        let tensor = pb::Tensor::try_from(&self.codebook)?;
        Ok(serde_json::to_value(ProductQuantizationMetadata {
            codebook_position,
            num_bits: self.num_bits,
            num_sub_vectors: self.num_sub_vectors,
            dimension: self.dimension,
            codebook: None,
            codebook_tensor: tensor.encode_to_vec(),
        })?)
    }

    fn from_metadata(metadata: &Self::Metadata, distance_type: DistanceType) -> Result<Quantizer> {
        Ok(Quantizer::Product(Self::new(
            metadata.num_sub_vectors,
            metadata.num_bits,
            metadata.dimension,
            metadata.codebook.as_ref().unwrap().clone(),
            distance_type,
        )))
    }
}

impl TryFrom<&ProductQuantizer> for pb::Pq {
    type Error = Error;

    fn try_from(pq: &ProductQuantizer) -> Result<Self> {
        let tensor = pb::Tensor::try_from(&pq.codebook)?;
        Ok(Self {
            num_bits: pq.num_bits,
            num_sub_vectors: pq.num_sub_vectors as u32,
            dimension: pq.dimension as u32,
            codebook: vec![],
            codebook_tensor: Some(tensor),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::iter::repeat;

    use approx::assert_relative_eq;
    use arrow::datatypes::UInt8Type;
    use arrow_array::Float16Array;
    use half::f16;
    use lance_linalg::distance::l2_distance_batch;
    use lance_linalg::kernels::argmin;
    use lance_testing::datagen::generate_random_array;
    use num_traits::Zero;

    #[test]
    fn test_f16_pq_to_protobuf() {
        let pq = ProductQuantizer::new(
            4,
            8,
            16,
            FixedSizeListArray::try_new_from_values(
                Float16Array::from_iter_values(repeat(f16::zero()).take(256 * 16)),
                16,
            )
            .unwrap(),
            DistanceType::L2,
        );
        let proto: pb::Pq = pb::Pq::try_from(&pq).unwrap();
        assert_eq!(proto.num_bits, 8);
        assert_eq!(proto.num_sub_vectors, 4);
        assert_eq!(proto.dimension, 16);
        assert!(proto.codebook.is_empty());
        assert!(proto.codebook_tensor.is_some());

        let tensor = proto.codebook_tensor.as_ref().unwrap();
        assert_eq!(tensor.data_type, pb::tensor::DataType::Float16 as i32);
        assert_eq!(tensor.shape, vec![256, 16]);
    }

    #[test]
    fn test_l2_distance() {
        const DIM: usize = 512;
        const TOTAL: usize = 66; // 64 + 2 to make sure reminder is handled correctly.
        let codebook = generate_random_array(256 * DIM);
        let pq = ProductQuantizer::new(
            16,
            8,
            DIM,
            FixedSizeListArray::try_new_from_values(codebook, DIM as i32).unwrap(),
            DistanceType::L2,
        );
        let pq_code = UInt8Array::from_iter_values((0..16 * TOTAL).map(|v| v as u8));
        let query = generate_random_array(DIM);

        let dists = pq.compute_distances(&query, &pq_code).unwrap();

        let sub_vec_len = DIM / 16;
        let expected = pq_code
            .values()
            .chunks(16)
            .map(|code| {
                code.iter()
                    .enumerate()
                    .flat_map(|(sub_idx, c)| {
                        let subvec_centroids = pq.centroids::<datatypes::Float32Type>(sub_idx);
                        let subvec =
                            &query.values()[sub_idx * sub_vec_len..(sub_idx + 1) * sub_vec_len];
                        l2_distance_batch(
                            subvec,
                            &subvec_centroids
                                [*c as usize * sub_vec_len..(*c as usize + 1) * sub_vec_len],
                            sub_vec_len,
                        )
                    })
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        dists
            .values()
            .iter()
            .zip(expected.iter())
            .for_each(|(v, e)| {
                assert_relative_eq!(*v, *e, epsilon = 1e-4);
            });
    }

    #[test]
    fn test_pq_transform() {
        const DIM: usize = 16;
        const TOTAL: usize = 64;
        let codebook = generate_random_array(DIM * 256);
        let pq = ProductQuantizer::new(
            4,
            8,
            DIM,
            FixedSizeListArray::try_new_from_values(codebook, DIM as i32).unwrap(),
            DistanceType::L2,
        );

        let vectors = generate_random_array(DIM * TOTAL);
        let fsl = FixedSizeListArray::try_new_from_values(vectors.clone(), DIM as i32).unwrap();
        let pq_code = pq.quantize(&fsl).unwrap();

        let mut expected = Vec::with_capacity(TOTAL * 4);
        vectors.values().chunks_exact(DIM).for_each(|vec| {
            vec.chunks_exact(DIM / 4)
                .enumerate()
                .for_each(|(sub_idx, sub_vec)| {
                    let centroids = pq.centroids::<datatypes::Float32Type>(sub_idx);
                    let dists = l2_distance_batch(sub_vec, centroids, DIM / 4);
                    let code = argmin(dists).unwrap() as u8;
                    expected.push(code);
                });
        });

        assert_eq!(pq_code.len(), TOTAL);
        assert_eq!(
            &expected,
            pq_code
                .as_fixed_size_list()
                .values()
                .as_primitive::<UInt8Type>()
                .values()
        );
    }
}
