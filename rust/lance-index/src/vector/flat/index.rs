// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Flat Vector Index.
//!

use std::{collections::HashSet, sync::Arc};

use arrow::array::AsArray;
use arrow_array::{Array, ArrayRef, Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use deepsize::DeepSizeOf;
use itertools::Itertools;
use lance_core::{Error, Result, ROW_ID_FIELD};
use lance_file::reader::FileReader;
use lance_linalg::distance::DistanceType;
use serde::{Deserialize, Serialize};
use snafu::{location, Location};

use crate::{
    prefilter::PreFilter,
    vector::{
        graph::{OrderedFloat, OrderedNode},
        quantizer::{Quantization, QuantizationType, Quantizer, QuantizerMetadata},
        storage::{DistCalculator, VectorStore},
        v3::subindex::IvfSubIndex,
        Query, DIST_COL,
    },
};

use super::storage::{FlatStorage, FLAT_COLUMN};

/// A Flat index is any index that stores no metadata, and
/// during query, it simply scans over the storage and returns the top k results
#[derive(Debug, Clone, Default, DeepSizeOf)]
pub struct FlatIndex {}

lazy_static::lazy_static! {
    static ref ANN_SEARCH_SCHEMA: SchemaRef = Schema::new(vec![
        Field::new(DIST_COL, DataType::Float32, true),
        ROW_ID_FIELD.clone(),
    ]).into();
}

#[derive(Default)]
pub struct FlatQueryParams {}

impl From<&Query> for FlatQueryParams {
    fn from(_: &Query) -> Self {
        Self {}
    }
}

impl IvfSubIndex for FlatIndex {
    type QueryParams = FlatQueryParams;
    type BuildParams = ();

    fn use_residual() -> bool {
        false
    }

    fn name() -> &'static str {
        "FLAT"
    }

    fn metadata_key() -> &'static str {
        "lance:flat"
    }

    fn schema() -> arrow_schema::SchemaRef {
        Schema::new(vec![Field::new("__flat_marker", DataType::UInt64, false)]).into()
    }

    fn search(
        &self,
        query: ArrayRef,
        k: usize,
        _params: Self::QueryParams,
        storage: &impl VectorStore,
        prefilter: Arc<dyn PreFilter>,
    ) -> Result<RecordBatch> {
        let dist_calc = storage.dist_calculator(query);

        let (row_ids, dists): (Vec<u64>, Vec<f32>) = match prefilter.is_empty() {
            true => (0..storage.len())
                .map(|id| OrderedNode {
                    id: id as u32,
                    dist: OrderedFloat(dist_calc.distance(id as u32)),
                })
                .sorted_unstable()
                .take(k)
                .map(
                    |OrderedNode {
                         id,
                         dist: OrderedFloat(dist),
                     }| (storage.row_id(id), dist),
                )
                .unzip(),
            false => {
                let filtered_row_ids = prefilter
                    .filter_row_ids(Box::new(storage.row_ids()))
                    .into_iter()
                    .collect::<HashSet<_>>();
                (0..storage.len())
                    .filter(|&id| !filtered_row_ids.contains(&storage.row_id(id as u32)))
                    .map(|id| OrderedNode {
                        id: id as u32,
                        dist: OrderedFloat(dist_calc.distance(id as u32)),
                    })
                    .sorted_unstable()
                    .take(k)
                    .map(
                        |OrderedNode {
                             id,
                             dist: OrderedFloat(dist),
                         }| (storage.row_id(id), dist),
                    )
                    .unzip()
            }
        };

        let (row_ids, dists) = (UInt64Array::from(row_ids), Float32Array::from(dists));

        Ok(RecordBatch::try_new(
            ANN_SEARCH_SCHEMA.clone(),
            vec![Arc::new(dists), Arc::new(row_ids)],
        )?)
    }

    fn load(_: RecordBatch) -> Result<Self> {
        Ok(Self {})
    }

    fn index_vectors(_: &impl VectorStore, _: Self::BuildParams) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {})
    }

    fn to_batch(&self) -> Result<RecordBatch> {
        Ok(RecordBatch::new_empty(Schema::empty().into()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, DeepSizeOf)]
pub struct FlatMetadata {
    pub dim: usize,
}

#[async_trait::async_trait]
impl QuantizerMetadata for FlatMetadata {
    async fn load(_: &FileReader) -> Result<Self> {
        unimplemented!("Flat will be used in new index builder which doesn't require this")
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
pub struct FlatQuantizer {
    dim: usize,
    distance_type: DistanceType,
}

impl FlatQuantizer {
    pub fn new(dim: usize, distance_type: DistanceType) -> Self {
        Self { dim, distance_type }
    }
}

impl Quantization for FlatQuantizer {
    type BuildParams = ();
    type Metadata = FlatMetadata;
    type Storage = FlatStorage;

    fn build(data: &dyn Array, distance_type: DistanceType, _: &Self::BuildParams) -> Result<Self> {
        let dim = data.as_fixed_size_list().value_length();
        Ok(Self::new(dim as usize, distance_type))
    }

    fn code_dim(&self) -> usize {
        self.dim
    }

    fn column(&self) -> &'static str {
        FLAT_COLUMN
    }

    fn from_metadata(metadata: &Self::Metadata, distance_type: DistanceType) -> Result<Quantizer> {
        Ok(Quantizer::Flat(Self {
            dim: metadata.dim,
            distance_type,
        }))
    }

    fn metadata(
        &self,
        _: Option<crate::vector::quantizer::QuantizationMetadata>,
    ) -> Result<serde_json::Value> {
        let metadata = FlatMetadata { dim: self.dim };
        Ok(serde_json::to_value(metadata)?)
    }

    fn metadata_key() -> &'static str {
        "flat"
    }

    fn quantization_type() -> QuantizationType {
        QuantizationType::Flat
    }

    fn quantize(&self, vectors: &dyn Array) -> Result<ArrayRef> {
        Ok(vectors.slice(0, vectors.len()))
    }
}

impl From<FlatQuantizer> for Quantizer {
    fn from(value: FlatQuantizer) -> Self {
        Self::Flat(value)
    }
}

impl TryFrom<Quantizer> for FlatQuantizer {
    type Error = Error;

    fn try_from(value: Quantizer) -> Result<Self> {
        match value {
            Quantizer::Flat(quantizer) => Ok(quantizer),
            _ => Err(Error::invalid_input(
                "quantizer is not FlatQuantizer",
                location!(),
            )),
        }
    }
}
