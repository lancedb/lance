// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Flat Vector Index.
//!

use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use arrow::array::AsArray;
use arrow_array::{Array, ArrayRef, Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use deepsize::DeepSizeOf;
use lance_core::{Error, Result, ROW_ID_FIELD};
use lance_file::reader::FileReader;
use lance_linalg::distance::DistanceType;
use serde::{Deserialize, Serialize};
use snafu::location;

use crate::{
    metrics::MetricsCollector,
    prefilter::PreFilter,
    vector::{
        graph::OrderedNode,
        quantizer::{Quantization, QuantizationType, Quantizer, QuantizerMetadata},
        storage::{DistCalculator, VectorStore},
        v3::subindex::IvfSubIndex,
        Query, DIST_COL,
    },
};

use super::storage::{FlatBinStorage, FlatFloatStorage, FLAT_COLUMN};

/// A Flat index is any index that stores no metadata, and
/// during query, it simply scans over the storage and returns the top k results
#[derive(Debug, Clone, Default, DeepSizeOf)]
pub struct FlatIndex {}

use std::sync::LazyLock;

static ANN_SEARCH_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Schema::new(vec![
        Field::new(DIST_COL, DataType::Float32, true),
        ROW_ID_FIELD.clone(),
    ])
    .into()
});

#[derive(Default)]
pub struct FlatQueryParams {
    lower_bound: Option<f32>,
    upper_bound: Option<f32>,
}

impl From<&Query> for FlatQueryParams {
    fn from(q: &Query) -> Self {
        Self {
            lower_bound: q.lower_bound,
            upper_bound: q.upper_bound,
        }
    }
}

impl IvfSubIndex for FlatIndex {
    type QueryParams = FlatQueryParams;
    type BuildParams = ();

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
        params: Self::QueryParams,
        storage: &impl VectorStore,
        prefilter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        let is_range_query = params.lower_bound.is_some() || params.upper_bound.is_some();
        let row_ids = storage.row_ids();
        let dist_calc = storage.dist_calculator(query);
        let mut res = BinaryHeap::with_capacity(k);
        metrics.record_comparisons(storage.len());

        match prefilter.is_empty() {
            true => {
                let dists = dist_calc.distance_all(k);

                if is_range_query {
                    let lower_bound = params.lower_bound.unwrap_or(f32::MIN).into();
                    let upper_bound = params.upper_bound.unwrap_or(f32::MAX).into();

                    for (&row_id, dist) in row_ids.zip(dists) {
                        let dist = dist.into();
                        if dist < lower_bound || dist >= upper_bound {
                            continue;
                        }
                        if res.len() < k {
                            res.push(OrderedNode::new(row_id, dist));
                        } else if res.peek().unwrap().dist > dist {
                            res.pop();
                            res.push(OrderedNode::new(row_id, dist));
                        }
                    }
                } else {
                    for (&row_id, dist) in row_ids.zip(dists) {
                        let dist = dist.into();
                        if res.len() < k {
                            res.push(OrderedNode::new(row_id, dist));
                        } else if res.peek().unwrap().dist > dist {
                            res.pop();
                            res.push(OrderedNode::new(row_id, dist));
                        }
                    }
                }
            }
            false => {
                let row_id_mask = prefilter.mask();
                if is_range_query {
                    let lower_bound = params.lower_bound.unwrap_or(f32::MIN).into();
                    let upper_bound = params.upper_bound.unwrap_or(f32::MAX).into();
                    for (id, &row_id) in row_ids.enumerate() {
                        if !row_id_mask.selected(row_id) {
                            continue;
                        }
                        let dist = dist_calc.distance(id as u32).into();
                        if dist < lower_bound || dist >= upper_bound {
                            continue;
                        }

                        if res.len() < k {
                            res.push(OrderedNode::new(row_id, dist));
                        } else if res.peek().unwrap().dist > dist {
                            res.pop();
                            res.push(OrderedNode::new(row_id, dist));
                        }
                    }
                } else {
                    for (id, &row_id) in row_ids.enumerate() {
                        if !row_id_mask.selected(row_id) {
                            continue;
                        }

                        let dist = dist_calc.distance(id as u32).into();
                        if res.len() < k {
                            res.push(OrderedNode::new(row_id, dist));
                        } else if res.peek().unwrap().dist > dist {
                            res.pop();
                            res.push(OrderedNode::new(row_id, dist));
                        }
                    }
                }
            }
        };

        // we don't need to sort the results by distances here
        // because there's a SortExec node in the query plan which sorts the results from all partitions
        let (row_ids, dists): (Vec<_>, Vec<_>) = res.into_iter().map(|r| (r.id, r.dist.0)).unzip();
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

    fn remap(&self, _: &HashMap<u64, Option<u64>>) -> Result<Self> {
        Ok(self.clone())
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
    type Storage = FlatFloatStorage;

    fn build(data: &dyn Array, distance_type: DistanceType, _: &Self::BuildParams) -> Result<Self> {
        let dim = data.as_fixed_size_list().value_length();
        Ok(Self::new(dim as usize, distance_type))
    }

    fn retrain(&mut self, _: &dyn Array) -> Result<()> {
        Ok(())
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

    fn metadata(&self, _: Option<crate::vector::quantizer::QuantizationMetadata>) -> FlatMetadata {
        FlatMetadata { dim: self.dim }
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

    fn field(&self) -> Field {
        Field::new(
            FLAT_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                self.dim as i32,
            ),
            true,
        )
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

#[derive(Debug, Clone, DeepSizeOf)]
pub struct FlatBinQuantizer {
    dim: usize,
    distance_type: DistanceType,
}

impl FlatBinQuantizer {
    pub fn new(dim: usize, distance_type: DistanceType) -> Self {
        Self { dim, distance_type }
    }
}

impl Quantization for FlatBinQuantizer {
    type BuildParams = ();
    type Metadata = FlatMetadata;
    type Storage = FlatBinStorage;

    fn build(data: &dyn Array, distance_type: DistanceType, _: &Self::BuildParams) -> Result<Self> {
        let dim = data.as_fixed_size_list().value_length();
        Ok(Self::new(dim as usize, distance_type))
    }

    fn retrain(&mut self, _: &dyn Array) -> Result<()> {
        Ok(())
    }

    fn code_dim(&self) -> usize {
        self.dim
    }

    fn column(&self) -> &'static str {
        FLAT_COLUMN
    }

    fn from_metadata(metadata: &Self::Metadata, distance_type: DistanceType) -> Result<Quantizer> {
        Ok(Quantizer::FlatBin(Self {
            dim: metadata.dim,
            distance_type,
        }))
    }

    fn metadata(&self, _: Option<crate::vector::quantizer::QuantizationMetadata>) -> FlatMetadata {
        FlatMetadata { dim: self.dim }
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

    fn field(&self) -> Field {
        Field::new(
            FLAT_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                self.dim as i32,
            ),
            true,
        )
    }
}

impl From<FlatBinQuantizer> for Quantizer {
    fn from(value: FlatBinQuantizer) -> Self {
        Self::FlatBin(value)
    }
}

impl TryFrom<Quantizer> for FlatBinQuantizer {
    type Error = Error;

    fn try_from(value: Quantizer) -> Result<Self> {
        match value {
            Quantizer::FlatBin(quantizer) => Ok(quantizer),
            _ => Err(Error::invalid_input(
                "quantizer is not FlatBinQuantizer",
                location!(),
            )),
        }
    }
}
