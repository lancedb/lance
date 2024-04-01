// Copyright 2024 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{ops::Range, sync::Arc};

use arrow::{array::AsArray, datatypes::Float32Type};
use arrow_array::{Array, FixedSizeListArray, RecordBatch, UInt64Array, UInt8Array};
use half::f16;
use itertools::Itertools;
use lance_core::{Error, Result, ROW_ID};
use lance_file::reader::FileReader;
use lance_io::object_store::ObjectStore;
use lance_linalg::distance::{DistanceType, MetricType};
use lance_table::format::SelfDescribingFileReader;
use num_traits::FromPrimitive;
use object_store::path::Path;
use serde::{Deserialize, Serialize};
use snafu::{location, Location};

use crate::{
    vector::{
        graph::{storage::DistCalculator, VectorStorage},
        SQ_CODE_COLUMN,
    },
    IndexMetadata, INDEX_METADATA_SCHEMA_KEY,
};

use super::scale_to_u8;

pub const SQ_METADATA_KEY: &str = "lance:sq";

#[derive(Clone, Serialize, Deserialize)]
pub struct ScalarQuantizationMetadata {
    pub num_bits: u16,
    pub bounds: Range<f64>,
}

impl ScalarQuantizationMetadata {
    pub fn load(reader: &FileReader) -> Result<Self> {
        let metadata_str = reader
            .schema()
            .metadata
            .get(SQ_METADATA_KEY)
            .ok_or(Error::Index {
                message: format!(
                    "Reading SQ metadata: metadata key {} not found",
                    SQ_METADATA_KEY
                ),
                location: location!(),
            })?;
        serde_json::from_str(metadata_str).map_err(|_| Error::Index {
            message: format!("Failed to parse index metadata: {}", metadata_str),
            location: location!(),
        })
    }
}

pub struct ScalarQuantizationStorage {
    metric_type: MetricType,

    // Metadata
    num_bits: u16,
    bounds: Range<f64>,

    // Row IDs and SQ codes
    batch: RecordBatch,

    // Helper fields, references to the batch
    row_ids: Arc<UInt64Array>,
    sq_codes: Arc<FixedSizeListArray>,
}

impl ScalarQuantizationStorage {
    pub fn new(
        num_bits: u16,
        metric_type: MetricType,
        bounds: Range<f64>,
        batch: RecordBatch,
    ) -> Result<Self> {
        let row_ids = Arc::new(
            batch
                .column_by_name(ROW_ID)
                .ok_or(Error::Index {
                    message: format!("Row ID column not found in the batch"),
                    location: location!(),
                })?
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .clone(),
        );
        let sq_codes = Arc::new(
            batch
                .column_by_name(SQ_CODE_COLUMN)
                .ok_or(Error::Index {
                    message: format!("SQ code column not found in the batch"),
                    location: location!(),
                })?
                .as_fixed_size_list()
                .clone(),
        );

        Ok(Self {
            num_bits,
            metric_type,
            bounds,
            batch,
            row_ids,
            sq_codes,
        })
    }

    pub fn num_bits(&self) -> u16 {
        self.num_bits
    }

    pub fn metric_type(&self) -> MetricType {
        self.metric_type
    }

    pub fn bounds(&self) -> Range<f64> {
        self.bounds.clone()
    }

    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }

    pub fn row_ids(&self) -> &[u64] {
        self.row_ids.values()
    }

    pub fn sq_codes(&self) -> &Arc<FixedSizeListArray> {
        &self.sq_codes
    }

    /// Load a partition of SQ storage from disk.
    ///
    /// Parameters
    /// ----------
    /// - *reader: file reader
    /// - *range: row range of the partition
    /// - *metric_type: metric type of the vectors
    /// - *metadata: scalar quantization metadata
    pub async fn load_partition(
        reader: &FileReader,
        range: std::ops::Range<usize>,
        metric_type: MetricType,
        metadata: &ScalarQuantizationMetadata,
    ) -> Result<Self> {
        let schema = reader.schema();
        let batch = reader.read_range(range, schema, None).await?;

        Self::new(
            metadata.num_bits,
            metric_type,
            metadata.bounds.clone(),
            batch,
        )
    }

    pub async fn load(object_store: &ObjectStore, path: &Path) -> Result<Self> {
        let reader = FileReader::try_new_self_described(object_store, path, None).await?;
        let schema = reader.schema();

        let metadata_str = schema
            .metadata
            .get(INDEX_METADATA_SCHEMA_KEY)
            .ok_or(Error::Index {
                message: format!(
                    "Reading SQ storage: index key {} not found",
                    INDEX_METADATA_SCHEMA_KEY
                ),
                location: location!(),
            })?;
        let index_metadata: IndexMetadata =
            serde_json::from_str(metadata_str).map_err(|_| Error::Index {
                message: format!("Failed to parse index metadata: {}", metadata_str),
                location: location!(),
            })?;
        let metric_type: MetricType = MetricType::try_from(index_metadata.distance_type.as_str())?;

        let sq_matadata = ScalarQuantizationMetadata::load(&reader)?;

        Self::load_partition(&reader, 0..reader.len(), metric_type, &sq_matadata).await
    }
}

impl VectorStorage for ScalarQuantizationStorage {
    fn len(&self) -> usize {
        self.batch.num_rows()
    }

    fn row_ids(&self) -> &[u64] {
        self.row_ids.values()
    }

    /// Return the metric type of the vectors.
    fn metric_type(&self) -> MetricType {
        self.metric_type
    }

    /// Create a [DistCalculator] to compute the distance between the query.
    ///
    /// Using dist calcualtor can be more efficient as it can pre-compute some
    /// values.
    fn dist_calculator(&self, query: &[f32]) -> Box<dyn DistCalculator> {
        Box::new(SQDistCalculator::new(
            query,
            self.sq_codes.clone(),
            self.bounds.clone(),
            self.metric_type,
        ))
    }
}

struct SQDistCalculator {
    distance_type: DistanceType,
    query_sq_code: Vec<f16>,

    // flatten sq codes
    sq_codes: Arc<FixedSizeListArray>,
}

impl SQDistCalculator {
    fn new(
        query: &[f32],
        sq_codes: Arc<FixedSizeListArray>,
        bounds: Range<f64>,
        distance_type: DistanceType,
    ) -> Self {
        // TODO: support f16/f64
        let query_sq_code = scale_to_u8::<Float32Type>(query, bounds)
            .into_iter()
            .map(|v| f16::from_u8(v).unwrap())
            .collect_vec();

        Self {
            distance_type,
            query_sq_code,
            sq_codes,
        }
    }

    fn get_sq_code(&self, id: u32) -> &[u8] {
        let dim = self.sq_codes.value_length() as usize;
        let values: &[u8] = self
            .sq_codes
            .values()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .values();
        &values[id as usize * dim..(id as usize + 1) * dim]
    }
}

impl DistCalculator for SQDistCalculator {
    fn distance(&self, ids: &[u32]) -> Vec<f32> {
        ids.iter()
            .map(|&id| {
                let sq_code = self
                    .get_sq_code(id)
                    .iter()
                    .map(|v| f16::from_u8(*v).unwrap())
                    .collect_vec();
                self.distance_type.func()(&self.query_sq_code, &sq_code)
            })
            .collect()
    }
}
