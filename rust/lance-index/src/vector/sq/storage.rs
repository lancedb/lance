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

use arrow_array::{RecordBatch, UInt64Array, UInt8Array};
use lance_core::{Error, Result, ROW_ID};
use lance_file::reader::FileReader;
use lance_io::object_store::ObjectStore;
use lance_linalg::distance::MetricType;
use lance_table::format::SelfDescribingFileReader;
use object_store::path::Path;
use serde::{Deserialize, Serialize};
use snafu::{location, Location};

use crate::{vector::SQ_CODE_COLUMN, IndexMetadata, INDEX_METADATA_SCHEMA_KEY};

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
    sq_codes: Arc<UInt8Array>,
}

impl ScalarQuantizationStorage {
    pub fn new(
        num_bits: u16,
        metric_type: MetricType,
        bounds: Range<f64>,
        batch: RecordBatch,
    ) -> Self {
        let row_ids = Arc::new(
            batch
                .column_by_name(ROW_ID)
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .clone(),
        );
        let sq_codes = Arc::new(
            batch
                .column_by_name(SQ_CODE_COLUMN)
                .unwrap()
                .as_any()
                .downcast_ref::<UInt8Array>()
                .unwrap()
                .clone(),
        );

        Self {
            num_bits,
            metric_type,
            bounds,
            batch,
            row_ids,
            sq_codes,
        }
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

    pub fn row_ids(&self) -> &Arc<UInt64Array> {
        &self.row_ids
    }

    pub fn sq_codes(&self) -> &Arc<UInt8Array> {
        &self.sq_codes
    }

    /// Load a partition of SQ storage from disk.
    ///
    /// Parameters
    /// ----------
    /// - *reader: &FileReader
    pub async fn load_partition(
        reader: &FileReader,
        range: std::ops::Range<usize>,
        metric_type: MetricType,
        metadata: &ScalarQuantizationMetadata,
    ) -> Result<Self> {
        let schema = reader.schema();
        let batch = reader.read_range(range, schema, None).await?;

        Ok(Self::new(
            metadata.num_bits,
            metric_type,
            metadata.bounds.clone(),
            batch,
        ))
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
