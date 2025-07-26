// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use async_trait::async_trait;
use deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use lance_file::reader::FileReader;
use serde::{Deserialize, Serialize};
use snafu::location;

use crate::vector::flat::storage::FlatBinStorage;
use crate::vector::quantizer::{QuantizerMetadata, QuantizerStorage};
use crate::vector::storage::VectorStore;

pub const RABBIT_METADATA_KEY: &str = "lance:rabbit";

#[derive(Debug, Clone, Serialize, Deserialize, DeepSizeOf)]
pub struct RabbitQuantizationMetadata {
    num_bits: u16,
    dim: usize,
}

impl RabbitQuantizationMetadata {
    pub fn new(num_bits: u16, dim: usize) -> Self {
        Self { num_bits, dim }
    }
}

#[async_trait]
impl QuantizerMetadata for RabbitQuantizationMetadata {
    async fn load(reader: &FileReader) -> Result<Self> {
        let metadata_str =
            reader
                .schema()
                .metadata
                .get(RABBIT_METADATA_KEY)
                .ok_or(Error::Index {
                    message: format!(
                        "Reading Rabbit metadata: metadata key {} not found",
                        RABBIT_METADATA_KEY
                    ),
                    location: location!(),
                })?;
        serde_json::from_str(metadata_str).map_err(|_| Error::Index {
            message: format!("Failed to parse index metadata: {}", metadata_str),
            location: location!(),
        })
    }
}

pub struct RabbitQuantizationStorage {
    metadata: RabbitQuantizationMetadata,
    storage: FlatBinStorage,
}

pub struct RabbitDistCalculator<'a> {
    query_codes: Vec<u8>,
    storage: &'a RabbitQuantizationStorage,
}

impl VectorStore for RabbitQuantizationStorage {
    type DistanceCalculator<'a> = RabbitDistCalculator<'a>;
}

impl QuantizerStorage for RabbitQuantizationStorage {
    type Metadata = RabbitQuantizationMetadata;

    fn try_from_batch(
        batch: RecordBatch,
        metadata: &Self::Metadata,
        distance_type: DistanceType,
        fri: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        Ok(Self {
            metadata: metadata.clone(),
            storage: FlatBinStorage::try_from_batch(batch, metadata, distance_type, fri)?,
        })
    }
}
