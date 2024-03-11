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

use std::sync::Arc;

use arrow_array::{Array, FixedSizeListArray, Float32Array};

use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};
use lance_file::{reader::FileReader, writer::FileWriter};
use lance_io::{traits::WriteExt, utils::read_message};
use lance_table::io::manifest::ManifestDescribing;
use log::debug;
use serde::{Deserialize, Serialize};
use serde_json::json;
use snafu::{location, Location};

use crate::pb::Ivf as PbIvf;

const IVF_METADATA_KEY: &str = "lance:ivf";

struct IvfMetadadta {
    /// Centroids of the IVF indices. Can be empty.
    centroids: Arc<FixedSizeListArray>,

    /// Length of each partition.
    lengths: Vec<u32>,

    /// pre-computed row offset for each partition, do not persist.
    partition_row_offsets: Vec<usize>,
}

/// The IVF metadata stored in the Lance Schema
#[derive(Serialize, Deserialize, Debug)]
struct IvfMetadataInSchema {
    // The file position to store the protobuf binary of IVF metadata.
    pb_position: usize,
}

impl IvfMetadadta {
    pub fn empty(dim: usize) -> Self {
        let values = Float32Array::from(Vec::<f32>::new());
        Self {
            centroids: Arc::new(
                FixedSizeListArray::try_new_from_values(values, dim as i32)
                    .expect("Creating empty centroids"),
            ),
            lengths: vec![],
            partition_row_offsets: vec![0],
        }
    }

    pub async fn load(reader: &FileReader) -> Result<Self> {
        let schema = reader.schema();
        let meta_str = schema.metadata.get(IVF_METADATA_KEY).ok_or(Error::Index {
            message: format!("{} not found during search", IVF_METADATA_KEY),
            location: location!(),
        })?;
        let ivf_metadata: IvfMetadataInSchema =
            serde_json::from_str(meta_str).map_err(|e| Error::Index {
                message: format!("Failed to parse IVF metadata: {}", e),
                location: location!(),
            })?;

        let pb: PbIvf = read_message(
            reader.object_reader.as_ref(),
            ivf_metadata.pb_position as usize,
        )
        .await?;
        IvfMetadadta::try_from(pb)
    }

    /// Write the IVF metadata to the lance file.
    pub async fn write(&self, writer: &mut FileWriter<ManifestDescribing>) -> Result<()> {
        let pb = PbIvf::try_from(self)?;
        let pos = writer.object_writer.write_protobuf(&pb).await?;
        let ivf_metadata = IvfMetadataInSchema { pb_position: pos };
        writer.add_metadata(IVF_METADATA_KEY, &serde_json::to_string(&ivf_metadata)?);
        Ok(())
    }

    pub fn add_partition(&mut self, num_rows: u32) {
        self.lengths.push(num_rows);
        let last_offset = self.partition_row_offsets.last().copied().unwrap_or(0);
        self.partition_row_offsets
            .push(last_offset + num_rows as usize);
    }

    pub fn has_centroids(&self) -> bool {
        self.centroids.len() > 0
    }

    pub fn num_partitions(&self) -> usize {
        self.lengths.len()
    }

    /// Range of the rows for one partition.
    pub fn row_range(&self, partition: usize) -> (usize, usize) {
        let start = self.partition_row_offsets[partition];
        let end = self.partition_row_offsets[partition + 1];
        (start, end)
    }
}

impl TryFrom<PbIvf> for IvfMetadadta {
    type Error = Error;

    fn try_from(proto: PbIvf) -> Result<Self> {
        let centroids = if let Some(tensor) = proto.centroids_tensor.as_ref() {
            debug!("Ivf: loading IVF centroids from index format v2");
            Arc::new(FixedSizeListArray::try_from(tensor)?)
        } else {
            return Err(Error::Schema {
                message: "Centroids tensor not found".to_string(),
                location: location!(),
            });
        };
        // We are not using offsets from the protobuf, which was the file offset in the
        // v1 index format. It will be deprecated soon.
        //
        // This new offset uses the row offset in the lance file.
        let offsets = proto.lengths.iter().scan(0, |state, &x| {
            let old = *state;
            *state += x as usize;
            Some(old)
        });
        Ok(Self {
            centroids,
            lengths: proto.lengths.clone(),
            partition_row_offsets: offsets.collect(),
        })
    }
}

impl TryFrom<&IvfMetadadta> for PbIvf {
    type Error = Error;

    fn try_from(meta: &IvfMetadadta) -> Result<Self> {
        let lengths = meta.lengths.clone();

        Ok(Self {
            centroids: vec![], // Deprecated
            lengths,
            offsets: vec![], // Deprecated
            centroids_tensor: Some(meta.centroids.as_ref().try_into()?),
        })
    }
}

#[cfg(test)]
mod tests {}
