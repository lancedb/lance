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

use std::ops::Range;
use std::sync::Arc;

use arrow_array::{Array, FixedSizeListArray, Float32Array};

use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};
use lance_file::{reader::FileReader, writer::FileWriter};
use lance_io::{traits::WriteExt, utils::read_message};
use lance_table::io::manifest::ManifestDescribing;
use log::debug;
use serde::{Deserialize, Serialize};
use snafu::{location, Location};

use crate::pb::Ivf as PbIvf;

const IVF_METADATA_KEY: &str = "lance:ivf";

#[warn(dead_code)]
#[derive(Debug, PartialEq)]
pub struct IvfData {
    /// Centroids of the IVF indices. Can be empty.
    centroids: Arc<FixedSizeListArray>,

    /// Length of each partition.
    lengths: Vec<u32>,

    /// pre-computed row offset for each partition, do not persist.
    partition_row_offsets: Vec<usize>,
}

/// The IVF metadata stored in the Lance Schema
#[derive(Serialize, Deserialize, Debug)]
struct IvfMetadata {
    // The file position to store the protobuf binary of IVF metadata.
    pb_position: usize,
}

impl IvfData {
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
        let ivf_metadata: IvfMetadata =
            serde_json::from_str(meta_str).map_err(|e| Error::Index {
                message: format!("Failed to parse IVF metadata: {}", e),
                location: location!(),
            })?;

        let pb: PbIvf = read_message(
            reader.object_reader.as_ref(),
            ivf_metadata.pb_position as usize,
        )
        .await?;
        Self::try_from(pb)
    }

    /// Write the IVF metadata to the lance file.
    pub async fn write(&self, writer: &mut FileWriter<ManifestDescribing>) -> Result<()> {
        let pb = PbIvf::try_from(self)?;
        let pos = writer.object_writer.write_protobuf(&pb).await?;
        let ivf_metadata = IvfMetadata { pb_position: pos };
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
    pub fn row_range(&self, partition: usize) -> Range<usize> {
        let start = self.partition_row_offsets[partition];
        let end = self.partition_row_offsets[partition + 1];
        start..end
    }
}

impl TryFrom<PbIvf> for IvfData {
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
        let offsets = [0]
            .iter()
            .chain(proto.lengths.iter())
            .scan(0_usize, |state, &x| {
                *state += x as usize;
                Some(*state)
            });
        Ok(Self {
            centroids,
            lengths: proto.lengths.clone(),
            partition_row_offsets: offsets.collect(),
        })
    }
}

impl TryFrom<&IvfData> for PbIvf {
    type Error = Error;

    fn try_from(meta: &IvfData) -> Result<Self> {
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
mod tests {
    use arrow_array::{Float32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_core::datatypes::Schema;
    use lance_io::object_store::ObjectStore;
    use lance_table::format::SelfDescribingFileReader;
    use object_store::path::Path;

    use super::*;

    #[test]
    fn test_ivf_find_rows() {
        let mut ivf = IvfData::empty(4);
        ivf.add_partition(20);
        ivf.add_partition(50);

        assert_eq!(ivf.row_range(0), 0..20);
        assert_eq!(ivf.row_range(1), 20..70);
    }

    #[tokio::test]
    async fn test_write_and_load() {
        let mut ivf = IvfData::empty(4);
        ivf.add_partition(20);
        ivf.add_partition(50);

        let object_store = ObjectStore::memory();
        let path = Path::from("/foo");
        let arrow_schema = ArrowSchema::new(vec![Field::new("a", DataType::Float32, true)]);
        let schema = Schema::try_from(&arrow_schema).unwrap();

        {
            let mut writer =
                FileWriter::try_new(&object_store, &path, schema.clone(), &Default::default())
                    .await
                    .unwrap();
            // Write some dummy data
            let batch = RecordBatch::try_new(
                Arc::new(arrow_schema),
                vec![Arc::new(Float32Array::from(vec![Some(1.0)]))],
            )
            .unwrap();
            writer.write(&[batch]).await.unwrap();
            ivf.write(&mut writer).await.unwrap();
            writer.finish().await.unwrap();
        }

        let reader = FileReader::try_new_self_described(&object_store, &path, None)
            .await
            .unwrap();
        assert!(reader.schema().metadata.contains_key(IVF_METADATA_KEY));

        let ivf2 = IvfData::load(&reader).await.unwrap();
        assert_eq!(ivf, ivf2);
        assert_eq!(ivf2.num_partitions(), 2);
    }
}
