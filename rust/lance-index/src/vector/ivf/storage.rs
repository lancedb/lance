// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;

use arrow_array::{Array, ArrayRef, FixedSizeListArray, UInt32Array};
use deepsize::DeepSizeOf;
use itertools::Itertools;
use lance_core::{Error, Result};
use lance_file::{reader::FileReader, writer::FileWriter};
use lance_io::{traits::WriteExt, utils::read_message};
use lance_linalg::distance::DistanceType;
use lance_table::io::manifest::ManifestDescribing;
use log::debug;
use serde::{Deserialize, Serialize};
use snafu::{location, Location};

use crate::pb::Ivf as PbIvf;

pub const IVF_METADATA_KEY: &str = "lance:ivf";
pub const IVF_PARTITION_KEY: &str = "lance:ivf:partition";

/// Ivf Model
#[derive(Debug, Clone, PartialEq)]
pub struct IvfModel {
    /// Centroids of each partition.
    ///
    /// It is a 2-D `(num_partitions * dimension)` of vector array.
    pub centroids: Option<FixedSizeListArray>,

    /// Offset of each partition in the file.
    pub offsets: Vec<usize>,

    /// Number of vectors in each partition.
    pub lengths: Vec<u32>,
}

impl DeepSizeOf for IvfModel {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.centroids
            .as_ref()
            .map(|centroids| centroids.get_array_memory_size())
            .unwrap_or_default()
            + self.lengths.deep_size_of_children(context)
            + self.offsets.deep_size_of_children(context)
    }
}

impl IvfModel {
    pub fn empty() -> Self {
        Self {
            centroids: None,
            offsets: vec![],
            lengths: vec![],
        }
    }

    pub fn new(centroids: FixedSizeListArray) -> Self {
        Self {
            centroids: Some(centroids),
            offsets: vec![],
            lengths: vec![],
        }
    }

    pub fn centroid(&self, partition: usize) -> Option<ArrayRef> {
        self.centroids.as_ref().map(|c| c.value(partition))
    }

    /// Ivf model dimension.
    pub fn dimension(&self) -> usize {
        self.centroids
            .as_ref()
            .map(|c| c.value_length() as usize)
            .unwrap_or(0)
    }

    /// Number of IVF partitions.
    pub fn num_partitions(&self) -> usize {
        self.centroids
            .as_ref()
            .map(|c| c.len())
            .unwrap_or_else(|| self.offsets.len())
    }

    pub fn partition_size(&self, part: usize) -> usize {
        self.lengths[part] as usize
    }

    /// Use the query vector to find `nprobes` closest partitions.
    pub fn find_partitions(
        &self,
        query: &dyn Array,
        nprobes: usize,
        distance_type: DistanceType,
    ) -> Result<UInt32Array> {
        let internal = crate::vector::ivf::new_ivf_transformer(
            self.centroids.clone().unwrap(),
            distance_type,
            vec![],
        );
        internal.find_partitions(query, nprobes)
    }

    /// Add the offset and length of one partition.
    pub fn add_partition(&mut self, len: u32) {
        self.offsets.push(
            self.offsets.last().cloned().unwrap_or_default()
                + self.lengths.last().cloned().unwrap_or_default() as usize,
        );
        self.lengths.push(len);
    }

    /// Add the offset and length of one partition with the given offset.
    /// this is used for old index format of IVF_PQ.
    pub fn add_partition_with_offset(&mut self, offset: usize, len: u32) {
        self.offsets.push(offset);
        self.lengths.push(len);
    }

    pub fn row_range(&self, partition: usize) -> Range<usize> {
        let start = self.offsets[partition];
        let end = start + self.lengths[partition] as usize;
        start..end
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
}

/// Convert IvfModel to protobuf.
impl TryFrom<&IvfModel> for PbIvf {
    type Error = Error;

    fn try_from(ivf: &IvfModel) -> Result<Self> {
        let lengths = ivf.lengths.clone();

        Ok(Self {
            centroids: vec![], // Deprecated
            lengths,
            offsets: ivf.offsets.iter().map(|x| *x as u64).collect(),
            centroids_tensor: ivf.centroids.as_ref().map(|c| c.try_into()).transpose()?,
        })
    }
}

/// Convert IvfModel to protobuf.
impl TryFrom<PbIvf> for IvfModel {
    type Error = Error;

    fn try_from(proto: PbIvf) -> Result<Self> {
        let centroids = if let Some(tensor) = proto.centroids_tensor.as_ref() {
            debug!("Ivf: loading IVF centroids from index format v2");
            Some(FixedSizeListArray::try_from(tensor)?)
        } else {
            None
        };
        // We are not using offsets from the protobuf, which was the file offset in the
        // v1 index format. It will be deprecated soon.
        //
        // This new offset uses the row offset in the lance file.
        let offsets = match proto.offsets.len() {
            0 => proto
                .lengths
                .iter()
                .scan(0_usize, |state, &x| {
                    let old = *state;
                    *state += x as usize;
                    Some(old)
                })
                .collect_vec(),
            _ => proto.offsets.iter().map(|x| *x as usize).collect(),
        };
        assert_eq!(offsets.len(), proto.lengths.len());
        Ok(Self {
            centroids,
            offsets,
            lengths: proto.lengths.clone(),
        })
    }
}

/// The IVF metadata stored in the Lance Schema
#[derive(Serialize, Deserialize, Debug)]
struct IvfMetadata {
    // The file position to store the protobuf binary of IVF metadata.
    pb_position: usize,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Float32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_core::datatypes::Schema;
    use lance_io::object_store::ObjectStore;
    use lance_table::format::SelfDescribingFileReader;
    use object_store::path::Path;

    use super::*;

    #[test]
    fn test_ivf_find_rows() {
        let mut ivf = IvfModel::empty();
        ivf.add_partition(20);
        ivf.add_partition(50);

        assert_eq!(ivf.row_range(0), 0..20);
        assert_eq!(ivf.row_range(1), 20..70);
    }

    #[tokio::test]
    async fn test_write_and_load() {
        let mut ivf = IvfModel::empty();
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

        let ivf2 = IvfModel::load(&reader).await.unwrap();
        assert_eq!(ivf, ivf2);
        assert_eq!(ivf2.num_partitions(), 2);
    }
}
