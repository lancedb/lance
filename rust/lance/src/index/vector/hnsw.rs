// Copyright 2023 Lance Developers.
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

use std::{
    any::Any,
    collections::HashMap,
    fmt::{Debug, Formatter},
    sync::Arc,
};

use arrow_array::{cast::AsArray, types::Float32Type, Float32Array, RecordBatch, UInt32Array};

use arrow_schema::DataType;
use async_trait::async_trait;
use lance_arrow::*;
use lance_core::{datatypes::Schema, Error, Result};
use lance_file::reader::FileReader;
use lance_index::{
    vector::{
        graph::{VectorStorage, NEIGHBORS_FIELD},
        hnsw::{HnswMetadata, HNSW, VECTOR_ID_FIELD},
        ivf::storage::IVF_PARTITION_KEY,
        Query, DIST_COL,
    },
    Index, IndexType,
};
use lance_io::traits::Reader;
use lance_linalg::distance::DistanceType;
use roaring::RoaringBitmap;
use serde_json::json;
use snafu::{location, Location};
use tracing::instrument;

#[cfg(feature = "opq")]
use super::opq::train_opq;
use super::VectorIndex;
use crate::index::prefilter::PreFilter;

#[derive(Clone)]
pub struct HNSWIndex {
    hnsw: HNSW,
    storage: Option<Arc<dyn VectorStorage>>,
    partition_metadata: Option<Vec<HnswMetadata>>,
}

impl Debug for HNSWIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.hnsw.fmt(f)
    }
}

impl HNSWIndex {
    pub async fn try_new(hnsw: HNSW, reader: Arc<dyn Reader>) -> Result<Self> {
        let schema = Schema::try_from(&arrow_schema::Schema::new(vec![
            NEIGHBORS_FIELD.clone(),
            VECTOR_ID_FIELD.clone(),
        ]))?;

        let reader = FileReader::try_new_from_reader(
            reader.path(),
            reader.clone(),
            None,
            schema,
            0,
            0,
            2,
            None,
        )
        .await?;

        let partition_metadata = match reader.schema().metadata.get(IVF_PARTITION_KEY) {
            Some(json) => {
                let metadata: Vec<HnswMetadata> = serde_json::from_str(json)?;
                Some(metadata)
            }
            None => None,
        };

        Ok(Self {
            hnsw,
            storage: None,
            partition_metadata,
        })
    }

    fn get_partition_metadata(&self, partition_id: usize) -> Result<HnswMetadata> {
        match self.partition_metadata {
            Some(ref metadata) => Ok(metadata[partition_id].clone()),
            None => Err(Error::Index {
                message: "No partition metadata found".to_string(),
                location: location!(),
            }),
        }
    }
}

#[async_trait]
impl Index for HNSWIndex {
    /// Cast to [Any].
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Cast to [Index]
    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    /// Retrieve index statistics as a JSON Value
    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(json!({
            "index_type": "HNSW",
            "distance_type": self.metric_type().to_string(),
        }))
    }

    /// Get the type of the index
    fn index_type(&self) -> IndexType {
        IndexType::Vector
    }

    /// Read through the index and determine which fragment ids are covered by the index
    ///
    /// This is a kind of slow operation.  It's better to use the fragment_bitmap.  This
    /// only exists for cases where the fragment_bitmap has become corrupted or missing.
    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!()
    }
}

#[async_trait]
impl VectorIndex for HNSWIndex {
    #[instrument(level = "debug", skip_all, name = "HNSWIndex::search")]
    async fn search(&self, query: &Query, _pre_filter: Arc<PreFilter>) -> Result<RecordBatch> {
        let results = self.hnsw.search(
            query.key.as_primitive::<Float32Type>().as_slice(),
            query.k,
            30,
            None,
        )?;

        let node_ids = UInt32Array::from_iter_values(results.iter().map(|x| x.0));
        let distances = Arc::new(Float32Array::from_iter_values(results.iter().map(|x| x.1)));

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new(DIST_COL, DataType::Float32, true),
            arrow_schema::Field::new("_node_id", DataType::UInt32, true),
        ]));
        Ok(RecordBatch::try_new(
            schema,
            vec![distances, Arc::new(node_ids)],
        )?)
    }

    fn is_loadable(&self) -> bool {
        true
    }

    fn use_residual(&self) -> bool {
        false
    }

    fn check_can_remap(&self) -> Result<()> {
        Ok(())
    }

    async fn load(
        &self,
        reader: Arc<dyn Reader>,
        _offset: usize,
        _length: usize,
    ) -> Result<Box<dyn VectorIndex>> {
        let schema = Schema::try_from(&arrow_schema::Schema::new(vec![
            NEIGHBORS_FIELD.clone(),
            VECTOR_ID_FIELD.clone(),
        ]))?;

        let reader = FileReader::try_new_from_reader(
            reader.path(),
            reader.clone(),
            None,
            schema,
            0,
            0,
            2,
            None,
        )
        .await?;

        let hnsw = HNSW::load(&reader, self.storage.clone().unwrap()).await?;

        Ok(Box::new(Self {
            hnsw,
            storage: self.storage.clone(),
            partition_metadata: self.partition_metadata.clone(),
        }))
    }

    async fn load_partition(
        &self,
        reader: Arc<dyn Reader>,
        offset: usize,
        length: usize,
        partition_id: usize,
    ) -> Result<Box<dyn VectorIndex>> {
        let schema = Schema::try_from(&arrow_schema::Schema::new(vec![
            NEIGHBORS_FIELD.clone(),
            VECTOR_ID_FIELD.clone(),
        ]))?;

        let reader = FileReader::try_new_from_reader(
            reader.path(),
            reader.clone(),
            None,
            schema,
            0,
            0,
            2,
            None,
        )
        .await?;

        let metadata = self.get_partition_metadata(partition_id)?;
        let hnsw = HNSW::load_partition(
            &reader,
            offset..length,
            self.metric_type(),
            self.storage.clone().unwrap(),
            metadata,
        )
        .await?;

        Ok(Box::new(Self {
            hnsw,
            storage: self.storage.clone(),
            partition_metadata: self.partition_metadata.clone(),
        }))
    }

    fn remap(&mut self, _mapping: &HashMap<u64, Option<u64>>) -> Result<()> {
        Err(Error::Index {
            message: "Remapping HNSW in this way not supported".to_string(),
            location: location!(),
        })
    }

    fn metric_type(&self) -> DistanceType {
        self.hnsw.distance_type()
    }
}
