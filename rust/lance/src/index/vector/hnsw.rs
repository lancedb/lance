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

use std::{any::Any, collections::HashMap, ops::Range, sync::Arc};

use arrow::compute::take;
use arrow_array::{
    cast::AsArray, types::Float32Type, Array, Float32Array, RecordBatch, UInt32Array,
};

use arrow_schema::DataType;
use arrow_select::concat::concat_batches;
use async_trait::async_trait;
use futures::{stream, Stream, StreamExt, TryStreamExt};
use lance_arrow::*;
use lance_core::{
    datatypes::{Field, Schema},
    Error, Result, ROW_ID_FIELD,
};
use lance_file::{
    format::{MAGIC, MAJOR_VERSION, MINOR_VERSION},
    reader::FileReader,
};
use lance_index::{
    optimize::OptimizeOptions,
    vector::{
        graph::{memory::InMemoryVectorStorage, NEIGHBORS_FIELD},
        hnsw::{builder::HnswBuildParams, HNSWBuilder, HNSW, VECTOR_ID_FIELD},
        Query, DIST_COL,
    },
    Index, IndexType,
};
use lance_io::{
    memory::MemoryBufReader, stream::RecordBatchStream, traits::Reader,
    utils::read_fixed_stride_array,
};
use lance_linalg::{
    distance::{Cosine, Dot, MetricType, L2},
    MatrixView,
};
use log::{debug, info};
use object_store::{memory::InMemory, path::Path};
use rand::{rngs::SmallRng, SeedableRng};
use roaring::RoaringBitmap;
use serde::Serialize;
use serde_json::json;
use snafu::{location, Location};
use tracing::{instrument, span, Level};
use uuid::Uuid;

#[cfg(feature = "opq")]
use super::opq::train_opq;
use super::{pq::PQIndex, utils::maybe_sample_training_data, VectorIndex};
use crate::dataset::builder::DatasetBuilder;
use crate::{
    dataset::Dataset,
    index::{pb, prefilter::PreFilter, INDEX_FILE_NAME},
    session::Session,
};

#[derive(Debug, Clone)]
pub struct HNSWIndex {
    hnsw: HNSW,
    // row_ids: Option<Arc<dyn Array>>,
}

impl HNSWIndex {
    pub fn new(hnsw: HNSW) -> Self {
        Self {
            hnsw,
            // row_ids: None,
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
            "metric_type": self.hnsw.metric_type().to_string(),
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
        todo!()
    }
}

#[async_trait]
impl VectorIndex for HNSWIndex {
    #[instrument(level = "debug", skip_all, name = "IVFIndex::search")]
    async fn search(&self, query: &Query, pre_filter: Arc<PreFilter>) -> Result<RecordBatch> {
        let results = self.hnsw.search(
            query.key.as_primitive::<Float32Type>().as_slice(),
            query.k,
            30,
        )?;

        let node_ids = UInt32Array::from_iter_values(results.iter().map(|x| x.0));

        // let row_ids = take(&self.row_ids.as_ref().unwrap(), &node_ids, None)?;
        let distances = Arc::new(Float32Array::from_iter_values(results.iter().map(|x| x.1)));

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new(DIST_COL, DataType::Float32, true),
            // ROW_ID_FIELD.clone(),
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
        reader: &dyn Reader,
        offset: usize,
        length: usize,
    ) -> Result<Box<dyn VectorIndex>> {
        let bytes = reader
            .get_range(Range {
                start: offset,
                end: offset + length,
            })
            .await?;

        let reader = MemoryBufReader::new(bytes, reader.block_size(), reader.path().clone());

        let schema = Schema::try_from(&arrow_schema::Schema::new(vec![
            NEIGHBORS_FIELD.clone(),
            VECTOR_ID_FIELD.clone(),
        ]))?;

        let file_reader =
            FileReader::try_new_from_reader(Box::new(reader), None, schema, 0, 0, None).await?;

        let hnsw = HNSW::load(&file_reader).await?;

        // let offset = file_reader
        //     .schema()
        //     .metadata
        //     .get("lance:binary_offset")
        //     .ok_or(Error::Index {
        //         message: "lance:binary_offset not found in schema metadata".to_string(),
        //         location: location!(),
        //     })?
        //     .parse::<usize>()
        //     .map_err(|e| Error::Index {
        //         message: format!("Failed to parse lance:binary_offset: {}", e),
        //         location: location!(),
        //     })?;

        // let row_ids =
        //     read_fixed_stride_array(reader, &DataType::UInt64, offset, self.hnsw.len(), ..).await?;

        Ok(Box::new(Self {
            hnsw,
            // row_ids: Some(row_ids),
        }))
    }

    fn remap(&mut self, _mapping: &HashMap<u64, Option<u64>>) -> Result<()> {
        // This will be needed if we want to clean up IVF to allow more than just
        // one layer (e.g. IVF -> IVF -> PQ).  We need to pass on the call to
        // remap to the lower layers.

        // Currently, remapping for IVF is implemented in remap_index_file which
        // mirrors some of the other IVF routines like build_ivf_pq_index
        Err(Error::Index {
            message: "Remapping IVF in this way not supported".to_string(),
            location: location!(),
        })
    }

    fn metric_type(&self) -> MetricType {
        self.hnsw.metric_type()
    }
}
