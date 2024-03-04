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
    sync::{Arc, Weak},
};

use arrow_arith::numeric::sub;
use arrow_array::{
    cast::{as_struct_array, AsArray},
    types::{Float16Type, Float32Type, Float64Type},
    Array, FixedSizeListArray, Float32Array, RecordBatch, StructArray, UInt32Array,
};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::DataType;
use arrow_select::{concat::concat_batches, take::take};
use async_trait::async_trait;
use futures::{
    stream::{self, StreamExt},
    TryStreamExt,
};
use lance_arrow::*;
use lance_core::{datatypes::Field, Error, Result};
use lance_file::format::{MAGIC, MAJOR_VERSION, MINOR_VERSION};
use lance_index::{
    optimize::OptimizeOptions,
    vector::{hnsw::HNSW, Query, DIST_COL},
    Index, IndexType,
};
use lance_io::{
    encodings::plain::PlainEncoder,
    local::to_local_path,
    object_store::ObjectStore,
    object_writer::ObjectWriter,
    stream::RecordBatchStream,
    traits::{Reader, WriteExt, Writer},
};
use lance_linalg::distance::{Cosine, Dot, MetricType, L2};
use log::{debug, info};
use object_store::path::Path;
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
    inner: HNSW,
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
            "metric_type": self.inner.metric_type().to_string(),
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
    async fn search(&self, query: &Query, pre_filter: Arc<PreFilter>) -> Result<RecordBatch> {}

    fn is_loadable(&self) -> bool {
        false
    }

    fn use_residual(&self) -> bool {
        false
    }

    fn check_can_remap(&self) -> Result<()> {
        Ok(())
    }

    async fn load(
        &self,
        _reader: &dyn Reader,
        _offset: usize,
        _length: usize,
    ) -> Result<Box<dyn VectorIndex>> {
        Err(Error::Index {
            message: "Flat index does not support load".to_string(),
            location: location!(),
        })
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
        self.metric_type
    }
}
