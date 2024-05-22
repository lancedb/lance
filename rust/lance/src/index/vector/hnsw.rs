// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    any::Any,
    collections::HashMap,
    fmt::{Debug, Formatter},
    sync::Arc,
};

use arrow_array::{Float32Array, RecordBatch, UInt64Array};
use async_trait::async_trait;
use lance_core::{datatypes::Schema, Error, Result};
use lance_file::reader::FileReader;
use lance_index::vector::{hnsw::HNSW, quantizer::Quantizer};
use lance_index::{
    vector::{
        graph::{VectorStore, NEIGHBORS_FIELD},
        hnsw::{HnswMetadata, VECTOR_ID_FIELD},
        ivf::storage::IVF_PARTITION_KEY,
        quantizer::{IvfQuantizationStorage, Quantization},
        Query,
    },
    Index, IndexType,
};
use lance_io::traits::Reader;
use lance_linalg::distance::DistanceType;
use lance_table::format::SelfDescribingFileReader;
use roaring::RoaringBitmap;
use serde_json::json;
use snafu::{location, Location};
use tracing::instrument;

use super::VectorIndex;
use crate::index::prefilter::PreFilter;
use crate::RESULT_SCHEMA;

pub mod builder;

#[derive(Clone)]
pub(crate) struct HNSWIndexOptions {
    pub use_residual: bool,
}

#[derive(Clone)]
pub(crate) struct HNSWIndex<Q: Quantization> {
    distance_type: DistanceType,

    // Some(T) if the index is loaded, None otherwise
    hnsw: Option<HNSW>,
    storage: Option<Arc<Q::Storage>>,

    // TODO: move these into IVFIndex after the refactor is complete
    partition_storage: IvfQuantizationStorage<Q>,
    partition_metadata: Option<Vec<HnswMetadata>>,

    options: HNSWIndexOptions,
}

impl<Q: Quantization> Debug for HNSWIndex<Q> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.hnsw.fmt(f)
    }
}

impl<Q: Quantization> HNSWIndex<Q> {
    pub async fn try_new(
        distance_type: DistanceType,
        reader: Arc<dyn Reader>,
        aux_reader: Arc<dyn Reader>,
        options: HNSWIndexOptions,
    ) -> Result<Self> {
        let reader = FileReader::try_new_self_described_from_reader(reader.clone(), None).await?;

        let partition_metadata = match reader.schema().metadata.get(IVF_PARTITION_KEY) {
            Some(json) => {
                let metadata: Vec<HnswMetadata> = serde_json::from_str(json)?;
                Some(metadata)
            }
            None => None,
        };

        let ivf_store = IvfQuantizationStorage::open(aux_reader).await?;
        Ok(Self {
            distance_type,
            hnsw: None,
            storage: None,
            partition_storage: ivf_store,
            partition_metadata,
            options,
        })
    }

    pub fn quantizer(&self) -> &Quantizer {
        self.partition_storage.quantizer()
    }

    pub fn metadata(&self) -> HnswMetadata {
        self.partition_metadata.as_ref().unwrap()[0].clone()
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
impl<Q: Quantization + Send + Sync + 'static> Index for HNSWIndex<Q> {
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
impl<Q: Quantization + Send + Sync + 'static> VectorIndex for HNSWIndex<Q> {
    #[instrument(level = "debug", skip_all, name = "HNSWIndex::search")]
    async fn search(&self, query: &Query, pre_filter: Arc<PreFilter>) -> Result<RecordBatch> {
        let schema = RESULT_SCHEMA.clone();

        let hnsw = self.hnsw.as_ref().ok_or(Error::Index {
            message: "HNSW index not loaded".to_string(),
            location: location!(),
        })?;

        let storage = self.storage.as_ref().ok_or(Error::Index {
            message: "vector storage not loaded".to_string(),
            location: location!(),
        })?;

        if hnsw.is_empty() {
            return Ok(RecordBatch::new_empty(schema));
        }

        let row_ids = storage.row_ids();
        let bitmap = if pre_filter.is_empty() {
            None
        } else {
            pre_filter.wait_for_ready().await?;

            let indices = pre_filter.filter_row_ids(row_ids);
            Some(
                RoaringBitmap::from_sorted_iter(indices.into_iter().map(|i| i as u32)).map_err(
                    |e| Error::Index {
                        message: format!("Error creating RoaringBitmap: {}", e),
                        location: location!(),
                    },
                )?,
            )
        };

        let refine_factor = query.refine_factor.unwrap_or(1) as usize;
        let k = query.k * refine_factor;
        let ef = query.ef.unwrap_or(k + k / 2);
        if ef < k {
            return Err(Error::Index {
                message: "ef must be greater than or equal to k".to_string(),
                location: location!(),
            });
        }

        let results = hnsw.search_basic(query.key.clone(), k, ef, bitmap, storage.as_ref())?;

        let row_ids = UInt64Array::from_iter_values(results.iter().map(|x| row_ids[x.id as usize]));
        let distances = Arc::new(Float32Array::from_iter_values(
            results.iter().map(|x| x.dist.0),
        ));

        Ok(RecordBatch::try_new(
            schema,
            vec![distances, Arc::new(row_ids)],
        )?)
    }

    fn is_loadable(&self) -> bool {
        true
    }

    fn use_residual(&self) -> bool {
        self.options.use_residual
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

        let storage = Arc::new(self.partition_storage.load_partition(0).await?);
        let hnsw = HNSW::load(&reader, storage.clone()).await?;

        Ok(Box::new(Self {
            distance_type: self.distance_type,
            hnsw: Some(hnsw),
            storage: Some(storage),
            partition_storage: self.partition_storage.clone(),
            partition_metadata: self.partition_metadata.clone(),
            options: self.options.clone(),
        }))
    }

    async fn load_partition(
        &self,
        reader: Arc<dyn Reader>,
        offset: usize,
        length: usize,
        partition_id: usize,
    ) -> Result<Box<dyn VectorIndex>> {
        let reader = FileReader::try_new_self_described_from_reader(reader, None).await?;

        let metadata = self.get_partition_metadata(partition_id)?;
        let storage = Arc::new(self.partition_storage.load_partition(partition_id).await?);
        let hnsw = HNSW::load_partition(
            &reader,
            offset..offset + length,
            self.metric_type(),
            storage.clone(),
            metadata,
        )
        .await?;

        Ok(Box::new(Self {
            distance_type: self.distance_type,
            hnsw: Some(hnsw),
            storage: Some(storage),
            partition_storage: self.partition_storage.clone(),
            partition_metadata: self.partition_metadata.clone(),
            options: self.options.clone(),
        }))
    }

    fn row_ids(&self) -> &[u64] {
        self.storage.as_ref().unwrap().row_ids()
    }

    fn remap(&mut self, _mapping: &HashMap<u64, Option<u64>>) -> Result<()> {
        Err(Error::Index {
            message: "Remapping HNSW in this way not supported".to_string(),
            location: location!(),
        })
    }

    fn metric_type(&self) -> DistanceType {
        self.distance_type
    }
}
