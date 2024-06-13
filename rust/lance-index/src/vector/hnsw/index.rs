// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    any::Any,
    collections::HashMap,
    fmt::{Debug, Formatter},
    sync::Arc,
};

use arrow_array::{RecordBatch, UInt32Array};
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use lance_core::{datatypes::Schema, Error, Result};
use lance_file::reader::FileReader;
use lance_io::traits::Reader;
use lance_linalg::distance::DistanceType;
use lance_table::format::SelfDescribingFileReader;
use roaring::RoaringBitmap;
use serde_json::json;
use snafu::{location, Location};
use tracing::instrument;

use crate::prefilter::PreFilter;
use crate::vector::v3::subindex::{IvfSubIndex, SUB_INDEX_METADATA_KEY};
use crate::{
    vector::{
        graph::NEIGHBORS_FIELD,
        hnsw::{HnswMetadata, HNSW, VECTOR_ID_FIELD},
        ivf::storage::IVF_PARTITION_KEY,
        quantizer::{IvfQuantizationStorage, Quantization, Quantizer},
        storage::VectorStore,
        Query, VectorIndex,
    },
    Index, IndexType,
};

#[derive(Clone, DeepSizeOf)]
pub struct HNSWIndexOptions {
    pub use_residual: bool,
}

#[derive(Clone, DeepSizeOf)]
pub struct HNSWIndex<Q: Quantization> {
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
            "distance_type": self.partition_storage.distance_type().to_string(),
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
    async fn search(&self, query: &Query, pre_filter: Arc<dyn PreFilter>) -> Result<RecordBatch> {
        let hnsw = self.hnsw.as_ref().ok_or(Error::Index {
            message: "HNSW index not loaded".to_string(),
            location: location!(),
        })?;

        let storage = self.storage.as_ref().ok_or(Error::Index {
            message: "vector storage not loaded".to_string(),
            location: location!(),
        })?;

        let refine_factor = query.refine_factor.unwrap_or(1) as usize;
        let k = query.k * refine_factor;

        hnsw.search(
            query.key.clone(),
            k,
            query.into(),
            storage.as_ref(),
            pre_filter,
        )
    }

    fn find_partitions(&self, _: &Query) -> Result<UInt32Array> {
        unimplemented!("only for IVF")
    }

    async fn search_in_partition(
        &self,
        _: usize,
        _: &Query,
        _: Arc<dyn PreFilter>,
    ) -> Result<RecordBatch> {
        unimplemented!("only for IVF")
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
        let batch = reader.read_range(0..reader.len(), reader.schema()).await?;
        let hnsw = HNSW::load(batch)?;

        Ok(Box::new(Self {
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
        let batch = reader
            .read_range(offset..offset + length, reader.schema())
            .await?;
        let mut schema = batch.schema_ref().as_ref().clone();
        schema.metadata.insert(
            SUB_INDEX_METADATA_KEY.to_string(),
            serde_json::to_string(&metadata)?,
        );
        let batch = batch.with_schema(schema.into())?;
        let hnsw = HNSW::load(batch)?;

        Ok(Box::new(Self {
            hnsw: Some(hnsw),
            storage: Some(storage),
            partition_storage: self.partition_storage.clone(),
            partition_metadata: self.partition_metadata.clone(),
            options: self.options.clone(),
        }))
    }

    fn row_ids(&self) -> Box<dyn Iterator<Item = &'_ u64> + '_> {
        Box::new(self.storage.as_ref().unwrap().row_ids())
    }

    fn remap(&mut self, _mapping: &HashMap<u64, Option<u64>>) -> Result<()> {
        Err(Error::Index {
            message: "Remapping HNSW in this way not supported".to_string(),
            location: location!(),
        })
    }

    fn metric_type(&self) -> DistanceType {
        self.partition_storage.distance_type()
    }
}
