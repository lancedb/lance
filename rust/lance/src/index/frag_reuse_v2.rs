// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stable-partition commits and asynchronous translation through mixed FRI histories.

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch, UInt32Array, UInt64Array};
use async_trait::async_trait;
use futures::TryStreamExt;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_index::scalar::{IndexFile, IndexReader, IndexWriter};
use lance_io::stream::{RecordBatchStream, RecordBatchStreamAdapter};

use lance_core::utils::address::RowAddress;
use lance_core::utils::stable_partition::CountsMatrix;
use lance_core::{Error, Result};
use lance_index::frag_reuse::row_map::RowMapReader;
use lance_index::frag_reuse::{
    CompactFragReuseIndex, FRAG_REUSE_INDEX_NAME, FragReuseIndexDetails, FragReuseVersion,
};
use lance_index::scalar::IndexStore;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_table::format::{Fragment, ROW_MAP_FILE_NAME, StablePartitionTransition};
use lance_table::system_index::frag_reuse::FragDigest;
use lance_table::transaction::{Operation, RewriteGroup, Transaction};
use roaring::RoaringBitmap;
use tokio::sync::OnceCell;
use uuid::Uuid;

use crate::Dataset;
use crate::index::frag_reuse::load_frag_reuse_index_details;
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
use lance_index::metrics::NoOpMetricsCollector;
use lance_table::io::manifest::read_manifest_indexes;

enum Mapping {
    Ordered(CompactFragReuseIndex),
    Partition {
        transition: StablePartitionTransition,
        store: Box<LanceIndexStore>,
        reader: OnceCell<RowMapReader>,
    },
}

struct Rewrite {
    sources: RoaringBitmap,
    destinations: RoaringBitmap,
    mapping: Mapping,
}

/// A snapshot's V1 compactions and V2 stable partitions in fragment-lineage order.
///
/// Row-map labels are loaded only for requested blocks. Metadata does not need
/// to pretend V1's builder version is its successful commit timestamp.
pub struct MixedFragReuseIndex {
    rewrites: Vec<Rewrite>,
}

impl DeepSizeOf for MixedFragReuseIndex {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.rewrites.capacity() * std::mem::size_of::<Rewrite>()
            + self
                .rewrites
                .iter()
                .map(|rewrite| {
                    rewrite.sources.serialized_size()
                        + rewrite.destinations.serialized_size()
                        + match &rewrite.mapping {
                            Mapping::Ordered(mapping) => mapping.deep_size_of_children(context),
                            Mapping::Partition {
                                transition,
                                store,
                                reader,
                            } => {
                                transition.deep_size_of_children(context)
                                    + store.deep_size_of_children(context)
                                    + reader.get().map_or(0, |reader| {
                                        reader.counts().num_blocks()
                                            * reader.counts().num_destinations() as usize
                                            * std::mem::size_of::<u32>()
                                    })
                            }
                        }
                })
                .sum::<usize>()
    }
}

impl MixedFragReuseIndex {
    /// Load rewrite metadata, leaving stable-partition label files unopened.
    pub async fn open(dataset: &Dataset) -> Result<Self> {
        let mut rewrites = Vec::new();
        let indices = read_manifest_indexes(
            &dataset.object_store,
            &dataset.manifest_location,
            &dataset.manifest,
        )
        .await?;
        if let Some(index) = indices
            .iter()
            .find(|index| index.name == FRAG_REUSE_INDEX_NAME)
        {
            let details = load_frag_reuse_index_details(dataset, index).await?;
            for version in &details.versions {
                for group in &version.groups {
                    rewrites.push(Rewrite {
                        sources: group.old_frags.iter().map(|f| f.id as u32).collect(),
                        destinations: group.new_frags.iter().map(|f| f.id as u32).collect(),
                        mapping: Mapping::Ordered(CompactFragReuseIndex::try_new(
                            index.uuid,
                            FragReuseIndexDetails {
                                versions: vec![FragReuseVersion {
                                    dataset_version: version.dataset_version,
                                    groups: vec![group.clone()],
                                }],
                            },
                        )?),
                    });
                }
            }
        }
        for transition in dataset.manifest.stable_partition_transitions.iter() {
            transition.validate()?;
            if transition.committed_version <= transition.source_dataset_version
                || transition.committed_version > dataset.manifest.version
            {
                return Err(Error::corrupt_file_named(
                    "manifest",
                    format!(
                        "row map {} has invalid commit version {} for source {} and manifest {}",
                        transition.row_map_id,
                        transition.committed_version,
                        transition.source_dataset_version,
                        dataset.manifest.version,
                    ),
                ));
            }
            if transition.base_id.is_some() {
                return Err(Error::not_supported(
                    "mixed row-map reads from external storage bases are not implemented",
                ));
            }
            let directory = dataset
                .row_maps_dir()
                .join(transition.row_map_id.to_string());
            let cache = dataset.metadata_cache.file_metadata_cache(&directory);
            let store =
                LanceIndexStore::new(dataset.object_store.clone(), directory, Arc::new(cache))
                    .with_file_sizes(HashMap::from([(
                        ROW_MAP_FILE_NAME.to_string(),
                        transition.row_map_size_bytes,
                    )]));
            rewrites.push(Rewrite {
                sources: transition.source_ids(),
                destinations: transition.destination_ids(),
                mapping: Mapping::Partition {
                    transition: transition.clone(),
                    store: Box::new(store),
                    reader: OnceCell::new(),
                },
            });
        }
        let order = rewrite_order(&rewrites)?;
        let mut nodes: Vec<Option<Rewrite>> = rewrites.into_iter().map(Some).collect();
        let mut rewrites = Vec::with_capacity(nodes.len());
        for index in order {
            if let Some(node) = nodes[index].take() {
                rewrites.push(node);
            }
        }
        Ok(Self { rewrites })
    }

    /// Derive conservative full-fragment coverage through both mapping kinds.
    pub fn remap_fragment_bitmap(&self, coverage: &RoaringBitmap) -> RoaringBitmap {
        let mut coverage = coverage.clone();
        for rewrite in &self.rewrites {
            let all = rewrite.sources.is_subset(&coverage);
            coverage -= &rewrite.sources;
            if all {
                coverage |= &rewrite.destinations;
            }
        }
        coverage
    }

    /// Derive each segment's probe coverage using the logical index's joint provenance.
    ///
    /// Derived bitmaps may overlap: all contributing segments must be searched.
    /// A segment already naming a destination owns it directly, so translated
    /// entries from older segments are excluded before predicate evaluation.
    pub(crate) fn segment_coverage(&self, sources: &[RoaringBitmap]) -> Vec<RoaringBitmap> {
        let mut coverage = sources.to_vec();
        for rewrite in &self.rewrites {
            let union = coverage
                .iter()
                .fold(RoaringBitmap::new(), |mut union, bitmap| {
                    union |= bitmap;
                    union
                });
            let complete = rewrite.sources.is_subset(&union);
            let direct = &union & &rewrite.destinations;
            for bitmap in &mut coverage {
                let contributes = !bitmap.is_disjoint(&rewrite.sources);
                *bitmap -= &rewrite.sources;
                if complete && contributes {
                    *bitmap |= &rewrite.destinations - &direct;
                }
            }
        }
        coverage
    }

    fn direct_coverage(&self, coverage: &RoaringBitmap) -> RoaringBitmap {
        let mut coverage = coverage.clone();
        for rewrite in &self.rewrites {
            let complete = rewrite.sources.is_subset(&coverage);
            coverage -= &rewrite.sources;
            if complete && matches!(rewrite.mapping, Mapping::Ordered(_)) {
                coverage |= &rewrite.destinations;
            }
        }
        coverage
    }

    /// Whether this index's stored addresses need a stable-partition mapping.
    pub(crate) fn requires_partition_translation(&self, coverage: Option<&RoaringBitmap>) -> bool {
        let Some(coverage) = coverage else {
            return true;
        };
        let mut coverage = coverage.clone();
        for rewrite in &self.rewrites {
            if !rewrite.sources.is_disjoint(&coverage)
                && matches!(rewrite.mapping, Mapping::Partition { .. })
            {
                return true;
            }
            let affected = !rewrite.sources.is_disjoint(&coverage);
            coverage -= &rewrite.sources;
            if affected {
                coverage |= &rewrite.destinations;
            }
        }
        false
    }

    /// Translate physical addresses, preserving input order and deleted positions.
    ///
    /// Deletions that happened after the last rewrite must still be checked in
    /// the target snapshot's deletion vectors before returning query results.
    pub async fn translate(&self, addresses: &[RowAddress]) -> Result<Vec<Option<RowAddress>>> {
        let mut translated: Vec<Option<RowAddress>> = addresses.iter().copied().map(Some).collect();
        for rewrite in &self.rewrites {
            match &rewrite.mapping {
                Mapping::Ordered(mapping) => {
                    for address in &mut translated {
                        if let Some(current) = *address {
                            *address = mapping.remap_row_id(current.into()).map(RowAddress::from);
                        }
                    }
                }
                Mapping::Partition {
                    transition,
                    store,
                    reader,
                } => {
                    let mut base = 0u64;
                    let sources: HashMap<u32, (u64, usize)> = transition
                        .sources
                        .iter()
                        .map(|source| {
                            let start = base;
                            base += source.physical_rows as u64;
                            (source.id as u32, (start, source.physical_rows))
                        })
                        .collect();
                    let mut requests = Vec::new();
                    for (position, address) in translated.iter().enumerate() {
                        let Some(address) = address else { continue };
                        let Some(&(start, rows)) = sources.get(&address.fragment_id()) else {
                            continue;
                        };
                        if address.row_offset() as usize >= rows {
                            return Err(Error::invalid_input(format!(
                                "row offset {} exceeds source fragment {} length {rows}",
                                address.row_offset(),
                                address.fragment_id(),
                            )));
                        }
                        requests.push((start + u64::from(address.row_offset()), position));
                    }
                    if requests.is_empty() {
                        continue;
                    }
                    let reader = reader
                        .get_or_try_init(|| async {
                            let reader =
                                RowMapReader::open(store.open_index_file(ROW_MAP_FILE_NAME).await?)
                                    .await?;
                            validate_counts(transition, reader.counts())?;
                            Ok::<_, Error>(reader)
                        })
                        .await?;
                    // Limit scattered index pages to sixteen logical label blocks
                    // per read, instead of materializing every touched block at once.
                    requests.sort_unstable_by_key(|&(row, _)| row);
                    let block_rows = u64::from(reader.counts().block_rows());
                    let mut remaining = requests.as_slice();
                    while let Some(&(first_row, _)) = remaining.first() {
                        let first_block = first_row / block_rows;
                        let count = remaining
                            .partition_point(|&(row, _)| row / block_rows - first_block < 16);
                        let (batch, rest) = remaining.split_at(count);
                        let source_rows = batch.iter().map(|&(row, _)| row).collect::<Vec<_>>();
                        for (&(_, position), address) in
                            batch.iter().zip(reader.translate_many(&source_rows).await?)
                        {
                            translated[position] = address.map(|(label, offset)| {
                                RowAddress::new_from_parts(
                                    transition.destinations[usize::from(label)].id as u32,
                                    offset,
                                )
                            });
                        }
                        remaining = rest;
                    }
                }
            }
        }
        Ok(translated)
    }
}

fn rewrite_order(rewrites: &[Rewrite]) -> Result<Vec<usize>> {
    let mut consumers = HashMap::new();
    let mut produced = RoaringBitmap::new();
    for (index, rewrite) in rewrites.iter().enumerate() {
        for source in &rewrite.sources {
            if consumers.insert(source, index).is_some() {
                return Err(Error::invalid_input(format!(
                    "fragment {source} is consumed by multiple rewrites"
                )));
            }
        }
        for destination in &rewrite.destinations {
            if !produced.insert(destination) {
                return Err(Error::invalid_input(format!(
                    "fragment {destination} is produced by multiple rewrites"
                )));
            }
        }
    }
    let mut successors = vec![RoaringBitmap::new(); rewrites.len()];
    let mut incoming = vec![0usize; rewrites.len()];
    for (index, rewrite) in rewrites.iter().enumerate() {
        for destination in &rewrite.destinations {
            if let Some(&next) = consumers.get(&destination)
                && successors[index].insert(next as u32)
            {
                incoming[next] += 1;
            }
        }
    }
    let mut ready: VecDeque<usize> = incoming
        .iter()
        .enumerate()
        .filter_map(|(i, &n)| (n == 0).then_some(i))
        .collect();
    let mut order = Vec::with_capacity(rewrites.len());
    while let Some(index) = ready.pop_front() {
        order.push(index);
        for next in &successors[index] {
            incoming[next as usize] -= 1;
            if incoming[next as usize] == 0 {
                ready.push_back(next as usize);
            }
        }
    }
    if order.len() != rewrites.len() {
        return Err(Error::invalid_input("cycle in fragment rewrite history"));
    }
    Ok(order)
}

fn validate_counts(transition: &StablePartitionTransition, counts: &CountsMatrix) -> Result<()> {
    let physical: u64 = transition
        .sources
        .iter()
        .map(|f| f.physical_rows as u64)
        .sum();
    if counts.total_rows() != physical
        || counts.num_destinations() as usize != transition.destinations.len()
    {
        return Err(Error::invalid_input(
            "row-map source or destination dimensions differ from transition metadata",
        ));
    }
    for (label, destination) in transition.destinations.iter().enumerate() {
        if u64::from(counts.total(label as u16)) != destination.physical_rows as u64 {
            return Err(Error::invalid_input(format!(
                "row-map label {label} count differs from destination {} row count",
                destination.id
            )));
        }
    }
    Ok(())
}

pub(crate) fn mixed_cache_id(dataset: &Dataset, v1: Option<Uuid>) -> Option<Uuid> {
    if dataset.manifest.stable_partition_transitions.is_empty() {
        return v1;
    }
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    dataset.uri().hash(&mut hash);
    dataset.manifest_location.path.hash(&mut hash);
    dataset.manifest_location.e_tag.hash(&mut hash);
    v1.hash(&mut hash);
    for t in dataset.manifest.stable_partition_transitions.iter() {
        t.row_map_id.hash(&mut hash);
    }
    Some(Uuid::from_u128(
        (u128::from(hash.finish()) << 64) | u128::from(dataset.manifest.version),
    ))
}

/// B-tree decode adapter. Translates bounded batches before the scalar index
/// evaluates predicates, preserving lazy row-map I/O and current coverage.
#[derive(Clone)]
pub(crate) struct TranslatedIndexStore {
    inner: Arc<dyn IndexStore>,
    mapping: Arc<MixedFragReuseIndex>,
    coverage: RoaringBitmap,
}

impl std::fmt::Debug for TranslatedIndexStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranslatedIndexStore")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl DeepSizeOf for TranslatedIndexStore {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.inner.deep_size_of_children(context)
            + self.mapping.deep_size_of_children(context)
            + self.coverage.serialized_size()
    }
}

impl TranslatedIndexStore {
    pub(crate) async fn new(
        dataset: &Dataset,
        inner: Arc<dyn IndexStore>,
        coverage: RoaringBitmap,
    ) -> Result<Self> {
        Ok(Self {
            inner,
            mapping: Arc::new(MixedFragReuseIndex::open(dataset).await?),
            coverage,
        })
    }
}

#[async_trait]
impl IndexStore for TranslatedIndexStore {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_arc(&self) -> Arc<dyn IndexStore> {
        Arc::new(self.clone())
    }
    fn io_parallelism(&self) -> usize {
        self.inner.io_parallelism()
    }
    async fn new_index_file(
        &self,
        name: &str,
        schema: Arc<arrow_schema::Schema>,
    ) -> Result<Box<dyn IndexWriter>> {
        self.inner.new_index_file(name, schema).await
    }
    async fn open_index_file(&self, name: &str) -> Result<Arc<dyn IndexReader>> {
        Ok(Arc::new(TranslatedIndexReader {
            inner: self.inner.open_index_file(name).await?,
            mapping: self.mapping.clone(),
            coverage: self.coverage.clone(),
        }))
    }
    fn with_io_priority(&self, priority: u64) -> Arc<dyn IndexStore> {
        Arc::new(Self {
            inner: self.inner.with_io_priority(priority),
            ..self.clone()
        })
    }
    async fn copy_index_file(
        &self,
        name: &str,
        _destination: &dyn IndexStore,
    ) -> Result<IndexFile> {
        Err(Error::not_supported(format!(
            "copying translated B-tree file {name} requires rebuilding its page lookup"
        )))
    }

    async fn rename_index_file(&self, name: &str, new_name: &str) -> Result<IndexFile> {
        self.inner.rename_index_file(name, new_name).await
    }
    async fn delete_index_file(&self, name: &str) -> Result<()> {
        self.inner.delete_index_file(name).await
    }
    async fn list_files_with_sizes(&self) -> Result<Vec<IndexFile>> {
        self.inner.list_files_with_sizes().await
    }
}

#[derive(Clone)]
struct TranslatedIndexReader {
    inner: Arc<dyn IndexReader>,
    mapping: Arc<MixedFragReuseIndex>,
    coverage: RoaringBitmap,
}

impl TranslatedIndexReader {
    async fn translate(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let Ok(column) = batch.schema().index_of("ids") else {
            return Ok(batch);
        };
        // B-tree page files store physical row addresses in the `ids` column.
        let ids = batch
            .column_by_name("ids")
            .ok_or_else(|| Error::invalid_input("B-tree page is missing its ids column"))?
            .as_primitive_opt::<arrow_array::types::UInt64Type>()
            .ok_or_else(|| Error::invalid_input("B-tree row addresses must be UInt64"))?;
        let addresses: Vec<RowAddress> =
            ids.values().iter().copied().map(RowAddress::from).collect();
        let translated = self.mapping.translate(&addresses).await?;
        let mut positions = Vec::with_capacity(batch.num_rows());
        let mut addresses = Vec::with_capacity(batch.num_rows());
        for (position, address) in translated.into_iter().enumerate() {
            if ids.is_valid(position)
                && let Some(address) = address
                && self.coverage.contains(address.fragment_id())
            {
                positions.push(position as u32);
                addresses.push(u64::from(address));
            }
        }
        let positions = UInt32Array::from(positions);
        let mut columns = batch
            .columns()
            .iter()
            .map(|array| arrow::compute::take(array, &positions, None))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        columns[column] = Arc::new(UInt64Array::from(addresses));
        Ok(RecordBatch::try_new(batch.schema(), columns)?)
    }
}

#[async_trait]
impl IndexReader for TranslatedIndexReader {
    async fn read_record_batch(&self, n: u64, batch_size: u64) -> Result<RecordBatch> {
        self.translate(self.inner.read_record_batch(n, batch_size).await?)
            .await
    }
    async fn read_range(
        &self,
        range: Range<usize>,
        projection: Option<&[&str]>,
    ) -> Result<RecordBatch> {
        self.translate(self.inner.read_range(range, projection).await?)
            .await
    }
    async fn read_range_stream(
        &self,
        range: Range<usize>,
        projection: Option<&[&str]>,
    ) -> Result<Pin<Box<dyn RecordBatchStream>>> {
        let stream = self.inner.read_range_stream(range, projection).await?;
        let schema = stream.schema();
        let reader = self.clone();
        let stream = stream.and_then(move |batch| {
            let reader = reader.clone();
            async move { reader.translate(batch).await }
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
    async fn read_global_buffer(&self, index: u32) -> Result<bytes::Bytes> {
        self.inner.read_global_buffer(index).await
    }
    async fn num_batches(&self, batch_size: u64) -> u32 {
        self.inner.num_batches(batch_size).await
    }
    fn num_rows(&self) -> usize {
        self.inner.num_rows()
    }
    fn schema(&self) -> &lance_core::datatypes::Schema {
        self.inner.schema()
    }
    fn file_size_bytes(&self) -> Option<u64> {
        self.inner.file_size_bytes()
    }
}

/// Remap nullable physical row addresses through the retained V1/V2 history.
///
/// Returns `None` when no mapping history exists. Input NULLs and rows dropped
/// by a rewrite remain NULL; addresses outside retained history pass through.
/// Current deletion vectors must still be applied by the caller.
///
/// ```
/// # use lance::{Dataset, Result};
/// # use arrow_array::UInt64Array;
/// # use lance::index::frag_reuse_v2::remap_row_addrs;
/// # async fn remap(dataset: &Dataset, addresses: UInt64Array) -> Result<()> {
/// let translated = remap_row_addrs(dataset, &addresses).await?;
/// # Ok(())
/// # }
/// ```
pub async fn remap_row_addrs(
    dataset: &Dataset,
    addresses: &UInt64Array,
) -> Result<Option<UInt64Array>> {
    if dataset.manifest.stable_partition_transitions.is_empty() {
        let index = dataset.open_frag_reuse_index(&NoOpMetricsCollector).await?;
        return Ok(index.map(|index| index.remap_row_ids_array(Arc::new(addresses.clone()))));
    }
    let (positions, inputs): (Vec<usize>, Vec<RowAddress>) = addresses
        .iter()
        .enumerate()
        .filter_map(|(position, value)| value.map(|value| (position, RowAddress::from(value))))
        .unzip();
    let mapping = MixedFragReuseIndex::open(dataset).await?;
    let translated = mapping.translate(&inputs).await?;
    let mut output = vec![None; addresses.len()];
    for (position, address) in positions.into_iter().zip(translated) {
        output[position] = address.map(u64::from);
    }
    Ok(Some(UInt64Array::from(output)))
}

/// Admission policy for sources that are not fully covered by an existing index.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StablePartitionCoverage {
    /// Reject a rewrite that would leave destinations uncovered by an existing index.
    #[default]
    RequireFull,
    /// Explicitly accept scanning destinations for indices with incomplete coverage.
    AllowUnindexed,
}

/// Reject another stable partition while a source still needs translated index entries.
/// This also follows V1 compactions of unresolved destinations.
pub(crate) async fn validate_partition_sources(
    dataset: &Dataset,
    operation: &Operation,
) -> Result<()> {
    let Operation::Rewrite {
        stable_partition: Some(transition),
        ..
    } = operation
    else {
        return Ok(());
    };
    let mapping = MixedFragReuseIndex::open(dataset).await?;
    let indices = read_manifest_indexes(
        &dataset.object_store,
        &dataset.manifest_location,
        &dataset.manifest,
    )
    .await?;
    let mut provenance: HashMap<&str, RoaringBitmap> = HashMap::new();
    for index in &indices {
        if index.name != FRAG_REUSE_INDEX_NAME
            && let Some(bitmap) = &index.fragment_bitmap
        {
            *provenance.entry(&index.name).or_default() |= bitmap;
        }
    }
    for (name, bitmap) in provenance {
        let unresolved = mapping.remap_fragment_bitmap(&bitmap) - mapping.direct_coverage(&bitmap);
        let affected = &unresolved & transition.source_ids();
        if !affected.is_empty() {
            return Err(Error::invalid_input(format!(
                "stable partition sources {:?} still require translated entries from index '{name}'; rebuild their indices before repartitioning",
                affected.iter().collect::<Vec<_>>()
            )));
        }
    }
    Ok(())
}

/// Commit a prepared stable partition and its immutable row map atomically.
///
/// Write `row_map.lance` under `_row_maps/<row_map_id>` using `RowMapWriter`
/// first. Sources must be from this dataset snapshot, destinations must have
/// reserved IDs, and their rows must match the stable-partition label order.
///
/// ```
/// # use lance::{Dataset, Result};
/// # use lance::index::frag_reuse_v2::{commit_stable_partition, StablePartitionCoverage};
/// # use lance_table::format::Fragment;
/// # use uuid::Uuid;
/// # async fn install(dataset: &mut Dataset, sources: Vec<Fragment>, destinations: Vec<Fragment>,
/// #                  row_map_id: Uuid, file_size: u64) -> Result<()> {
/// commit_stable_partition(dataset, sources, destinations, row_map_id, file_size, StablePartitionCoverage::RequireFull).await
/// # }
/// ```
pub async fn commit_stable_partition(
    dataset: &mut Dataset,
    sources: Vec<Fragment>,
    destinations: Vec<Fragment>,
    row_map_id: Uuid,
    row_map_size_bytes: u64,
    coverage: StablePartitionCoverage,
) -> Result<()> {
    if coverage == StablePartitionCoverage::RequireFull {
        let mut indexed: HashMap<String, RoaringBitmap> = HashMap::new();
        for index in dataset.load_indices().await?.iter() {
            if index.name != FRAG_REUSE_INDEX_NAME {
                *indexed.entry(index.name.clone()).or_default() |=
                    index.fragment_bitmap.clone().unwrap_or_default();
            }
        }
        // Include affected unsupported indices: hiding them from queries must not
        // turn a request to preserve their coverage into implicit permission to scan.
        for index in read_manifest_indexes(
            &dataset.object_store,
            &dataset.manifest_location,
            &dataset.manifest,
        )
        .await?
        {
            if index.name != FRAG_REUSE_INDEX_NAME {
                indexed.entry(index.name).or_default();
            }
        }
        let source_ids: RoaringBitmap = sources.iter().map(|fragment| fragment.id as u32).collect();
        for (name, bitmap) in indexed {
            if !source_ids.is_subset(&bitmap) {
                return Err(Error::invalid_input(format!(
                    "index '{name}' does not cover every stable-partition source; use StablePartitionCoverage::AllowUnindexed to accept scan fallback"
                )));
            }
        }
    }
    let digest = |fragment: &Fragment| -> Result<FragDigest> {
        if fragment
            .deletion_file
            .as_ref()
            .is_some_and(|file| file.num_deleted_rows.is_none())
        {
            return Err(Error::invalid_input(format!(
                "fragment {} is missing its deleted row count",
                fragment.id,
            )));
        }
        Ok(FragDigest {
            id: fragment.id,
            physical_rows: fragment.physical_rows.ok_or_else(|| {
                Error::invalid_input(format!(
                    "fragment {} has no physical row count",
                    fragment.id
                ))
            })?,
            num_deleted_rows: fragment
                .deletion_file
                .as_ref()
                .and_then(|d| d.num_deleted_rows)
                .unwrap_or(0),
        })
    };
    let transition = StablePartitionTransition {
        source_dataset_version: dataset.manifest.version,
        sources: sources.iter().map(digest).collect::<Result<_>>()?,
        destinations: destinations.iter().map(digest).collect::<Result<_>>()?,
        row_map_id,
        row_map_size_bytes,
        base_id: None,
        committed_version: 0,
    };
    transition.validate()?;
    let store = LanceIndexStore::new(
        dataset.object_store.clone(),
        dataset.row_maps_dir().join(row_map_id.to_string()),
        Arc::new(lance_core::cache::LanceCache::with_capacity(1024 * 1024)),
    );
    let file = store.open_index_file(ROW_MAP_FILE_NAME).await?;
    if file
        .file_size_bytes()
        .is_some_and(|size| size != row_map_size_bytes)
    {
        return Err(Error::invalid_input(format!(
            "row map {row_map_id} file size {:?} differs from declared size {row_map_size_bytes}",
            file.file_size_bytes()
        )));
    }
    let reader = RowMapReader::open(file).await?;
    validate_counts(&transition, reader.counts())?;
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::Rewrite {
            groups: vec![RewriteGroup {
                old_fragments: sources,
                new_fragments: destinations,
            }],
            rewritten_indices: Vec::new(),
            frag_reuse_index: None,
            stable_partition: Some(Box::new(transition)),
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::cleanup::{CleanupPolicyBuilder, cleanup_old_versions};
    use crate::dataset::index::frag_reuse::cleanup_frag_reuse_index;
    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::dataset::{InsertBuilder, WriteMode, WriteParams};
    use crate::index::DatasetIndexExt;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::types::Int32Type;
    use lance_index::IndexType;
    use lance_index::frag_reuse::row_map::{RowMapWriter, SourceRows};
    use lance_index::scalar::ScalarIndexParams;

    async fn prepare(dataset: &mut Dataset) -> (Vec<Fragment>, Vec<Fragment>, Uuid, u64) {
        prepare_sources(dataset, dataset.fragments().to_vec()).await
    }

    async fn prepare_sources(
        dataset: &mut Dataset,
        sources: Vec<Fragment>,
    ) -> (Vec<Fragment>, Vec<Fragment>, Uuid, u64) {
        let batch = dataset
            .scan()
            .with_fragments(sources.clone())
            .try_into_batch()
            .await
            .unwrap();
        let values = batch["i"].as_primitive::<Int32Type>();
        let labels: Vec<u16> = values.iter().map(|v| (v.unwrap_or(0) % 2) as u16).collect();
        let mut destinations = Vec::new();
        for label in 0..2 {
            let positions = UInt32Array::from(
                labels
                    .iter()
                    .enumerate()
                    .filter_map(|(position, &destination)| {
                        (destination == label).then_some(position as u32)
                    })
                    .collect::<Vec<_>>(),
            );
            let batch = RecordBatch::try_new(
                batch.schema(),
                vec![arrow::compute::take(values, &positions, None).unwrap()],
            )
            .unwrap();
            let transaction = InsertBuilder::new(Arc::new(dataset.clone()))
                .with_params(&WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                })
                .execute_uncommitted(vec![batch])
                .await
                .unwrap();
            let Operation::Append { fragments } = transaction.operation else {
                panic!("expected append")
            };
            destinations.extend(fragments);
        }
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::ReserveFragments {
                        num_fragments: destinations.len() as u32,
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();
        let first = dataset.manifest.max_fragment_id.unwrap() + 1 - destinations.len() as u32;
        for (i, destination) in destinations.iter_mut().enumerate() {
            destination.id = u64::from(first) + i as u64;
        }
        let mut source_rows = Vec::new();
        for source in &sources {
            let fragment = dataset.get_fragment(source.id as usize).unwrap();
            let deleted = fragment
                .get_deletion_vector()
                .await
                .unwrap()
                .map(|v| v.iter().collect());
            source_rows.push(SourceRows {
                physical_rows: fragment.metadata().physical_rows.unwrap() as u64,
                deleted,
            });
        }
        let id = Uuid::new_v4();
        let store = LanceIndexStore::with_format_version(
            dataset.object_store.clone(),
            dataset.row_maps_dir().join(id.to_string()),
            Arc::new(lance_core::cache::LanceCache::with_capacity(1024 * 1024)),
            lance_file::version::ConcreteFileVersion::V2_1,
        );
        let writer = store
            .new_index_file(ROW_MAP_FILE_NAME, RowMapWriter::schema())
            .await
            .unwrap();
        let mut writer = RowMapWriter::try_new_with_block_rows(writer, source_rows, 2, 1).unwrap();
        writer.append_labels(&labels).await.unwrap();
        let (file, _) = writer.finish().await.unwrap();
        (sources, destinations, id, file.size_bytes)
    }

    async fn dataset() -> Dataset {
        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(4), FragmentRowCount::from(4))
            .await
            .unwrap();
        // The generated values are already sorted. Avoid making tiny test
        // fixtures compete for DataFusion's shared sort-spill reservations.
        let batch = dataset
            .scan()
            .with_row_id()
            .project_with_transform(&[("value", "i")])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let reader = arrow_array::RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let params = ScalarIndexParams::default();
        let index = crate::index::create::CreateIndexBuilder::new(
            &mut dataset,
            &["i"],
            IndexType::BTree,
            &params,
        )
        .name("i_idx".to_string())
        .preprocessed_data(Box::new(reader))
        .execute_uncommitted()
        .await
        .unwrap();
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::CreateIndex {
                        new_indices: vec![index],
                        removed_indices: Vec::new(),
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();
        dataset
    }

    #[rstest::rstest]
    #[case::deferred(true)]
    #[case::eager(false)]
    #[tokio::test]
    async fn mixed_rewrites_preserve_rows_and_index_queries(#[case] deferred: bool) {
        let mut dataset = dataset().await;
        let original = dataset
            .scan()
            .with_row_address()
            .try_into_batch()
            .await
            .unwrap();
        let original_addresses: Vec<RowAddress> = original[lance_core::ROW_ADDR]
            .as_primitive::<arrow_array::types::UInt64Type>()
            .values()
            .iter()
            .copied()
            .map(RowAddress::from)
            .collect();
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 8,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        dataset.delete("i = 3").await.unwrap();
        let (sources, destinations, id, size) = prepare(&mut dataset).await;
        commit_stable_partition(
            &mut dataset,
            sources,
            destinations,
            id,
            size,
            StablePartitionCoverage::RequireFull,
        )
        .await
        .unwrap();
        assert_eq!(dataset.manifest.stable_partition_transitions.len(), 1);
        let index = dataset.load_index_by_name("i_idx").await.unwrap().unwrap();
        assert_eq!(
            index.fragment_bitmap.as_ref().unwrap(),
            dataset.fragment_bitmap.as_ref()
        );
        let plan = dataset
            .scan()
            .filter("i = 2")
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert!(plan.contains("ScalarIndexQuery"), "{plan}");
        for value in [0, 2, 3, 8, 15] {
            assert_eq!(
                dataset
                    .count_rows(Some(format!("i = {value}")))
                    .await
                    .unwrap(),
                usize::from(value != 3)
            );
        }
        dataset.delete("i = 8").await.unwrap();
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 32,
                defer_index_remap: deferred,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        let mapping = MixedFragReuseIndex::open(&dataset).await.unwrap();
        let translated = mapping.translate(&original_addresses).await.unwrap();
        let nullable = UInt64Array::from(vec![
            Some(u64::from(original_addresses[0])),
            None,
            Some(u64::from(original_addresses[3])),
        ]);
        assert_eq!(
            remap_row_addrs(&dataset, &nullable).await.unwrap().unwrap(),
            UInt64Array::from(vec![
                translated[0].map(u64::from),
                None,
                translated[3].map(u64::from)
            ])
        );
        let current = dataset
            .scan()
            .with_row_address()
            .try_into_batch()
            .await
            .unwrap();
        let values = current["i"].as_primitive::<Int32Type>();
        let current_addresses =
            current[lance_core::ROW_ADDR].as_primitive::<arrow_array::types::UInt64Type>();
        let actual: HashMap<i32, u64> = values
            .values()
            .iter()
            .copied()
            .zip(current_addresses.values().iter().copied())
            .collect();
        for (value, translated) in translated.iter().enumerate() {
            assert_eq!(
                translated.map(u64::from),
                actual.get(&(value as i32)).copied()
            );
        }
        assert_eq!(dataset.count_rows(Some("i >= 0".into())).await.unwrap(), 14);
        let versions = load_frag_reuse_index_details(
            &dataset,
            &dataset
                .load_index_by_name(FRAG_REUSE_INDEX_NAME)
                .await
                .unwrap()
                .unwrap(),
        )
        .await
        .unwrap()
        .versions
        .len();
        cleanup_frag_reuse_index(&mut dataset).await.unwrap();
        assert_eq!(
            load_frag_reuse_index_details(
                &dataset,
                &dataset
                    .load_index_by_name(FRAG_REUSE_INDEX_NAME)
                    .await
                    .unwrap()
                    .unwrap()
            )
            .await
            .unwrap()
            .versions
            .len(),
            versions
        );
        cleanup_old_versions(
            &dataset,
            CleanupPolicyBuilder::default()
                .delete_unverified(true)
                .build(),
        )
        .await
        .unwrap();
        let reopened = dataset
            .checkout_version(dataset.manifest.version)
            .await
            .unwrap();
        assert_eq!(
            MixedFragReuseIndex::open(&reopened)
                .await
                .unwrap()
                .translate(&original_addresses)
                .await
                .unwrap(),
            translated
        );
        assert_eq!(reopened.count_rows(Some("i = 15".into())).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn segmented_coverage_and_direct_destinations() {
        let mut dataset = dataset().await;
        dataset.drop_index("i_idx").await.unwrap();
        let params = ScalarIndexParams::default();
        let mut segments = Vec::new();
        for fragments in [vec![0, 1], vec![2, 3]] {
            let sources = dataset
                .fragments()
                .iter()
                .filter(|fragment| fragments.contains(&(fragment.id as u32)))
                .cloned()
                .collect();
            let batch = dataset
                .scan()
                .with_fragments(sources)
                .with_row_id()
                .project_with_transform(&[("value", "i")])
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            let reader =
                arrow_array::RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
            segments.push(
                dataset
                    .create_index_builder(&["i"], IndexType::BTree, &params)
                    .name("i_idx".to_string())
                    .fragments(fragments)
                    .preprocessed_data(Box::new(reader))
                    .execute_uncommitted()
                    .await
                    .unwrap(),
            );
        }
        dataset
            .commit_existing_index_segments("i_idx", "i", segments)
            .await
            .unwrap();
        let original = dataset.load_indices_by_name("i_idx").await.unwrap();
        let (sources, destinations, id, size) = prepare(&mut dataset).await;
        commit_stable_partition(
            &mut dataset,
            sources,
            destinations,
            id,
            size,
            StablePartitionCoverage::RequireFull,
        )
        .await
        .unwrap();
        let derived = dataset.load_indices_by_name("i_idx").await.unwrap();
        assert_eq!(derived.len(), 2);
        for segment in &derived {
            assert_eq!(
                segment.fragment_bitmap.as_ref().unwrap(),
                dataset.fragment_bitmap.as_ref()
            );
        }
        for value in 0..16 {
            assert_eq!(
                dataset
                    .count_rows(Some(format!("i = {value}")))
                    .await
                    .unwrap(),
                1
            );
        }
        let plan = dataset
            .scan()
            .filter("i = 2")
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert!(plan.contains("ScalarIndexQuery"), "{plan}");
        let persisted = read_manifest_indexes(
            &dataset.object_store,
            &dataset.manifest_location,
            &dataset.manifest,
        )
        .await
        .unwrap();
        for segment in &original {
            assert_eq!(
                persisted
                    .iter()
                    .find(|index| index.uuid == segment.uuid)
                    .unwrap()
                    .fragment_bitmap,
                segment.fragment_bitmap
            );
        }
        let destination = dataset.fragments()[0].clone();
        let batch = dataset
            .scan()
            .with_fragments(vec![destination.clone()])
            .with_row_id()
            .project_with_transform(&[("value", "i")])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let reader = arrow_array::RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let direct = dataset
            .create_index_builder(&["i"], IndexType::BTree, &params)
            .name("i_idx".to_string())
            .replace(true)
            .fragments(vec![destination.id as u32])
            .preprocessed_data(Box::new(reader))
            .execute_uncommitted()
            .await
            .unwrap();
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::CreateIndex {
                        new_indices: vec![direct],
                        removed_indices: vec![],
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();
        let segments = dataset.load_indices_by_name("i_idx").await.unwrap();
        assert_eq!(segments.len(), 3);
        for segment in &segments {
            if original.iter().any(|old| old.uuid == segment.uuid) {
                assert!(
                    !segment
                        .fragment_bitmap
                        .as_ref()
                        .unwrap()
                        .contains(destination.id as u32)
                );
            }
        }
        for value in 0..16 {
            assert_eq!(
                dataset
                    .count_rows(Some(format!("i = {value}")))
                    .await
                    .unwrap(),
                1
            );
        }
        // A later ordinary commit must not persist derived, overlapping bitmaps.
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 32,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        let persisted = read_manifest_indexes(
            &dataset.object_store,
            &dataset.manifest_location,
            &dataset.manifest,
        )
        .await
        .unwrap();
        for segment in &original {
            assert_eq!(
                persisted
                    .iter()
                    .find(|index| index.uuid == segment.uuid)
                    .unwrap()
                    .fragment_bitmap,
                segment.fragment_bitmap
            );
        }
        assert_eq!(dataset.count_rows(Some("i >= 0".into())).await.unwrap(), 16);
        for value in 0..16 {
            assert_eq!(
                dataset
                    .count_rows(Some(format!("i = {value}")))
                    .await
                    .unwrap(),
                1
            );
        }
    }

    #[tokio::test]
    async fn append_and_repeated_partitions_keep_partial_index_coverage_safe() {
        let mut dataset = dataset().await;
        let (sources, destinations, id, size) = prepare(&mut dataset).await;
        commit_stable_partition(
            &mut dataset,
            sources,
            destinations,
            id,
            size,
            StablePartitionCoverage::RequireFull,
        )
        .await
        .unwrap();
        let (sources, destinations, id, size) = prepare(&mut dataset).await;
        let error = commit_stable_partition(
            &mut dataset,
            sources,
            destinations,
            id,
            size,
            StablePartitionCoverage::AllowUnindexed,
        )
        .await
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("still require translated entries")
        );
        // Rebuild direct coverage before another order-changing rewrite.
        dataset
            .create_index(
                &["i"],
                IndexType::BTree,
                Some("i_idx".into()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        let batch = arrow_array::record_batch!(("i", Int32, [16, 17])).unwrap();
        let transaction = InsertBuilder::new(Arc::new(dataset.clone()))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute_uncommitted(vec![batch])
            .await
            .unwrap();
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();
        assert_eq!(dataset.manifest.stable_partition_transitions.len(), 1);
        assert_eq!(dataset.count_rows(Some("i >= 0".into())).await.unwrap(), 18);
        assert_eq!(dataset.count_rows(Some("i = 17".into())).await.unwrap(), 1);
        let before = dataset
            .scan()
            .with_row_address()
            .try_into_batch()
            .await
            .unwrap();
        let (sources, destinations, id, size) = prepare(&mut dataset).await;
        let error = commit_stable_partition(
            &mut dataset,
            sources.clone(),
            destinations.clone(),
            id,
            size,
            StablePartitionCoverage::RequireFull,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("AllowUnindexed"));
        commit_stable_partition(
            &mut dataset,
            sources,
            destinations,
            id,
            size,
            StablePartitionCoverage::AllowUnindexed,
        )
        .await
        .unwrap();
        let addresses = before[lance_core::ROW_ADDR]
            .as_primitive::<arrow_array::types::UInt64Type>()
            .values()
            .iter()
            .copied()
            .map(RowAddress::from)
            .collect::<Vec<_>>();
        let mapped = MixedFragReuseIndex::open(&dataset)
            .await
            .unwrap()
            .translate(&addresses)
            .await
            .unwrap();
        let current = dataset
            .scan()
            .with_row_address()
            .try_into_batch()
            .await
            .unwrap();
        let actual = current["i"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .zip(
                current[lance_core::ROW_ADDR]
                    .as_primitive::<arrow_array::types::UInt64Type>()
                    .values()
                    .iter()
                    .copied(),
            )
            .collect::<HashMap<_, _>>();
        for (value, address) in before["i"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .zip(mapped)
        {
            assert_eq!(address.map(u64::from), actual.get(value).copied());
        }
        assert_eq!(dataset.manifest.stable_partition_transitions.len(), 2);
        // Appended rows were never indexed, so mixed destinations cannot claim full coverage.
        let index = dataset.load_index_by_name("i_idx").await.unwrap().unwrap();
        assert!(index.fragment_bitmap.as_ref().unwrap().is_empty());
        assert_eq!(dataset.count_rows(Some("i >= 0".into())).await.unwrap(), 18);
        assert_eq!(dataset.count_rows(Some("i = 17".into())).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn unsupported_index_falls_back_without_losing_its_metadata() {
        let mut dataset = dataset().await;
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                Some("bitmap".to_string()),
                &ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::Bitmap),
                false,
            )
            .await
            .unwrap();
        let (sources, destinations, id, size) = prepare(&mut dataset).await;
        commit_stable_partition(
            &mut dataset,
            sources,
            destinations,
            id,
            size,
            StablePartitionCoverage::RequireFull,
        )
        .await
        .unwrap();
        assert!(
            dataset
                .load_index_by_name("bitmap")
                .await
                .unwrap()
                .is_none()
        );
        let all = crate::index::load_all_indices(&dataset).await.unwrap();
        assert!(all.iter().any(|index| index.name == "bitmap"));
        dataset.drop_index("i_idx").await.unwrap();
        let plan = dataset
            .scan()
            .filter("i = 5")
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert!(!plan.contains("ScalarIndexQuery"), "{plan}");
        assert_eq!(dataset.count_rows(Some("i = 5".into())).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn disjoint_partitions_rebase_and_preserve_both_row_maps() {
        let mut dataset = dataset().await;
        let first_sources = dataset.fragments()[..2].to_vec();
        let second_sources = dataset.fragments()[2..].to_vec();
        let (sources_a, destinations_a, id_a, size_a) =
            prepare_sources(&mut dataset, first_sources).await;
        let (sources_b, destinations_b, id_b, size_b) =
            prepare_sources(&mut dataset, second_sources).await;
        let mut stale = dataset.clone();
        commit_stable_partition(
            &mut dataset,
            sources_a,
            destinations_a,
            id_a,
            size_a,
            StablePartitionCoverage::RequireFull,
        )
        .await
        .unwrap();
        commit_stable_partition(
            &mut stale,
            sources_b,
            destinations_b,
            id_b,
            size_b,
            StablePartitionCoverage::RequireFull,
        )
        .await
        .unwrap();
        assert_eq!(stale.manifest.stable_partition_transitions.len(), 2);
        assert_eq!(stale.count_rows(Some("i >= 0".into())).await.unwrap(), 16);
        for value in [1, 5, 9, 13] {
            assert_eq!(
                stale
                    .count_rows(Some(format!("i = {value}")))
                    .await
                    .unwrap(),
                1
            );
        }
    }

    #[tokio::test]
    async fn cleanup_retains_committed_maps_and_removes_abandoned_maps() {
        let mut dataset = dataset().await;
        let (sources, destinations, committed, size) = prepare(&mut dataset).await;
        commit_stable_partition(
            &mut dataset,
            sources,
            destinations,
            committed,
            size,
            StablePartitionCoverage::RequireFull,
        )
        .await
        .unwrap();
        let (_, _, abandoned, _) = prepare(&mut dataset).await;
        let committed_path = dataset
            .row_maps_dir()
            .join(committed.to_string())
            .join(ROW_MAP_FILE_NAME);
        let abandoned_path = dataset
            .row_maps_dir()
            .join(abandoned.to_string())
            .join(ROW_MAP_FILE_NAME);
        cleanup_old_versions(&dataset, CleanupPolicyBuilder::default().build())
            .await
            .unwrap();
        assert!(dataset.object_store.size(&abandoned_path).await.is_ok());
        cleanup_old_versions(
            &dataset,
            CleanupPolicyBuilder::default()
                .delete_unverified(true)
                .build(),
        )
        .await
        .unwrap();
        assert_eq!(
            dataset.object_store.size(&committed_path).await.unwrap(),
            size
        );
        let error = dataset
            .object_store
            .size(&abandoned_path)
            .await
            .unwrap_err();
        assert!(error.is_not_found(), "{error}");
        assert_eq!(dataset.count_rows(Some("i = 9".into())).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn indexed_null_values_survive_partition_and_delete() {
        let batch = arrow_array::record_batch!((
            "i",
            Int32,
            [Some(0), None, Some(1), None, Some(2), Some(3)]
        ))
        .unwrap();
        let reader = arrow_array::RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let mut dataset = Dataset::write(
            reader,
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                Some("nullable".to_string()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        assert_eq!(
            dataset.count_rows(Some("i IS NULL".into())).await.unwrap(),
            2
        );
        let (sources, destinations, id, size) = prepare(&mut dataset).await;
        commit_stable_partition(
            &mut dataset,
            sources,
            destinations,
            id,
            size,
            StablePartitionCoverage::RequireFull,
        )
        .await
        .unwrap();
        assert_eq!(
            dataset.count_rows(Some("i IS NULL".into())).await.unwrap(),
            2
        );
        assert_eq!(
            dataset
                .count_rows(Some("i IS NOT NULL".into()))
                .await
                .unwrap(),
            4
        );
        dataset.delete("i IS NULL").await.unwrap();
        assert_eq!(
            dataset.count_rows(Some("i IS NULL".into())).await.unwrap(),
            0
        );
        assert_eq!(
            dataset
                .count_rows(Some("i IS NOT NULL".into()))
                .await
                .unwrap(),
            4
        );
    }

    #[tokio::test]
    async fn stale_source_delete_rejects_partition() {
        let mut dataset = dataset().await;
        let (sources, destinations, id, size) = prepare(&mut dataset).await;
        let mut stale = dataset.clone();
        dataset.delete("i = 1").await.unwrap();
        let error = commit_stable_partition(
            &mut stale,
            sources,
            destinations,
            id,
            size,
            StablePartitionCoverage::RequireFull,
        )
        .await
        .unwrap_err();
        assert!(matches!(
            error,
            Error::RetryableCommitConflict { .. } | Error::InvalidInput { .. }
        ));
        assert_eq!(dataset.count_rows(Some("i = 1".into())).await.unwrap(), 0);
        assert!(dataset.manifest.stable_partition_transitions.is_empty());
    }
}
