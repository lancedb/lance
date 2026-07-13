// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Metadata for index

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Cursor;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use futures::StreamExt;
use lance_core::deepsize::DeepSizeOf;
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use prost::Message;
use roaring::RoaringBitmap;
use uuid::Uuid;

use super::{
    FieldGeneration, LogicalIndexCoverage, LogicalIndexCoverageCloneProvenance,
    LogicalIndexCoverageFile, LogicalIndexCoverageShard, LogicalRowAddressSelection,
    RowAddressLogicalDomain, RowReferenceDomain, pb,
};
use lance_core::cache::{CacheEntryReader, CacheEntryWriter};
use lance_core::{Error, Result};

/// Metadata about a single file within an index segment.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct IndexFile {
    /// Path relative to the index directory (e.g., "index.idx", "auxiliary.idx")
    pub path: String,
    /// Size of the file in bytes
    pub size_bytes: u64,
}

/// Index metadata
#[derive(Debug, Clone, PartialEq)]
pub struct IndexMetadata {
    /// Unique ID across all dataset versions.
    pub uuid: Uuid,

    /// Fields to build the index.
    pub fields: Vec<i32>,

    /// Human readable index name
    pub name: String,

    /// The version of the dataset this index was last updated on
    ///
    /// This is set when the index is created (based on the version used to train the index)
    /// This is updated when the index is updated or remapped
    pub dataset_version: u64,

    /// The fragment ids this index covers.
    ///
    /// This may contain fragment ids that no longer exist in the dataset.
    ///
    /// If this is None, then this is unknown.
    pub fragment_bitmap: Option<RoaringBitmap>,

    /// Metadata specific to the index type
    ///
    /// This is an Option because older versions of Lance may not have this defined.  However, it should always
    /// be present in newer versions.
    pub index_details: Option<Arc<prost_types::Any>>,

    /// The index version.
    pub index_version: i32,

    /// Timestamp when the index was created
    ///
    /// This field is optional for backward compatibility. For existing indices created before
    /// this field was added, this will be None.
    pub created_at: Option<DateTime<Utc>>,

    /// The base path index of the index files. Used when the index is imported or referred from another dataset.
    /// Lance uses it as key of the base_paths field in Manifest to determine the actual base path of the index files.
    pub base_id: Option<u32>,

    /// List of files and their sizes for this index segment.
    /// This enables skipping HEAD calls when opening indices and provides
    /// visibility into index storage size via describe_indices().
    /// This is None if the file sizes are unknown. This happens for indices created
    /// before this field was added.
    pub files: Option<Vec<IndexFile>>,

    /// Row-reference domain for storage-version-2.3 index postings.
    pub row_reference_domain: Option<RowReferenceDomain>,

    /// Exact stable logical row-address coverage for storage version 2.3.
    pub logical_coverage: Option<LogicalIndexCoverage>,
}

impl IndexMetadata {
    pub fn effective_fragment_bitmap(
        &self,
        existing_fragments: &RoaringBitmap,
    ) -> Option<RoaringBitmap> {
        let fragment_bitmap = self.fragment_bitmap.as_ref()?;
        Some(fragment_bitmap & existing_fragments)
    }

    /// Returns a map of relative file paths to their sizes.
    /// Returns an empty map if file information is not available.
    pub fn file_size_map(&self) -> HashMap<String, u64> {
        self.files
            .as_ref()
            .map(|files| {
                files
                    .iter()
                    .map(|f| (f.path.clone(), f.size_bytes))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns the total size of all files in this index segment in bytes.
    /// Returns None if file information is not available.
    pub fn total_size_bytes(&self) -> Option<u64> {
        self.files
            .as_ref()
            .map(|files| files.iter().map(|f| f.size_bytes).sum())
    }

    /// Returns the set of fragments which are part of the fragment bitmap
    /// but no longer in the dataset.
    pub fn deleted_fragment_bitmap(
        &self,
        existing_fragments: &RoaringBitmap,
    ) -> Option<RoaringBitmap> {
        let fragment_bitmap = self.fragment_bitmap.as_ref()?;
        Some(fragment_bitmap - existing_fragments)
    }
}

const LOGICAL_COVERAGE_FINGERPRINT_SIZE: usize = 16;
pub const LOGICAL_INDEX_COVERAGE_ARTIFACT_ENCODING_VERSION: u32 = 1;

/// Core-authored coverage provenance persisted in an existing index file.
///
/// This is the commit-time source of truth for storage-version-2.3 coverage.
/// Coverage carried by a staged-index API request is only checked against this
/// artifact and is never admitted independently.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalIndexCoverageArtifact {
    pub namespace_uuid: Uuid,
    pub index_uuid: Uuid,
    pub build_dataset_version: u64,
    pub source_layout_fingerprint: Vec<u8>,
    pub logical_domains: Vec<RowAddressLogicalDomain>,
    pub field_default_generations: Vec<FieldGeneration>,
    pub index_commit_floors: Vec<FieldGeneration>,
    pub coverage: LogicalIndexCoverage,
    pub fingerprint: Vec<u8>,
    pub object_id: Uuid,
    pub field_ids: Vec<i32>,
}

impl DeepSizeOf for LogicalIndexCoverageArtifact {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.source_layout_fingerprint
            .deep_size_of_children(context)
            + self.logical_domains.deep_size_of_children(context)
            + self
                .field_default_generations
                .deep_size_of_children(context)
            + self.index_commit_floors.deep_size_of_children(context)
            + self.coverage.deep_size_of_children(context)
            + self.fingerprint.deep_size_of_children(context)
            + self.field_ids.deep_size_of_children(context)
    }
}

fn logical_coverage_fingerprint(bytes: &[u8]) -> Vec<u8> {
    const OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
    let mut hash = OFFSET;
    for byte in bytes {
        hash ^= *byte as u128;
        hash = hash.wrapping_mul(PRIME);
    }
    hash.to_le_bytes().to_vec()
}

fn coverage_shard_fingerprint(shard: &LogicalIndexCoverageShard) -> Result<Vec<u8>> {
    let selection = shard
        .selection
        .as_ref()
        .ok_or_else(|| Error::invalid_input("logical coverage detail is external"))?
        .canonical_proto()
        .encode_to_vec();
    let mut bytes = Vec::with_capacity(
        32 + selection.len() + shard.field_ids.len() * 4 + shard.validated_through.len() * 12,
    );
    bytes.extend_from_slice(b"lance.logical-index-coverage-shard.v1\0");
    bytes.extend_from_slice(&(selection.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&selection);
    bytes.extend_from_slice(&shard.row_count.to_le_bytes());
    bytes.extend_from_slice(&(shard.field_ids.len() as u64).to_le_bytes());
    for field_id in &shard.field_ids {
        bytes.extend_from_slice(&field_id.to_le_bytes());
    }
    bytes.extend_from_slice(&(shard.validated_through.len() as u64).to_le_bytes());
    for generation in &shard.validated_through {
        bytes.extend_from_slice(&generation.field_id.to_le_bytes());
        bytes.extend_from_slice(&generation.generation.to_le_bytes());
    }
    Ok(logical_coverage_fingerprint(&bytes))
}

fn encode_logical_fragment_bitmap(selection: &LogicalRowAddressSelection) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    selection
        .logical_fragment_bitmap()?
        .serialize_into(&mut bytes)?;
    Ok(bytes)
}

fn decode_logical_fragment_bitmap(bytes: &[u8]) -> Result<RoaringBitmap> {
    if bytes.is_empty() {
        return Ok(RoaringBitmap::new());
    }
    let mut cursor = Cursor::new(bytes);
    let bitmap = RoaringBitmap::deserialize_from(&mut cursor)?;
    if cursor.position() != bytes.len() as u64 {
        return Err(Error::invalid_input(
            "logical index coverage fragment bitmap has trailing bytes",
        ));
    }
    Ok(bitmap)
}

fn append_coverage_shard_summary(bytes: &mut Vec<u8>, shard: &LogicalIndexCoverageShard) {
    append_length_prefixed(bytes, &shard.fingerprint);
    bytes.extend_from_slice(&shard.row_count.to_le_bytes());
    bytes.extend_from_slice(&(shard.field_ids.len() as u64).to_le_bytes());
    for field_id in &shard.field_ids {
        bytes.extend_from_slice(&field_id.to_le_bytes());
    }
    bytes.extend_from_slice(&(shard.validated_through.len() as u64).to_le_bytes());
    for generation in &shard.validated_through {
        bytes.extend_from_slice(&generation.field_id.to_le_bytes());
        bytes.extend_from_slice(&generation.generation.to_le_bytes());
    }
    append_length_prefixed(bytes, &shard.logical_fragment_bitmap);
}

fn coverage_root_fingerprint(shards: &[LogicalIndexCoverageShard]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(48 + shards.len() * LOGICAL_COVERAGE_FINGERPRINT_SIZE);
    bytes.extend_from_slice(b"lance.logical-index-coverage.v1\0");
    bytes.extend_from_slice(&(shards.len() as u64).to_le_bytes());
    for shard in shards {
        append_coverage_shard_summary(&mut bytes, shard);
    }
    logical_coverage_fingerprint(&bytes)
}

fn canonical_index_files(files: &[IndexFile]) -> Result<Vec<&IndexFile>> {
    let mut files = files.iter().collect::<Vec<_>>();
    files.sort_unstable_by(|left, right| left.path.cmp(&right.path));
    if files.is_empty()
        || files
            .iter()
            .any(|file| file.path.is_empty() || file.size_bytes == 0)
        || files.windows(2).any(|pair| pair[0].path == pair[1].path)
    {
        return Err(Error::invalid_input(
            "logical index coverage requires canonical non-empty index files",
        ));
    }
    Ok(files)
}

fn append_length_prefixed(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

fn bound_coverage_root_fingerprint(
    shards: &[LogicalIndexCoverageShard],
    namespace_uuid: Uuid,
    index_uuid: Uuid,
    external: &LogicalIndexCoverageFile,
    clone_provenance: Option<&LogicalIndexCoverageCloneProvenance>,
    files: &[IndexFile],
) -> Result<Vec<u8>> {
    let files = canonical_index_files(files)?;
    let mut bytes = Vec::with_capacity(
        160 + shards.len() * LOGICAL_COVERAGE_FINGERPRINT_SIZE
            + files.iter().map(|file| file.path.len() + 16).sum::<usize>(),
    );
    bytes.extend_from_slice(b"lance.logical-index-coverage-bound.v2\0");
    bytes.extend_from_slice(namespace_uuid.as_bytes());
    bytes.extend_from_slice(index_uuid.as_bytes());
    bytes.extend_from_slice(&(shards.len() as u64).to_le_bytes());
    for shard in shards {
        append_coverage_shard_summary(&mut bytes, shard);
    }
    append_length_prefixed(&mut bytes, external.path.as_bytes());
    bytes.extend_from_slice(&external.offset.to_le_bytes());
    bytes.extend_from_slice(&external.byte_length.to_le_bytes());
    bytes.extend_from_slice(&external.global_buffer_index.to_le_bytes());
    bytes.extend_from_slice(&external.object_size.to_le_bytes());
    bytes.extend_from_slice(external.object_id.as_bytes());
    bytes.extend_from_slice(external.artifact_namespace_uuid.as_bytes());
    append_length_prefixed(&mut bytes, &external.artifact_layout_fingerprint);
    match clone_provenance {
        Some(provenance) => {
            bytes.push(1);
            bytes.extend_from_slice(provenance.source_namespace_uuid.as_bytes());
            bytes.extend_from_slice(provenance.target_namespace_uuid.as_bytes());
            append_length_prefixed(&mut bytes, &provenance.source_coverage_fingerprint);
            bytes.extend_from_slice(provenance.transaction_uuid.as_bytes());
            bytes.extend_from_slice(&provenance.depth.to_le_bytes());
            bytes.push(u8::from(provenance.is_shallow));
            bytes.extend_from_slice(&provenance.source_manifest_version.to_le_bytes());
        }
        None => bytes.push(0),
    }
    bytes.extend_from_slice(&(files.len() as u64).to_le_bytes());
    for file in files {
        append_length_prefixed(&mut bytes, file.path.as_bytes());
        bytes.extend_from_slice(&file.size_bytes.to_le_bytes());
    }
    Ok(logical_coverage_fingerprint(&bytes))
}

fn coverage_artifact_fingerprint(artifact: &LogicalIndexCoverageArtifact) -> Vec<u8> {
    let mut proto = pb::LogicalIndexCoverageArtifact::from(artifact);
    proto.fingerprint.clear();
    logical_coverage_fingerprint(&proto.encode_to_vec())
}

impl LogicalIndexCoverageArtifact {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        namespace_uuid: Uuid,
        index_uuid: Uuid,
        build_dataset_version: u64,
        source_layout_fingerprint: Vec<u8>,
        mut logical_domains: Vec<RowAddressLogicalDomain>,
        mut field_default_generations: Vec<FieldGeneration>,
        mut index_commit_floors: Vec<FieldGeneration>,
        coverage: LogicalIndexCoverage,
        object_id: Uuid,
        mut field_ids: Vec<i32>,
    ) -> Result<Self> {
        logical_domains.sort_unstable_by_key(|domain| domain.logical_fragment_id);
        field_default_generations.sort_unstable();
        index_commit_floors.sort_unstable();
        field_ids.sort_unstable();
        field_ids.dedup();
        let mut artifact = Self {
            namespace_uuid,
            index_uuid,
            build_dataset_version,
            source_layout_fingerprint,
            logical_domains,
            field_default_generations,
            index_commit_floors,
            coverage,
            fingerprint: Vec::new(),
            object_id,
            field_ids,
        };
        artifact.fingerprint = coverage_artifact_fingerprint(&artifact);
        artifact.validate()?;
        Ok(artifact)
    }

    /// Validate artifact integrity and its logical-domain/base-generation proof.
    pub fn validate(&self) -> Result<()> {
        if self.namespace_uuid.is_nil()
            || self.index_uuid.is_nil()
            || self.object_id.is_nil()
            || self.build_dataset_version == 0
            || self.source_layout_fingerprint.len() != LOGICAL_COVERAGE_FINGERPRINT_SIZE
            || self.fingerprint.len() != LOGICAL_COVERAGE_FINGERPRINT_SIZE
        {
            return Err(Error::invalid_input(
                "logical index coverage artifact has invalid identity or version metadata",
            ));
        }
        if self.coverage.external.is_some() {
            return Err(Error::invalid_input(
                "logical index coverage artifact must contain inline authoritative selections",
            ));
        }
        if self.field_ids.is_empty()
            || self.field_ids.iter().any(|field_id| *field_id < 0)
            || self.field_ids.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(Error::invalid_input(
                "logical index coverage artifact field ids are not canonical",
            ));
        }
        self.coverage.validate_exact(Some(&self.field_ids), None)?;
        if self
            .logical_domains
            .windows(2)
            .any(|pair| pair[0].logical_fragment_id >= pair[1].logical_fragment_id)
            || self.logical_domains.iter().any(|domain| {
                domain.slot_count == 0
                    || domain.creation_version == 0
                    || domain.creation_version > self.build_dataset_version
            })
        {
            return Err(Error::invalid_input(
                "logical index coverage artifact domains are invalid or non-canonical",
            ));
        }

        let defaults = canonical_artifact_generations(
            &self.field_default_generations,
            &self.field_ids,
            self.build_dataset_version,
            "field defaults",
        )?;
        let floors = canonical_artifact_generations(
            &self.index_commit_floors,
            &self.field_ids,
            self.build_dataset_version,
            "index commit floors",
        )?;
        let domains = self
            .logical_domains
            .iter()
            .map(|domain| (domain.logical_fragment_id, domain))
            .collect::<BTreeMap<_, _>>();
        let covered_domains = self.coverage.logical_fragment_bitmap()?;
        if covered_domains != RoaringBitmap::from_iter(domains.keys().copied()) {
            return Err(Error::invalid_input(
                "logical index coverage artifact domain proof does not match coverage",
            ));
        }
        for shard in &self.coverage.shards {
            let selection = shard.selection.as_ref().ok_or_else(|| {
                Error::invalid_input(
                    "logical index coverage artifact is missing authoritative selection detail",
                )
            })?;
            for range in selection.to_ranges()? {
                let domain = domains.get(&range.logical_fragment_id).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "logical index coverage references unknown domain {}",
                        range.logical_fragment_id
                    ))
                })?;
                if range.end_slot > domain.slot_count {
                    return Err(Error::invalid_input(format!(
                        "logical index coverage range ends at slot {} beyond domain {} size {}",
                        range.end_slot, range.logical_fragment_id, domain.slot_count
                    )));
                }
            }
            let shard_domains = selection.logical_fragment_bitmap()?;
            for watermark in &shard.validated_through {
                if watermark.generation > self.build_dataset_version {
                    return Err(Error::invalid_input(format!(
                        "logical index coverage field {} watermark {} exceeds build version {}",
                        watermark.field_id, watermark.generation, self.build_dataset_version
                    )));
                }
                let field_base = defaults[&watermark.field_id].max(floors[&watermark.field_id]);
                for logical_fragment_id in shard_domains.iter() {
                    let domain = domains[&logical_fragment_id];
                    let base_generation = field_base.max(domain.creation_version);
                    if watermark.generation < base_generation {
                        return Err(Error::invalid_input(format!(
                            "logical index coverage field {} watermark {} predates base generation {} for domain {}",
                            watermark.field_id,
                            watermark.generation,
                            base_generation,
                            logical_fragment_id
                        )));
                    }
                }
            }
        }
        if self.fingerprint != coverage_artifact_fingerprint(self) {
            return Err(Error::invalid_input(
                "logical index coverage artifact fingerprint does not match its contents",
            ));
        }
        Ok(())
    }
}

fn canonical_artifact_generations(
    generations: &[FieldGeneration],
    field_ids: &[i32],
    build_dataset_version: u64,
    context: &str,
) -> Result<BTreeMap<i32, u64>> {
    if generations.len() != field_ids.len()
        || generations
            .iter()
            .zip(field_ids)
            .any(|(generation, field_id)| {
                generation.field_id != *field_id
                    || generation.generation == 0
                    || generation.generation > build_dataset_version
            })
    {
        return Err(Error::invalid_input(format!(
            "logical index coverage artifact {context} do not match indexed fields"
        )));
    }
    Ok(generations
        .iter()
        .map(|generation| (generation.field_id, generation.generation))
        .collect())
}

impl TryFrom<pb::LogicalIndexCoverageArtifact> for LogicalIndexCoverageArtifact {
    type Error = Error;

    fn try_from(value: pb::LogicalIndexCoverageArtifact) -> Result<Self> {
        if value.encoding_version != LOGICAL_INDEX_COVERAGE_ARTIFACT_ENCODING_VERSION {
            return Err(Error::invalid_input(format!(
                "unsupported logical index coverage artifact encoding version {}",
                value.encoding_version
            )));
        }
        let artifact = Self {
            namespace_uuid: value
                .namespace_uuid
                .as_ref()
                .ok_or_else(|| {
                    Error::invalid_input("LogicalIndexCoverageArtifact.namespace_uuid is missing")
                })
                .and_then(Uuid::try_from)?,
            index_uuid: value
                .index_uuid
                .as_ref()
                .ok_or_else(|| {
                    Error::invalid_input("LogicalIndexCoverageArtifact.index_uuid is missing")
                })
                .and_then(Uuid::try_from)?,
            build_dataset_version: value.build_dataset_version,
            source_layout_fingerprint: value.source_layout_fingerprint,
            logical_domains: value
                .logical_domains
                .into_iter()
                .map(RowAddressLogicalDomain::try_from)
                .collect::<Result<Vec<_>>>()?,
            field_default_generations: value
                .field_default_generations
                .into_iter()
                .map(FieldGeneration::try_from)
                .collect::<Result<Vec<_>>>()?,
            index_commit_floors: value
                .index_commit_floors
                .into_iter()
                .map(FieldGeneration::try_from)
                .collect::<Result<Vec<_>>>()?,
            coverage: value
                .coverage
                .ok_or_else(|| {
                    Error::invalid_input("LogicalIndexCoverageArtifact.coverage is missing")
                })?
                .try_into()?,
            fingerprint: value.fingerprint,
            object_id: value
                .object_id
                .as_ref()
                .ok_or_else(|| {
                    Error::invalid_input("LogicalIndexCoverageArtifact.object_id is missing")
                })
                .and_then(Uuid::try_from)?,
            field_ids: value.field_ids,
        };
        artifact.validate()?;
        Ok(artifact)
    }
}

impl From<&LogicalIndexCoverageArtifact> for pb::LogicalIndexCoverageArtifact {
    fn from(value: &LogicalIndexCoverageArtifact) -> Self {
        Self {
            encoding_version: LOGICAL_INDEX_COVERAGE_ARTIFACT_ENCODING_VERSION,
            namespace_uuid: Some((&value.namespace_uuid).into()),
            index_uuid: Some((&value.index_uuid).into()),
            build_dataset_version: value.build_dataset_version,
            source_layout_fingerprint: value.source_layout_fingerprint.clone(),
            logical_domains: value.logical_domains.iter().map(Into::into).collect(),
            field_default_generations: value
                .field_default_generations
                .iter()
                .map(Into::into)
                .collect(),
            index_commit_floors: value.index_commit_floors.iter().map(Into::into).collect(),
            coverage: Some((&value.coverage).into()),
            fingerprint: value.fingerprint.clone(),
            object_id: Some((&value.object_id).into()),
            field_ids: value.field_ids.clone(),
        }
    }
}

fn first_covered_row(selection: &LogicalRowAddressSelection) -> Result<u64> {
    Ok(selection
        .select(0)?
        .map(|address| address.raw())
        .unwrap_or(u64::MAX))
}

fn selections_overlap(
    left: &LogicalRowAddressSelection,
    right: &LogicalRowAddressSelection,
) -> Result<bool> {
    left.overlaps(right)
}

impl LogicalIndexCoverageShard {
    /// Create one exact coverage shard and bind its membership and generation
    /// watermark to a deterministic fingerprint.
    pub fn new_exact(
        selection: LogicalRowAddressSelection,
        mut field_ids: Vec<i32>,
        mut validated_through: Vec<FieldGeneration>,
    ) -> Result<Self> {
        selection.validate()?;
        field_ids.sort_unstable();
        field_ids.dedup();
        validated_through.sort_unstable();
        if field_ids.is_empty()
            || field_ids.iter().any(|field_id| *field_id < 0)
            || validated_through.len() != field_ids.len()
            || validated_through
                .iter()
                .zip(&field_ids)
                .any(|(generation, field_id)| {
                    generation.field_id != *field_id || generation.generation == 0
                })
        {
            return Err(Error::invalid_input(
                "logical index coverage fields and validated-through watermarks must match",
            ));
        }
        let logical_fragment_bitmap = encode_logical_fragment_bitmap(&selection)?;
        let mut shard = Self {
            row_count: selection.cardinality(),
            selection: Some(selection),
            field_ids,
            validated_through,
            fingerprint: Vec::new(),
            logical_fragment_bitmap,
        };
        shard.fingerprint = coverage_shard_fingerprint(&shard)?;
        Ok(shard)
    }

    pub fn logical_fragment_bitmap(&self) -> Result<RoaringBitmap> {
        decode_logical_fragment_bitmap(&self.logical_fragment_bitmap)
    }

    /// Conservative overlap usable by manifest-only maintenance. External
    /// detail falls back to logical-domain intersection and therefore never
    /// treats unknown membership as disjoint.
    pub fn may_overlap(&self, selection: &LogicalRowAddressSelection) -> Result<bool> {
        if let Some(detail) = self.selection.as_ref() {
            detail.overlaps(selection)
        } else {
            Ok(!self
                .logical_fragment_bitmap()?
                .is_disjoint(&selection.logical_fragment_bitmap()?))
        }
    }
}

impl LogicalIndexCoverage {
    /// Create canonical inline exact coverage.
    pub fn new_exact(mut shards: Vec<LogicalIndexCoverageShard>) -> Result<Self> {
        shards.retain(|shard| shard.row_count != 0);
        for shard in &mut shards {
            let selection = shard.selection.as_ref().ok_or_else(|| {
                Error::invalid_input("new exact logical coverage requires inline selections")
            })?;
            shard.logical_fragment_bitmap = encode_logical_fragment_bitmap(selection)?;
            shard.fingerprint = coverage_shard_fingerprint(shard)?;
        }
        shards.sort_by_key(|shard| {
            shard
                .selection
                .as_ref()
                .and_then(|selection| first_covered_row(selection).ok())
                .unwrap_or(u64::MAX)
        });
        let mut coverage = Self {
            shards,
            external: None,
            fingerprint: Vec::new(),
            clone_provenance: None,
        };
        coverage.fingerprint = coverage_root_fingerprint(&coverage.shards);
        coverage.validate_exact(None, None)?;
        Ok(coverage)
    }

    /// Validate inline detail or an external summary without reading index data.
    pub fn validate_exact(
        &self,
        index_fields: Option<&[i32]>,
        schema_field_ids: Option<&BTreeSet<i32>>,
    ) -> Result<()> {
        if self.fingerprint.len() != LOGICAL_COVERAGE_FINGERPRINT_SIZE {
            return Err(Error::invalid_input(
                "logical index coverage requires a 16-byte fingerprint",
            ));
        }
        let expected_fields = index_fields.map(|fields| {
            let mut fields = fields.to_vec();
            fields.sort_unstable();
            fields.dedup();
            fields
        });
        let mut previous_first = None;
        let mut has_external_detail = false;
        for (shard_index, shard) in self.shards.iter().enumerate() {
            let summary_fragments = decode_logical_fragment_bitmap(&shard.logical_fragment_bitmap)?;
            if let Some(selection) = shard.selection.as_ref() {
                selection.validate()?;
                let first = first_covered_row(selection)?;
                if previous_first.is_some_and(|previous| previous >= first) {
                    return Err(Error::invalid_input(
                        "logical index coverage shards are not in canonical row order",
                    ));
                }
                previous_first = Some(first);
                if shard.row_count != selection.cardinality()
                    || summary_fragments != selection.logical_fragment_bitmap()?
                    || shard.fingerprint != coverage_shard_fingerprint(shard)?
                {
                    return Err(Error::invalid_input(format!(
                        "logical index coverage shard {shard_index} detail does not match its summary"
                    )));
                }
            } else {
                has_external_detail = true;
            }
            if shard.row_count == 0
                || summary_fragments.is_empty()
                || shard.fingerprint.len() != LOGICAL_COVERAGE_FINGERPRINT_SIZE
            {
                return Err(Error::invalid_input(format!(
                    "logical index coverage shard {shard_index} has an invalid count or fingerprint"
                )));
            }
            if shard.field_ids.is_empty()
                || shard.field_ids.windows(2).any(|pair| pair[0] >= pair[1])
                || shard.validated_through.len() != shard.field_ids.len()
                || shard.validated_through.iter().zip(&shard.field_ids).any(
                    |(generation, field_id)| {
                        generation.field_id != *field_id || generation.generation == 0
                    },
                )
            {
                return Err(Error::invalid_input(format!(
                    "logical index coverage shard {shard_index} has invalid field watermarks"
                )));
            }
            if expected_fields
                .as_ref()
                .is_some_and(|fields| fields != &shard.field_ids)
                || schema_field_ids
                    .is_some_and(|schema| shard.field_ids.iter().any(|id| !schema.contains(id)))
            {
                return Err(Error::invalid_input(format!(
                    "logical index coverage shard {shard_index} does not match index or schema fields"
                )));
            }
            if let Some(selection) = shard.selection.as_ref() {
                for previous in &self.shards[..shard_index] {
                    if let Some(previous) = previous.selection.as_ref()
                        && selections_overlap(previous, selection)?
                    {
                        return Err(Error::invalid_input(
                            "logical index coverage shards overlap",
                        ));
                    }
                }
            }
        }
        if has_external_detail && self.external.is_none() {
            return Err(Error::invalid_input(
                "logical index coverage summary is missing its external detail reference",
            ));
        }
        if self.external.is_none() && self.fingerprint != coverage_root_fingerprint(&self.shards) {
            return Err(Error::invalid_input(
                "logical index coverage root fingerprint does not match its shards",
            ));
        }
        if self.external.as_ref().is_some_and(|external| {
            external.path.is_empty()
                || external.byte_length == 0
                || external.global_buffer_index == 0
                || external.object_size == 0
                || external.object_id.is_nil()
                || external.artifact_namespace_uuid.is_nil()
                || external.artifact_layout_fingerprint.len() != LOGICAL_COVERAGE_FINGERPRINT_SIZE
                || external
                    .offset
                    .checked_add(external.byte_length)
                    .is_none_or(|end| end > external.object_size)
        }) {
            return Err(Error::invalid_input(
                "logical index coverage has an invalid external artifact reference",
            ));
        }
        if self.clone_provenance.as_ref().is_some_and(|provenance| {
            provenance.source_namespace_uuid.is_nil()
                || provenance.target_namespace_uuid.is_nil()
                || provenance.source_namespace_uuid == provenance.target_namespace_uuid
                || provenance.source_coverage_fingerprint.len() != LOGICAL_COVERAGE_FINGERPRINT_SIZE
                || provenance.transaction_uuid.is_nil()
                || provenance.depth == 0
                || provenance.source_manifest_version == 0
        }) {
            return Err(Error::invalid_input(
                "logical index coverage has invalid clone provenance",
            ));
        }
        if self.clone_provenance.is_some() && self.external.is_none() {
            return Err(Error::invalid_input(
                "unbound logical index coverage cannot carry clone provenance",
            ));
        }
        Ok(())
    }

    /// Bind coverage integrity to its owning index segment and immutable anchor.
    /// The complete declared file set participates in the root fingerprint so a
    /// path, size, anchor range, or object substitution is detected at open.
    pub fn bind_to_index(
        &mut self,
        namespace_uuid: Uuid,
        index_uuid: Uuid,
        external: LogicalIndexCoverageFile,
        files: &[IndexFile],
    ) -> Result<()> {
        if self.clone_provenance.is_some() || external.artifact_namespace_uuid != namespace_uuid {
            return Err(Error::invalid_input(
                "new logical index coverage must be authored by its current namespace",
            ));
        }
        let fingerprint = bound_coverage_root_fingerprint(
            &self.shards,
            namespace_uuid,
            index_uuid,
            &external,
            self.clone_provenance.as_ref(),
            files,
        )?;
        self.external = Some(external);
        self.fingerprint = fingerprint;
        self.validate_index_binding(namespace_uuid, index_uuid, files)
    }

    /// Rebind an immutable index artifact during a clone commit. The source
    /// binding is validated before a fixed-size provenance link is derived;
    /// callers cannot use this path for staged-index admission.
    #[allow(clippy::too_many_arguments)]
    pub fn rebind_for_clone(
        &mut self,
        source_namespace_uuid: Uuid,
        target_namespace_uuid: Uuid,
        index_uuid: Uuid,
        files: &[IndexFile],
        transaction_uuid: Uuid,
        is_shallow: bool,
        source_manifest_version: u64,
    ) -> Result<()> {
        self.validate_index_binding(source_namespace_uuid, index_uuid, files)?;
        if target_namespace_uuid.is_nil()
            || target_namespace_uuid == source_namespace_uuid
            || transaction_uuid.is_nil()
            || source_manifest_version == 0
        {
            return Err(Error::invalid_input(
                "logical index coverage clone rebind has invalid commit provenance",
            ));
        }
        let source_coverage_fingerprint = self.fingerprint.clone();
        let depth = self
            .clone_provenance
            .as_ref()
            .map(|provenance| provenance.depth)
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| Error::invalid_input("logical index coverage clone depth overflows"))?;
        self.clone_provenance = Some(LogicalIndexCoverageCloneProvenance {
            source_namespace_uuid,
            target_namespace_uuid,
            source_coverage_fingerprint,
            transaction_uuid,
            depth,
            is_shallow,
            source_manifest_version,
        });
        let external = self.external.as_ref().ok_or_else(|| {
            Error::invalid_input("logical index coverage clone lost its anchor reference")
        })?;
        self.fingerprint = bound_coverage_root_fingerprint(
            &self.shards,
            target_namespace_uuid,
            index_uuid,
            external,
            self.clone_provenance.as_ref(),
            files,
        )?;
        self.validate_index_binding(target_namespace_uuid, index_uuid, files)
    }

    /// Validate the manifest-only owner binding without opening index objects.
    pub fn validate_index_binding(
        &self,
        namespace_uuid: Uuid,
        index_uuid: Uuid,
        files: &[IndexFile],
    ) -> Result<()> {
        let external = self.external.as_ref().ok_or_else(|| {
            Error::invalid_input(
                "logical index coverage is missing its core-authored anchor reference",
            )
        })?;
        let declared_anchor = canonical_index_files(files)?
            .into_iter()
            .find(|file| file.path == external.path)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "logical index coverage anchor {} is not declared in IndexMetadata.files",
                    external.path
                ))
            })?;
        if declared_anchor.size_bytes != external.object_size {
            return Err(Error::invalid_input(format!(
                "logical index coverage anchor {} size {} does not match declared size {}",
                external.path, external.object_size, declared_anchor.size_bytes
            )));
        }
        match self.clone_provenance.as_ref() {
            Some(provenance) => {
                if provenance.target_namespace_uuid != namespace_uuid
                    || external.artifact_namespace_uuid == namespace_uuid
                    || (provenance.depth == 1
                        && external.artifact_namespace_uuid != provenance.source_namespace_uuid)
                {
                    return Err(Error::invalid_input(
                        "logical index coverage clone provenance does not match its target or immutable artifact",
                    ));
                }
            }
            None if external.artifact_namespace_uuid != namespace_uuid => {
                return Err(Error::invalid_input(
                    "logical index coverage artifact namespace differs without clone provenance",
                ));
            }
            None => {}
        }
        let expected = bound_coverage_root_fingerprint(
            &self.shards,
            namespace_uuid,
            index_uuid,
            external,
            self.clone_provenance.as_ref(),
            files,
        )?;
        if self.fingerprint != expected {
            return Err(Error::invalid_input(
                "logical index coverage root fingerprint does not match its index owner or files",
            ));
        }
        Ok(())
    }

    /// Drop large exact selections from the manifest while retaining their
    /// immutable per-shard count, routing bitmap, watermarks, and fingerprint.
    /// Returns whether any detail was externalized.
    pub fn externalize_detail_over(&mut self, inline_byte_limit: usize) -> Result<bool> {
        if self.external.is_none() {
            return Err(Error::invalid_input(
                "logical index coverage cannot externalize detail without an anchor",
            ));
        }
        let detail_bytes = self.shards.iter().try_fold(0usize, |total, shard| {
            let selection = shard.selection.as_ref().ok_or_else(|| {
                Error::invalid_input(
                    "logical index coverage cannot re-externalize unresolved detail",
                )
            })?;
            total
                .checked_add(selection.canonical_proto().encoded_len())
                .ok_or_else(|| Error::invalid_input("logical index coverage size overflows"))
        })?;
        if detail_bytes <= inline_byte_limit {
            return Ok(false);
        }
        for shard in &mut self.shards {
            shard.selection = None;
        }
        self.validate_exact(None, None)?;
        Ok(true)
    }

    /// Rehydrate a committed summary from its authoritative exact artifact.
    /// All manifest-resident summaries must agree before detail is accepted.
    pub fn resolve_from_authoritative(&self, authoritative: &Self) -> Result<Self> {
        authoritative.validate_exact(None, None)?;
        if authoritative.external.is_some() || self.shards.len() != authoritative.shards.len() {
            return Err(Error::invalid_input(
                "logical index coverage artifact does not match manifest shard count",
            ));
        }
        let mut resolved = self.clone();
        for (shard_index, (summary, exact)) in
            self.shards.iter().zip(&authoritative.shards).enumerate()
        {
            if summary.field_ids != exact.field_ids
                || summary.validated_through != exact.validated_through
                || summary.fingerprint != exact.fingerprint
                || summary.row_count != exact.row_count
                || summary.logical_fragment_bitmap != exact.logical_fragment_bitmap
                || summary
                    .selection
                    .as_ref()
                    .is_some_and(|selection| Some(selection) != exact.selection.as_ref())
            {
                return Err(Error::invalid_input(format!(
                    "logical index coverage shard {shard_index} summary differs from its authoritative artifact"
                )));
            }
            resolved.shards[shard_index].selection = exact.selection.clone();
        }
        resolved.validate_exact(None, None)?;
        Ok(resolved)
    }

    /// Logical fragment IDs touched by this exact coverage. This is an
    /// ephemeral query-planning projection, never persisted as a physical
    /// fragment bitmap.
    pub fn logical_fragment_bitmap(&self) -> Result<RoaringBitmap> {
        let mut fragments = RoaringBitmap::new();
        for shard in &self.shards {
            fragments |= shard.logical_fragment_bitmap()?;
        }
        Ok(fragments)
    }

    pub fn has_external_detail(&self) -> bool {
        self.shards.iter().any(|shard| shard.selection.is_none())
    }

    pub fn requires_authoritative_resolution(&self) -> bool {
        self.has_external_detail() || self.clone_provenance.is_some()
    }
}

impl DeepSizeOf for IndexMetadata {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.uuid.as_bytes().deep_size_of_children(context)
            + self.fields.deep_size_of_children(context)
            + self.name.deep_size_of_children(context)
            + self.dataset_version.deep_size_of_children(context)
            + self
                .fragment_bitmap
                .as_ref()
                .map(|fragment_bitmap| fragment_bitmap.serialized_size())
                .unwrap_or(0)
            + self.files.deep_size_of_children(context)
            + self
                .logical_coverage
                .as_ref()
                .map(|coverage| coverage.deep_size_of_children(context))
                .unwrap_or(0)
    }
}

impl TryFrom<pb::IndexMetadata> for IndexMetadata {
    type Error = Error;

    fn try_from(proto: pb::IndexMetadata) -> Result<Self> {
        let fragment_bitmap = if proto.fragment_bitmap.is_empty() {
            None
        } else {
            Some(RoaringBitmap::deserialize_from(
                &mut proto.fragment_bitmap.as_slice(),
            )?)
        };

        let files = if proto.files.is_empty() {
            None
        } else {
            Some(
                proto
                    .files
                    .into_iter()
                    .map(|f| IndexFile {
                        path: f.path,
                        size_bytes: f.size_bytes,
                    })
                    .collect(),
            )
        };

        Ok(Self {
            uuid: proto.uuid.as_ref().map(Uuid::try_from).ok_or_else(|| {
                Error::invalid_input("uuid field does not exist in Index metadata".to_string())
            })??,
            name: proto.name,
            fields: proto.fields,
            dataset_version: proto.dataset_version,
            fragment_bitmap,
            index_details: proto.index_details.map(Arc::new),
            index_version: proto.index_version.unwrap_or_default(),
            created_at: proto.created_at.map(|ts| {
                DateTime::from_timestamp_millis(ts as i64)
                    .expect("Invalid timestamp in index metadata")
            }),
            base_id: proto.base_id,
            files,
            row_reference_domain: proto
                .row_reference_domain
                .map(RowReferenceDomain::try_from)
                .transpose()?,
            logical_coverage: proto
                .logical_coverage
                .map(LogicalIndexCoverage::try_from)
                .transpose()?,
        })
    }
}

impl From<&IndexMetadata> for pb::IndexMetadata {
    fn from(idx: &IndexMetadata) -> Self {
        let mut fragment_bitmap = Vec::new();
        if let Some(bitmap) = &idx.fragment_bitmap
            && let Err(e) = bitmap.serialize_into(&mut fragment_bitmap)
        {
            // In theory, this should never error. But if we do, just
            // recover gracefully.
            log::error!("Failed to serialize fragment bitmap: {}", e);
            fragment_bitmap.clear();
        }

        let files = idx
            .files
            .as_ref()
            .map(|files| {
                files
                    .iter()
                    .map(|f| pb::IndexFile {
                        path: f.path.clone(),
                        size_bytes: f.size_bytes,
                    })
                    .collect()
            })
            .unwrap_or_default();

        Self {
            uuid: Some((&idx.uuid).into()),
            name: idx.name.clone(),
            fields: idx.fields.clone(),
            dataset_version: idx.dataset_version,
            fragment_bitmap,
            index_details: idx
                .index_details
                .as_ref()
                .map(|details| details.as_ref().clone()),
            index_version: Some(idx.index_version),
            created_at: idx.created_at.map(|dt| dt.timestamp_millis() as u64),
            base_id: idx.base_id,
            files,
            row_reference_domain: idx.row_reference_domain.map(Into::into),
            logical_coverage: idx.logical_coverage.as_ref().map(Into::into),
        }
    }
}

/// Returns a [`CacheCodec`](lance_core::cache::CacheCodec) for `Vec<IndexMetadata>`.
///
/// Uses `pb::IndexSection` (which wraps `repeated IndexMetadata`) as the wire
/// format, reusing the existing `TryFrom`/`From` conversions.
///
/// Uses [`CacheCodec::new`](lance_core::cache::CacheCodec::new) because the
/// orphan rule prevents `impl CacheCodecImpl for Vec<IndexMetadata>`.
type ArcAny = Arc<dyn std::any::Any + Send + Sync>;

/// Stable type identifier for the `Vec<IndexMetadata>` cache entry.
const INDEX_METADATA_TYPE_ID: &str = "lance.table.IndexMetadataList";
/// Body schema version written by this build.
const INDEX_METADATA_VERSION: u32 = 1;

fn serialize_index_metadata(
    any: &ArcAny,
    writer: &mut CacheEntryWriter<'_>,
) -> lance_core::Result<()> {
    let vec = any
        .downcast_ref::<Vec<IndexMetadata>>()
        .expect("index_metadata_codec: wrong type (this is a bug in the cache layer)");
    let section = pb::IndexSection {
        indices: vec.iter().map(pb::IndexMetadata::from).collect(),
    };
    writer.write_header(&section)
}

fn deserialize_index_metadata(reader: &mut CacheEntryReader<'_>) -> lance_core::Result<ArcAny> {
    let section: pb::IndexSection = reader.read_header()?;
    let indices: Vec<IndexMetadata> = section
        .indices
        .into_iter()
        .map(IndexMetadata::try_from)
        .collect::<lance_core::Result<_>>()?;
    Ok(Arc::new(indices))
}

pub fn index_metadata_codec() -> lance_core::cache::CacheCodec {
    lance_core::cache::CacheCodec::new(
        INDEX_METADATA_TYPE_ID,
        INDEX_METADATA_VERSION,
        serialize_index_metadata,
        deserialize_index_metadata,
    )
}

/// List all files in an index directory with their sizes.
///
/// Returns a list of `IndexFile` structs containing relative paths and sizes.
/// This is used to capture file metadata after index creation/modification.
pub async fn list_index_files_with_sizes(
    object_store: &ObjectStore,
    index_dir: &Path,
) -> Result<Vec<IndexFile>> {
    let mut files = Vec::new();
    let mut stream = object_store.read_dir_all(index_dir, None);
    while let Some(meta) = stream.next().await {
        let meta = meta?;
        // Get relative path by stripping the index_dir prefix
        let relative_path = meta
            .location
            .as_ref()
            .strip_prefix(index_dir.as_ref())
            .map(|s| s.trim_start_matches('/').to_string())
            .unwrap_or_else(|| meta.location.filename().unwrap_or("").to_string());
        files.push(IndexFile {
            path: relative_path,
            size_bytes: meta.size,
        });
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::LogicalRowAddressRange;
    use std::collections::HashMap;

    /// Demonstrates the pattern a disk-backed cache backend would use:
    /// serialize entries to bytes, store in a key-value map, then
    /// deserialize on retrieval.
    #[test]
    fn test_index_metadata_codec_roundtrip() {
        let codec = index_metadata_codec();

        let original = vec![
            IndexMetadata {
                uuid: Uuid::new_v4(),
                name: "my_index".to_string(),
                fields: vec![0, 1],
                dataset_version: 42,
                fragment_bitmap: Some(RoaringBitmap::from_iter([1, 2, 3])),
                index_details: None,
                index_version: 1,
                created_at: None,
                base_id: None,
                files: Some(vec![IndexFile {
                    path: "index.idx".to_string(),
                    size_bytes: 1024,
                }]),
                row_reference_domain: None,
                logical_coverage: None,
            },
            IndexMetadata {
                uuid: Uuid::new_v4(),
                name: "second_index".to_string(),
                fields: vec![2],
                dataset_version: 43,
                fragment_bitmap: None,
                index_details: None,
                index_version: 2,
                created_at: None,
                base_id: Some(7),
                files: None,
                row_reference_domain: None,
                logical_coverage: None,
            },
        ];

        // Simulate a disk-backed store: HashMap<String, Vec<u8>>
        let mut store: HashMap<String, Vec<u8>> = HashMap::new();

        // Serialize into the store
        let key = "dataset/v42/Vec<IndexMetadata>".to_string();
        let mut buf = Vec::new();
        let entry: Arc<dyn std::any::Any + Send + Sync> = Arc::new(original.clone());
        codec.serialize(&entry, &mut buf).unwrap();
        store.insert(key.clone(), buf);

        // Deserialize from the store
        let bytes = store.get(&key).unwrap();
        let recovered = codec
            .deserialize(&bytes::Bytes::copy_from_slice(bytes))
            .hit()
            .expect("entry should decode as a hit");
        let recovered = recovered
            .downcast::<Vec<IndexMetadata>>()
            .expect("downcast should succeed");

        assert_eq!(original.len(), recovered.len());
        for (orig, rec) in original.iter().zip(recovered.iter()) {
            assert_eq!(orig.uuid, rec.uuid);
            assert_eq!(orig.name, rec.name);
            assert_eq!(orig.fields, rec.fields);
            assert_eq!(orig.dataset_version, rec.dataset_version);
            assert_eq!(orig.fragment_bitmap, rec.fragment_bitmap);
            assert_eq!(orig.index_version, rec.index_version);
            assert_eq!(orig.base_id, rec.base_id);
            assert_eq!(orig.files, rec.files);
        }
    }

    fn coverage_shard(
        logical_fragment_id: u32,
        start_slot: u32,
        end_slot: u32,
    ) -> LogicalIndexCoverageShard {
        LogicalIndexCoverageShard::new_exact(
            LogicalRowAddressSelection::from_ranges(vec![LogicalRowAddressRange::new(
                logical_fragment_id,
                start_slot,
                end_slot,
            )])
            .unwrap(),
            vec![7],
            vec![FieldGeneration {
                field_id: 7,
                generation: 11,
            }],
        )
        .unwrap()
    }

    #[test]
    fn test_exact_logical_coverage_validates_fingerprints_and_projection() {
        let coverage =
            LogicalIndexCoverage::new_exact(vec![coverage_shard(9, 3, 6), coverage_shard(2, 0, 2)])
                .unwrap();
        coverage
            .validate_exact(Some(&[7]), Some(&BTreeSet::from([7])))
            .unwrap();
        assert_eq!(
            coverage.logical_fragment_bitmap().unwrap(),
            RoaringBitmap::from_iter([2, 9])
        );

        let mut tampered = coverage;
        tampered.shards[0].row_count += 1;
        assert!(tampered.validate_exact(Some(&[7]), None).is_err());

        let empty = LogicalIndexCoverage::new_exact(Vec::new()).unwrap();
        assert!(empty.shards.is_empty());
        empty.validate_exact(Some(&[7]), None).unwrap();
    }

    #[test]
    fn test_exact_logical_coverage_rejects_overlapping_shards() {
        let error =
            LogicalIndexCoverage::new_exact(vec![coverage_shard(3, 0, 4), coverage_shard(3, 2, 8)])
                .unwrap_err();
        assert!(error.to_string().contains("overlap"));
    }

    #[test]
    fn test_logical_coverage_binding_detects_anchor_and_file_tampering() {
        let namespace_uuid = Uuid::new_v4();
        let index_uuid = Uuid::new_v4();
        let files = vec![
            IndexFile {
                path: "pages.lance".to_string(),
                size_bytes: 2048,
            },
            IndexFile {
                path: "anchor.lance".to_string(),
                size_bytes: 1024,
            },
        ];
        let external = LogicalIndexCoverageFile {
            path: "anchor.lance".to_string(),
            offset: 128,
            byte_length: 256,
            global_buffer_index: 2,
            object_size: 1024,
            object_id: Uuid::new_v4(),
            artifact_namespace_uuid: namespace_uuid,
            artifact_layout_fingerprint: vec![1; LOGICAL_COVERAGE_FINGERPRINT_SIZE],
        };
        let mut coverage = LogicalIndexCoverage::new_exact(vec![coverage_shard(3, 0, 4)]).unwrap();
        coverage
            .bind_to_index(namespace_uuid, index_uuid, external, &files)
            .unwrap();
        coverage
            .validate_index_binding(namespace_uuid, index_uuid, &files)
            .unwrap();

        let mut path_tampered = coverage.clone();
        path_tampered.external.as_mut().unwrap().path = "pages.lance".to_string();
        assert!(
            path_tampered
                .validate_index_binding(namespace_uuid, index_uuid, &files)
                .is_err()
        );

        let mut range_tampered = coverage.clone();
        range_tampered.external.as_mut().unwrap().offset += 1;
        assert!(
            range_tampered
                .validate_index_binding(namespace_uuid, index_uuid, &files)
                .is_err()
        );

        let mut files_tampered = files.clone();
        files_tampered[0].size_bytes += 1;
        assert!(
            coverage
                .validate_index_binding(namespace_uuid, index_uuid, &files_tampered)
                .is_err()
        );
    }

    #[test]
    fn test_external_summary_resolves_only_matching_authoritative_detail() {
        let namespace_uuid = Uuid::new_v4();
        let index_uuid = Uuid::new_v4();
        let files = vec![IndexFile {
            path: "anchor.lance".to_string(),
            size_bytes: 1024,
        }];
        let exact = LogicalIndexCoverage::new_exact(vec![coverage_shard(3, 0, 4)]).unwrap();
        let mut summary = exact.clone();
        summary
            .bind_to_index(
                namespace_uuid,
                index_uuid,
                LogicalIndexCoverageFile {
                    path: "anchor.lance".to_string(),
                    offset: 128,
                    byte_length: 256,
                    global_buffer_index: 2,
                    object_size: 1024,
                    object_id: Uuid::new_v4(),
                    artifact_namespace_uuid: namespace_uuid,
                    artifact_layout_fingerprint: vec![1; LOGICAL_COVERAGE_FINGERPRINT_SIZE],
                },
                &files,
            )
            .unwrap();
        assert!(summary.externalize_detail_over(0).unwrap());
        assert!(summary.has_external_detail());

        let mut count_tampered = summary.clone();
        count_tampered.shards[0].row_count += 1;
        assert!(
            count_tampered
                .validate_index_binding(namespace_uuid, index_uuid, &files)
                .is_err()
        );

        let resolved = summary.resolve_from_authoritative(&exact).unwrap();
        assert_eq!(resolved.shards[0].selection, exact.shards[0].selection);
        resolved
            .validate_index_binding(namespace_uuid, index_uuid, &files)
            .unwrap();

        let mut mismatched = exact;
        mismatched.shards[0].row_count += 1;
        assert!(summary.resolve_from_authoritative(&mismatched).is_err());
    }

    #[test]
    fn test_clone_rebind_preserves_artifact_owner_and_chains_source_binding() {
        let source_namespace = Uuid::new_v4();
        let first_target_namespace = Uuid::new_v4();
        let second_target_namespace = Uuid::new_v4();
        let index_uuid = Uuid::new_v4();
        let files = vec![IndexFile {
            path: "anchor.lance".to_string(),
            size_bytes: 1024,
        }];
        let mut coverage = LogicalIndexCoverage::new_exact(vec![coverage_shard(3, 0, 4)]).unwrap();
        coverage
            .bind_to_index(
                source_namespace,
                index_uuid,
                LogicalIndexCoverageFile {
                    path: "anchor.lance".to_string(),
                    offset: 128,
                    byte_length: 256,
                    global_buffer_index: 2,
                    object_size: 1024,
                    object_id: Uuid::new_v4(),
                    artifact_namespace_uuid: source_namespace,
                    artifact_layout_fingerprint: vec![1; LOGICAL_COVERAGE_FINGERPRINT_SIZE],
                },
                &files,
            )
            .unwrap();
        let source_fingerprint = coverage.fingerprint.clone();
        coverage
            .rebind_for_clone(
                source_namespace,
                first_target_namespace,
                index_uuid,
                &files,
                Uuid::new_v4(),
                true,
                11,
            )
            .unwrap();
        assert!(
            coverage
                .validate_index_binding(source_namespace, index_uuid, &files)
                .is_err()
        );
        coverage
            .validate_index_binding(first_target_namespace, index_uuid, &files)
            .unwrap();
        let first_provenance = coverage.clone_provenance.as_ref().unwrap();
        assert_eq!(first_provenance.depth, 1);
        assert_eq!(
            first_provenance.source_coverage_fingerprint,
            source_fingerprint
        );
        assert_eq!(
            coverage.external.as_ref().unwrap().artifact_namespace_uuid,
            source_namespace
        );

        let first_target_fingerprint = coverage.fingerprint.clone();
        coverage
            .rebind_for_clone(
                first_target_namespace,
                second_target_namespace,
                index_uuid,
                &files,
                Uuid::new_v4(),
                false,
                12,
            )
            .unwrap();
        coverage
            .validate_index_binding(second_target_namespace, index_uuid, &files)
            .unwrap();
        let second_provenance = coverage.clone_provenance.as_ref().unwrap();
        assert_eq!(second_provenance.depth, 2);
        assert_eq!(
            second_provenance.source_coverage_fingerprint,
            first_target_fingerprint
        );
        assert!(!second_provenance.is_shallow);
    }
}
