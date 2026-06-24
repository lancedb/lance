// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::num::NonZero;
use std::sync::Arc;

use lance_core::Error;
use lance_core::deepsize::DeepSizeOf;
use lance_file::format::{MAJOR_VERSION, MINOR_VERSION};
use lance_file::version::LanceFileVersion;
use lance_io::utils::CachedFileSize;
use object_store::path::Path;
use roaring::RoaringBitmap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::format::pb;

use crate::rowids::version::{
    RowDatasetVersionMeta, created_at_version_meta_to_pb, last_updated_at_version_meta_to_pb,
};
use lance_core::datatypes::Schema;
use lance_core::error::Result;

/// Lance Data File
///
/// A data file is one piece of file storing data.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct DataFile {
    /// Relative path of the data file to dataset root.
    pub path: String,
    /// The ids of fields in this file.
    ///
    /// When identical across many fragments (common case), multiple `DataFile`
    /// instances share a single heap allocation via `Arc`, significantly
    /// reducing manifest memory for large tables.
    pub fields: Arc<[i32]>,
    /// The offsets of the fields listed in `fields`, empty in v1 files
    ///
    /// Note that -1 is a possibility and it indices that the field has
    /// no top-level column in the file.
    ///
    /// Columns that lack a field id may still exist as extra entries in
    /// `column_indices`; such columns are ignored by field-id–based projection.
    /// For example, some fields, such as blob fields, occupy multiple
    /// columns in the file but only have a single field id.
    pub column_indices: Arc<[i32]>,
    /// The major version of the file format used to write this file.
    pub file_major_version: u32,
    /// The minor version of the file format used to write this file.
    pub file_minor_version: u32,

    /// The size of the file in bytes, if known.
    pub file_size_bytes: CachedFileSize,

    /// The base path of the datafile, when the datafile is outside the dataset.
    pub base_id: Option<u32>,
}

// Custom Serialize: convert Arc<[i32]> to slice for transparent JSON output
impl Serialize for DataFile {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("DataFile", 7)?;
        s.serialize_field("path", &self.path)?;
        s.serialize_field("fields", self.fields.as_ref())?;
        s.serialize_field("column_indices", self.column_indices.as_ref())?;
        s.serialize_field("file_major_version", &self.file_major_version)?;
        s.serialize_field("file_minor_version", &self.file_minor_version)?;
        s.serialize_field("file_size_bytes", &self.file_size_bytes)?;
        s.serialize_field("base_id", &self.base_id)?;
        s.end()
    }
}

// Custom Deserialize: read Vec<i32> and convert to Arc<[i32]>
impl<'de> Deserialize<'de> for DataFile {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct DataFileHelper {
            path: String,
            fields: Vec<i32>,
            #[serde(default)]
            column_indices: Vec<i32>,
            #[serde(default)]
            file_major_version: u32,
            #[serde(default)]
            file_minor_version: u32,
            file_size_bytes: CachedFileSize,
            base_id: Option<u32>,
        }

        let helper = DataFileHelper::deserialize(deserializer)?;
        Ok(Self {
            path: helper.path,
            fields: Arc::from(helper.fields),
            column_indices: Arc::from(helper.column_indices),
            file_major_version: helper.file_major_version,
            file_minor_version: helper.file_minor_version,
            file_size_bytes: helper.file_size_bytes,
            base_id: helper.base_id,
        })
    }
}

impl DataFile {
    pub fn new(
        path: impl Into<String>,
        fields: Vec<i32>,
        column_indices: Vec<i32>,
        file_major_version: u32,
        file_minor_version: u32,
        file_size_bytes: Option<NonZero<u64>>,
        base_id: Option<u32>,
    ) -> Self {
        Self {
            path: path.into(),
            fields: Arc::from(fields),
            column_indices: Arc::from(column_indices),
            file_major_version,
            file_minor_version,
            file_size_bytes: file_size_bytes.into(),
            base_id,
        }
    }

    /// Create a new `DataFile` with the expectation that fields and column_indices will be set later
    pub fn new_unstarted(
        path: impl Into<String>,
        file_major_version: u32,
        file_minor_version: u32,
    ) -> Self {
        Self {
            path: path.into(),
            fields: Arc::from([]),
            column_indices: Arc::from([]),
            file_major_version,
            file_minor_version,
            file_size_bytes: Default::default(),
            base_id: None,
        }
    }

    pub fn new_legacy_from_fields(
        path: impl Into<String>,
        fields: Vec<i32>,
        base_id: Option<u32>,
    ) -> Self {
        Self::new(
            path,
            fields,
            vec![],
            MAJOR_VERSION as u32,
            MINOR_VERSION as u32,
            None,
            base_id,
        )
    }

    pub fn new_legacy(
        path: impl Into<String>,
        schema: &Schema,
        file_size_bytes: Option<NonZero<u64>>,
        base_id: Option<u32>,
    ) -> Self {
        let mut field_ids = schema.field_ids();
        field_ids.sort();
        Self::new(
            path,
            field_ids,
            vec![],
            MAJOR_VERSION as u32,
            MINOR_VERSION as u32,
            file_size_bytes,
            base_id,
        )
    }

    pub fn schema(&self, full_schema: &Schema) -> Schema {
        full_schema.project_by_ids(&self.fields, false)
    }

    pub fn is_legacy_file(&self) -> bool {
        self.file_major_version == 0 && self.file_minor_version < 3
    }

    pub fn validate(&self, base_path: &Path) -> Result<()> {
        if self.is_legacy_file() {
            if !self.fields.windows(2).all(|w| w[0] < w[1]) {
                return Err(Error::corrupt_file(
                    base_path.clone().join(self.path.clone()),
                    "contained unsorted or duplicate field ids",
                ));
            }
        } else if self.column_indices.len() < self.fields.len() {
            // Every recorded field id must have a column index, but not every column needs
            // to be associated with a field id (extra columns are allowed).
            return Err(Error::corrupt_file(
                base_path.clone().join(self.path.clone()),
                "contained fewer column_indices than fields",
            ));
        }
        Ok(())
    }
}

impl From<&DataFile> for pb::DataFile {
    fn from(df: &DataFile) -> Self {
        Self {
            path: df.path.clone(),
            fields: df.fields.to_vec(),
            column_indices: df.column_indices.to_vec(),
            file_major_version: df.file_major_version,
            file_minor_version: df.file_minor_version,
            file_size_bytes: df.file_size_bytes.get().map_or(0, |v| v.get()),
            base_id: df.base_id,
        }
    }
}

impl TryFrom<pb::DataFile> for DataFile {
    type Error = Error;

    fn try_from(proto: pb::DataFile) -> Result<Self> {
        Ok(Self {
            path: proto.path,
            fields: Arc::from(proto.fields),
            column_indices: Arc::from(proto.column_indices),
            file_major_version: proto.file_major_version,
            file_minor_version: proto.file_minor_version,
            file_size_bytes: CachedFileSize::new(proto.file_size_bytes),
            base_id: proto.base_id,
        })
    }
}

/// Which `(physical offset, field)` cells a [`DataOverlayFile`] provides values
/// for.
///
/// The coverage bitmaps index **physical** row offsets (positions in the base
/// data files, counting deleted rows), so they are stable across deletions, like
/// deletion vectors. Bitmaps are parsed from their 32-bit Roaring encoding once
/// when the fragment is loaded and held behind an `Arc` so cloning a fragment is
/// cheap; use [`DataOverlayFile::coverage_for_field`] to obtain the one that
/// applies to a given field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(into = "OverlayCoverageBytes", try_from = "OverlayCoverageBytes")]
pub enum OverlayCoverage {
    /// A single bitmap that applies to every field in the overlay's
    /// `data_file.fields` (a dense / rectangular overlay): every covered offset
    /// has a value for every field.
    Shared(Arc<RoaringBitmap>),
    /// One bitmap per field, in the same order as the overlay's
    /// `data_file.fields` (a sparse overlay): different fields may cover
    /// different offset sets.
    PerField(Vec<Arc<RoaringBitmap>>),
}

/// Serialized form of [`OverlayCoverage`] — each bitmap as its 32-bit Roaring
/// byte encoding. The in-memory form parses these once at load.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum OverlayCoverageBytes {
    Shared(Vec<u8>),
    PerField(Vec<Vec<u8>>),
}

fn deserialize_roaring(bytes: &[u8]) -> Result<RoaringBitmap> {
    RoaringBitmap::deserialize_from(bytes).map_err(|e| {
        Error::invalid_input(format!(
            "failed to deserialize overlay coverage bitmap: {e}"
        ))
    })
}

fn serialize_roaring(bitmap: &RoaringBitmap) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(bitmap.serialized_size());
    // Writing to a Vec is infallible.
    bitmap.serialize_into(&mut bytes).unwrap();
    bytes
}

impl From<OverlayCoverage> for OverlayCoverageBytes {
    fn from(coverage: OverlayCoverage) -> Self {
        match coverage {
            OverlayCoverage::Shared(bitmap) => Self::Shared(serialize_roaring(&bitmap)),
            OverlayCoverage::PerField(bitmaps) => {
                Self::PerField(bitmaps.iter().map(|b| serialize_roaring(b)).collect())
            }
        }
    }
}

impl TryFrom<OverlayCoverageBytes> for OverlayCoverage {
    type Error = Error;

    fn try_from(bytes: OverlayCoverageBytes) -> Result<Self> {
        Ok(match bytes {
            OverlayCoverageBytes::Shared(b) => Self::Shared(Arc::new(deserialize_roaring(&b)?)),
            OverlayCoverageBytes::PerField(bs) => Self::PerField(
                bs.iter()
                    .map(|b| deserialize_roaring(b).map(Arc::new))
                    .collect::<Result<_>>()?,
            ),
        })
    }
}

impl DeepSizeOf for OverlayCoverage {
    fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
        // RoaringBitmap does not expose its allocation size; its serialized size
        // is a cheap, close proxy for the heap it holds.
        let bitmap_heap = |bitmap: &RoaringBitmap| {
            std::mem::size_of::<RoaringBitmap>() + bitmap.serialized_size()
        };
        match self {
            Self::Shared(bitmap) => bitmap_heap(bitmap),
            Self::PerField(bitmaps) => {
                bitmaps.capacity() * std::mem::size_of::<Arc<RoaringBitmap>>()
                    + bitmaps.iter().map(|b| bitmap_heap(b)).sum::<usize>()
            }
        }
    }
}

impl OverlayCoverage {
    /// Build a dense coverage from a single bitmap shared across every field.
    pub fn dense(bitmap: RoaringBitmap) -> Self {
        Self::Shared(Arc::new(bitmap))
    }

    /// Build a sparse coverage from one bitmap per field.
    pub fn sparse(bitmaps: Vec<RoaringBitmap>) -> Self {
        Self::PerField(bitmaps.into_iter().map(Arc::new).collect())
    }
}

/// An overlay file supplies new values for a subset of `(physical offset, field)`
/// cells within a fragment, without rewriting the fragment's base data files. See
/// the Data Overlay Files specification for the full resolution, coverage, and
/// versioning rules.
///
/// The overlay's `data_file` stores one value column per field in
/// `data_file.fields`, with **no** row-offset key column. Within a value column,
/// the position of a covered offset's value is the **rank** (0-based count of set
/// bits below it) of that offset in the field's coverage bitmap.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct DataOverlayFile {
    /// The data file storing the overlay's new cell values.
    pub data_file: DataFile,
    /// Which cells this overlay provides values for.
    pub coverage: OverlayCoverage,
    /// The dataset version at which this overlay became effective (the version of
    /// the commit that introduced it, stamped at commit time and re-stamped on
    /// retry). Higher wins when two overlays cover the same `(offset, field)`.
    pub committed_version: u64,
}

impl DataOverlayFile {
    /// The parsed coverage bitmap that applies to the field stored at
    /// `field_pos` within `data_file.fields`.
    ///
    /// For a dense overlay the same shared bitmap is returned for every field;
    /// for a sparse overlay the per-field bitmap at `field_pos` is returned. The
    /// bitmap is already parsed, so this is a cheap `Arc` clone.
    pub fn coverage_for_field(&self, field_pos: usize) -> Result<Arc<RoaringBitmap>> {
        match &self.coverage {
            OverlayCoverage::Shared(bitmap) => Ok(bitmap.clone()),
            OverlayCoverage::PerField(bitmaps) => {
                bitmaps.get(field_pos).cloned().ok_or_else(|| {
                    Error::invalid_input(format!(
                        "overlay field_coverage has {} bitmaps but field position {} was requested",
                        bitmaps.len(),
                        field_pos
                    ))
                })
            }
        }
    }
}

/// Overlays are stored newest-last: a later list position is newer, and ties in
/// `committed_version` are broken by position. Loading stable-sorts by
/// `committed_version` so resolution can rely on the ordering without
/// re-checking; the stable sort preserves the position tiebreak for equal
/// versions.
fn sort_overlays_newest_last(overlays: &mut [DataOverlayFile]) {
    overlays.sort_by_key(|overlay| overlay.committed_version);
}

impl From<&DataOverlayFile> for pb::DataOverlayFile {
    fn from(overlay: &DataOverlayFile) -> Self {
        let coverage = match &overlay.coverage {
            OverlayCoverage::Shared(bitmap) => {
                pb::data_overlay_file::Coverage::SharedOffsetBitmap(serialize_roaring(bitmap))
            }
            OverlayCoverage::PerField(bitmaps) => {
                pb::data_overlay_file::Coverage::FieldCoverage(pb::FieldCoverage {
                    offset_bitmaps: bitmaps.iter().map(|b| serialize_roaring(b)).collect(),
                })
            }
        };
        Self {
            data_file: Some(pb::DataFile::from(&overlay.data_file)),
            coverage: Some(coverage),
            committed_version: overlay.committed_version,
        }
    }
}

impl TryFrom<pb::DataOverlayFile> for DataOverlayFile {
    type Error = Error;

    fn try_from(proto: pb::DataOverlayFile) -> Result<Self> {
        let data_file = proto
            .data_file
            .ok_or_else(|| Error::invalid_input("DataOverlayFile is missing its data_file"))?;
        let coverage = match proto.coverage {
            Some(pb::data_overlay_file::Coverage::SharedOffsetBitmap(bytes)) => {
                OverlayCoverage::Shared(Arc::new(deserialize_roaring(&bytes)?))
            }
            Some(pb::data_overlay_file::Coverage::FieldCoverage(fc)) => OverlayCoverage::PerField(
                fc.offset_bitmaps
                    .iter()
                    .map(|b| deserialize_roaring(b).map(Arc::new))
                    .collect::<Result<_>>()?,
            ),
            None => {
                return Err(Error::invalid_input(
                    "DataOverlayFile is missing its coverage",
                ));
            }
        };
        Ok(Self {
            data_file: DataFile::try_from(data_file)?,
            coverage,
            committed_version: proto.committed_version,
        })
    }
}

/// Interns repeated data so that fragments with identical content share a
/// single heap allocation via `Arc`.
///
/// At 20M fragments the deduplication typically saves multiple GB of heap
/// because every fragment in a homogeneous table carries the same field list,
/// and post-compaction fragments share identical version metadata bytes.
///
/// Uses a `Vec`-based linear scan when the cache is small (<=16 entries)
/// and upgrades to `HashMap` for larger caches. In the common homogeneous
/// case (1-3 unique values), linear scan avoids per-fragment hashing overhead.
#[derive(Default)]
pub struct DataFileFieldInterner {
    fields: InternCache<i32>,
    column_indices: InternCache<i32>,
    inline_bytes: InternCache<u8>,
}

/// A cache that uses linear scan for small sizes and HashMap for large.
/// The threshold is chosen so that scan + compare is cheaper than hash for
/// typical payload sizes (20-200 bytes).
enum InternCache<T: Eq + std::hash::Hash + Clone> {
    Small(Vec<Arc<[T]>>),
    Large(HashMap<Arc<[T]>, ()>),
}

const INTERN_CACHE_UPGRADE_THRESHOLD: usize = 16;

impl<T: Eq + std::hash::Hash + Clone> Default for InternCache<T> {
    fn default() -> Self {
        Self::Small(Vec::new())
    }
}

impl<T: Eq + std::hash::Hash + Clone> InternCache<T> {
    fn intern(&mut self, v: Vec<T>) -> Arc<[T]> {
        match self {
            Self::Small(entries) => {
                for existing in entries.iter() {
                    if existing.as_ref() == v.as_slice() {
                        return existing.clone();
                    }
                }
                let arc: Arc<[T]> = Arc::from(v);
                entries.push(arc.clone());
                if entries.len() > INTERN_CACHE_UPGRADE_THRESHOLD {
                    let mut map = HashMap::with_capacity(entries.len());
                    for e in entries.drain(..) {
                        map.insert(e, ());
                    }
                    *self = Self::Large(map);
                }
                arc
            }
            Self::Large(map) => {
                if let Some((existing, _)) = map.get_key_value(v.as_slice()) {
                    existing.clone()
                } else {
                    let arc: Arc<[T]> = Arc::from(v);
                    map.insert(arc.clone(), ());
                    arc
                }
            }
        }
    }
}

impl DataFileFieldInterner {
    /// Intern a `RowDatasetVersionMeta`, deduplicating inline byte payloads.
    /// Accepts the protobuf oneof value directly to avoid an intermediate
    /// `Arc<[u8]>` allocation that would need to be `.to_vec()`'d for the key lookup.
    fn intern_last_updated_version_meta(
        cache: &mut InternCache<u8>,
        pb: pb::data_fragment::LastUpdatedAtVersionSequence,
    ) -> Result<RowDatasetVersionMeta> {
        match pb {
            pb::data_fragment::LastUpdatedAtVersionSequence::InlineLastUpdatedAtVersions(data) => {
                Ok(RowDatasetVersionMeta::Inline(cache.intern(data)))
            }
            pb::data_fragment::LastUpdatedAtVersionSequence::ExternalLastUpdatedAtVersions(
                file,
            ) => Ok(RowDatasetVersionMeta::External(ExternalFile {
                path: file.path,
                offset: file.offset,
                size: file.size,
            })),
        }
    }

    /// Intern a `RowDatasetVersionMeta`, deduplicating inline byte payloads.
    fn intern_created_version_meta(
        cache: &mut InternCache<u8>,
        pb: pb::data_fragment::CreatedAtVersionSequence,
    ) -> Result<RowDatasetVersionMeta> {
        match pb {
            pb::data_fragment::CreatedAtVersionSequence::InlineCreatedAtVersions(data) => {
                Ok(RowDatasetVersionMeta::Inline(cache.intern(data)))
            }
            pb::data_fragment::CreatedAtVersionSequence::ExternalCreatedAtVersions(file) => {
                Ok(RowDatasetVersionMeta::External(ExternalFile {
                    path: file.path,
                    offset: file.offset,
                    size: file.size,
                }))
            }
        }
    }

    /// Convert a protobuf `DataFile`, interning `fields` and `column_indices`.
    pub fn intern_data_file(&mut self, proto: pb::DataFile) -> Result<DataFile> {
        Ok(DataFile {
            path: proto.path,
            fields: self.fields.intern(proto.fields),
            column_indices: self.column_indices.intern(proto.column_indices),
            file_major_version: proto.file_major_version,
            file_minor_version: proto.file_minor_version,
            file_size_bytes: CachedFileSize::new(proto.file_size_bytes),
            base_id: proto.base_id,
        })
    }

    /// Convert a protobuf `DataFragment`, interning fields and version metadata.
    pub fn intern_fragment(&mut self, p: pb::DataFragment) -> Result<Fragment> {
        let physical_rows = if p.physical_rows > 0 {
            Some(p.physical_rows as usize)
        } else {
            None
        };
        let last_updated_at_version_meta = p
            .last_updated_at_version_sequence
            .map(|pb| Self::intern_last_updated_version_meta(&mut self.inline_bytes, pb))
            .transpose()?;
        let created_at_version_meta = p
            .created_at_version_sequence
            .map(|pb| Self::intern_created_version_meta(&mut self.inline_bytes, pb))
            .transpose()?;
        Ok(Fragment {
            id: p.id,
            files: p
                .files
                .into_iter()
                .map(|f| self.intern_data_file(f))
                .collect::<Result<_>>()?,
            overlays: {
                let mut overlays = p
                    .overlays
                    .into_iter()
                    .map(DataOverlayFile::try_from)
                    .collect::<Result<Vec<_>>>()?;
                sort_overlays_newest_last(&mut overlays);
                overlays
            },
            deletion_file: p.deletion_file.map(DeletionFile::try_from).transpose()?,
            row_id_meta: p.row_id_sequence.map(RowIdMeta::try_from).transpose()?,
            physical_rows,
            last_updated_at_version_meta,
            created_at_version_meta,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
#[serde(rename_all = "lowercase")]
pub enum DeletionFileType {
    Array,
    Bitmap,
}

impl DeletionFileType {
    // TODO: pub(crate)
    pub fn suffix(&self) -> &str {
        match self {
            Self::Array => "arrow",
            Self::Bitmap => "bin",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct DeletionFile {
    pub read_version: u64,
    pub id: u64,
    pub file_type: DeletionFileType,
    /// Number of deleted rows in this file. If None, this is unknown.
    pub num_deleted_rows: Option<usize>,
    pub base_id: Option<u32>,
}

impl TryFrom<pb::DeletionFile> for DeletionFile {
    type Error = Error;

    fn try_from(value: pb::DeletionFile) -> Result<Self> {
        let file_type = match value.file_type {
            0 => DeletionFileType::Array,
            1 => DeletionFileType::Bitmap,
            _ => {
                return Err(Error::not_supported_source(
                    "Unknown deletion file type".into(),
                ));
            }
        };
        let num_deleted_rows = if value.num_deleted_rows == 0 {
            None
        } else {
            Some(value.num_deleted_rows as usize)
        };
        Ok(Self {
            read_version: value.read_version,
            id: value.id,
            file_type,
            num_deleted_rows,
            base_id: value.base_id,
        })
    }
}

/// A reference to a part of a file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct ExternalFile {
    pub path: String,
    pub offset: u64,
    pub size: u64,
}

/// Metadata about location of the row id sequence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub enum RowIdMeta {
    Inline(Vec<u8>),
    External(ExternalFile),
}

impl TryFrom<pb::data_fragment::RowIdSequence> for RowIdMeta {
    type Error = Error;

    fn try_from(value: pb::data_fragment::RowIdSequence) -> Result<Self> {
        match value {
            pb::data_fragment::RowIdSequence::InlineRowIds(data) => Ok(Self::Inline(data)),
            pb::data_fragment::RowIdSequence::ExternalRowIds(file) => {
                Ok(Self::External(ExternalFile {
                    path: file.path.clone(),
                    offset: file.offset,
                    size: file.size,
                }))
            }
        }
    }
}

/// Data fragment.
///
/// A fragment is a set of files which represent the different columns of the same rows.
/// If column exists in the schema, but the related file does not exist, treat this column as `nulls`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct Fragment {
    /// Fragment ID
    pub id: u64,

    /// Files within the fragment.
    pub files: Vec<DataFile>,

    /// Overlay files supplying new values for a subset of cells without
    /// rewriting the base data files. Order is significant: a later entry is
    /// newer than an earlier one. See [`DataOverlayFile`] for resolution rules.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub overlays: Vec<DataOverlayFile>,

    /// Optional file with deleted local row offsets.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deletion_file: Option<DeletionFile>,

    /// RowIndex
    #[serde(skip_serializing_if = "Option::is_none")]
    pub row_id_meta: Option<RowIdMeta>,

    /// Original number of rows in the fragment. If this is None, then it is
    /// unknown. This is only optional for legacy reasons. All new tables should
    /// have this set.
    pub physical_rows: Option<usize>,

    /// Last updated at version metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_updated_at_version_meta: Option<RowDatasetVersionMeta>,

    /// Created at version metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at_version_meta: Option<RowDatasetVersionMeta>,
}

impl Fragment {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            files: vec![],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: None,
            physical_rows: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        }
    }

    pub fn num_rows(&self) -> Option<usize> {
        match (self.physical_rows, &self.deletion_file) {
            // Known fragment length, no deletion file.
            (Some(len), None) => Some(len),
            // Known fragment length, but don't know deletion file size.
            (
                Some(len),
                Some(DeletionFile {
                    num_deleted_rows: Some(num_deleted_rows),
                    ..
                }),
            ) => Some(len - num_deleted_rows),
            _ => None,
        }
    }

    pub fn from_json(json: &str) -> Result<Self> {
        let fragment: Self = serde_json::from_str(json)?;
        Ok(fragment)
    }

    /// Create a `Fragment` with one DataFile
    pub fn with_file_legacy(
        id: u64,
        path: &str,
        schema: &Schema,
        physical_rows: Option<usize>,
    ) -> Self {
        Self {
            id,
            files: vec![DataFile::new_legacy(path, schema, None, None)],
            overlays: vec![],
            deletion_file: None,
            physical_rows,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        }
    }

    pub fn with_file(
        mut self,
        path: impl Into<String>,
        field_ids: Vec<i32>,
        column_indices: Vec<i32>,
        version: &LanceFileVersion,
        file_size_bytes: Option<NonZero<u64>>,
    ) -> Self {
        let (major, minor) = version.to_numbers();
        let data_file = DataFile::new(
            path,
            field_ids,
            column_indices,
            major,
            minor,
            file_size_bytes,
            None,
        );
        self.files.push(data_file);
        self
    }

    pub fn with_physical_rows(mut self, physical_rows: usize) -> Self {
        self.physical_rows = Some(physical_rows);
        self
    }

    pub fn add_file(
        &mut self,
        path: impl Into<String>,
        field_ids: Vec<i32>,
        column_indices: Vec<i32>,
        version: &LanceFileVersion,
        file_size_bytes: Option<NonZero<u64>>,
    ) {
        let (major, minor) = version.to_numbers();
        self.files.push(DataFile::new(
            path,
            field_ids,
            column_indices,
            major,
            minor,
            file_size_bytes,
            None,
        ));
    }

    /// Add a new [`DataFile`] to this fragment.
    pub fn add_file_legacy(&mut self, path: &str, schema: &Schema) {
        self.files
            .push(DataFile::new_legacy(path, schema, None, None));
    }

    // True if this fragment is made up of legacy v1 files, false otherwise
    pub fn has_legacy_files(&self) -> bool {
        // If any file in a fragment is legacy then all files in the fragment must be
        self.files[0].is_legacy_file()
    }

    // Helper method to infer the Lance version from a set of fragments
    //
    // Returns None if there are no data files
    // Returns an error if the data files have different versions
    pub fn try_infer_version(fragments: &[Self]) -> Result<Option<LanceFileVersion>> {
        // Otherwise we need to check the actual file versions
        // Determine version from first file
        let Some(sample_file) = fragments
            .iter()
            .find(|f| !f.files.is_empty())
            .map(|f| &f.files[0])
        else {
            return Ok(None);
        };
        let file_version = LanceFileVersion::try_from_major_minor(
            sample_file.file_major_version,
            sample_file.file_minor_version,
        )?;
        // Ensure all files match
        for frag in fragments {
            for file in &frag.files {
                let this_file_version = LanceFileVersion::try_from_major_minor(
                    file.file_major_version,
                    file.file_minor_version,
                )?;
                if file_version != this_file_version {
                    return Err(Error::invalid_input(format!(
                        "All data files must have the same version.  Detected both {} and {}",
                        file_version, this_file_version
                    )));
                }
            }
        }
        Ok(Some(file_version))
    }
}

impl TryFrom<pb::DataFragment> for Fragment {
    type Error = Error;

    fn try_from(p: pb::DataFragment) -> Result<Self> {
        let physical_rows = if p.physical_rows > 0 {
            Some(p.physical_rows as usize)
        } else {
            None
        };
        Ok(Self {
            id: p.id,
            files: p
                .files
                .into_iter()
                .map(DataFile::try_from)
                .collect::<Result<_>>()?,
            overlays: {
                let mut overlays = p
                    .overlays
                    .into_iter()
                    .map(DataOverlayFile::try_from)
                    .collect::<Result<Vec<_>>>()?;
                sort_overlays_newest_last(&mut overlays);
                overlays
            },
            deletion_file: p.deletion_file.map(DeletionFile::try_from).transpose()?,
            row_id_meta: p.row_id_sequence.map(RowIdMeta::try_from).transpose()?,
            physical_rows,
            last_updated_at_version_meta: p
                .last_updated_at_version_sequence
                .map(RowDatasetVersionMeta::try_from)
                .transpose()?,
            created_at_version_meta: p
                .created_at_version_sequence
                .map(RowDatasetVersionMeta::try_from)
                .transpose()?,
        })
    }
}

impl From<&Fragment> for pb::DataFragment {
    fn from(f: &Fragment) -> Self {
        let deletion_file = f.deletion_file.as_ref().map(|f| {
            let file_type = match f.file_type {
                DeletionFileType::Array => pb::deletion_file::DeletionFileType::ArrowArray,
                DeletionFileType::Bitmap => pb::deletion_file::DeletionFileType::Bitmap,
            };
            pb::DeletionFile {
                read_version: f.read_version,
                id: f.id,
                file_type: file_type.into(),
                num_deleted_rows: f.num_deleted_rows.unwrap_or_default() as u64,
                base_id: f.base_id,
            }
        });

        let row_id_sequence = f.row_id_meta.as_ref().map(|m| match m {
            RowIdMeta::Inline(data) => pb::data_fragment::RowIdSequence::InlineRowIds(data.clone()),
            RowIdMeta::External(file) => {
                pb::data_fragment::RowIdSequence::ExternalRowIds(pb::ExternalFile {
                    path: file.path.clone(),
                    offset: file.offset,
                    size: file.size,
                })
            }
        });
        let last_updated_at_version_sequence =
            last_updated_at_version_meta_to_pb(&f.last_updated_at_version_meta);
        let created_at_version_sequence = created_at_version_meta_to_pb(&f.created_at_version_meta);
        Self {
            id: f.id,
            files: f.files.iter().map(pb::DataFile::from).collect(),
            overlays: f.overlays.iter().map(pb::DataOverlayFile::from).collect(),
            deletion_file,
            row_id_sequence,
            physical_rows: f.physical_rows.unwrap_or_default() as u64,
            last_updated_at_version_sequence,
            created_at_version_sequence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{
        DataType, Field as ArrowField, Fields as ArrowFields, Schema as ArrowSchema,
    };
    use object_store::path::Path;
    use serde_json::{Value, json};

    #[test]
    fn test_data_overlay_roundtrip() {
        // A fragment carrying a dense overlay round-trips through protobuf and
        // back, and the parsed coverage bitmap is recovered per field.
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(1);
        bitmap.insert(3);

        let overlay = DataOverlayFile {
            data_file: DataFile::new_legacy_from_fields("overlay-0.lance", vec![3], None),
            coverage: OverlayCoverage::dense(bitmap.clone()),
            committed_version: 7,
        };
        let mut fragment = Fragment::new(0);
        fragment.files = vec![DataFile::new_legacy_from_fields(
            "base.lance",
            vec![1, 3],
            None,
        )];
        fragment.overlays = vec![overlay];

        let proto = pb::DataFragment::from(&fragment);
        assert_eq!(proto.overlays.len(), 1);
        let round_tripped = Fragment::try_from(proto).unwrap();
        assert_eq!(round_tripped, fragment);

        // Dense coverage applies to every field.
        let recovered = round_tripped.overlays[0].coverage_for_field(0).unwrap();
        assert_eq!(*recovered, bitmap);
        assert_eq!(
            *round_tripped.overlays[0].coverage_for_field(5).unwrap(),
            bitmap
        );
    }

    #[test]
    fn test_data_overlay_sparse_per_field_coverage() {
        // A sparse overlay carries one bitmap per field, recovered by position.
        let name_coverage = RoaringBitmap::from_iter([2u32, 3]);
        let embedding_coverage = RoaringBitmap::from_iter([1u32]);
        let overlay = DataOverlayFile {
            data_file: DataFile::new_legacy_from_fields("overlay-1.lance", vec![2, 4], None),
            coverage: OverlayCoverage::sparse(vec![
                name_coverage.clone(),
                embedding_coverage.clone(),
            ]),
            committed_version: 3,
        };
        let mut fragment = Fragment::new(1);
        fragment.overlays = vec![overlay];

        let round_tripped = Fragment::try_from(pb::DataFragment::from(&fragment)).unwrap();
        assert_eq!(
            *round_tripped.overlays[0].coverage_for_field(0).unwrap(),
            name_coverage
        );
        assert_eq!(
            *round_tripped.overlays[0].coverage_for_field(1).unwrap(),
            embedding_coverage
        );
    }

    #[test]
    fn test_data_overlay_missing_fields_error() {
        // A DataOverlayFile proto missing its coverage or data_file is rejected.
        let no_coverage = pb::DataOverlayFile {
            data_file: Some(pb::DataFile::from(&DataFile::new_legacy_from_fields(
                "overlay.lance",
                vec![3],
                None,
            ))),
            coverage: None,
            committed_version: 1,
        };
        let err = DataOverlayFile::try_from(no_coverage).unwrap_err();
        assert!(err.to_string().contains("missing its coverage"), "{err}");

        let no_data_file = pb::DataOverlayFile {
            data_file: None,
            coverage: Some(pb::data_overlay_file::Coverage::SharedOffsetBitmap(
                serialize_roaring(&RoaringBitmap::from_iter([0u32])),
            )),
            committed_version: 1,
        };
        let err = DataOverlayFile::try_from(no_data_file).unwrap_err();
        assert!(err.to_string().contains("missing its data_file"), "{err}");
    }

    #[test]
    fn test_overlays_sorted_newest_last_on_load() {
        // Overlays load stable-sorted by committed_version (newest last), with
        // list position preserved as the tiebreak for equal versions.
        let mk = |version: u64, field: i32| DataOverlayFile {
            data_file: DataFile::new_legacy_from_fields("o.lance", vec![field], None),
            coverage: OverlayCoverage::dense(RoaringBitmap::from_iter([0u32])),
            committed_version: version,
        };
        let mut fragment = Fragment::new(0);
        // Written out of order: v5, v2, v2 (second), v3.
        fragment.overlays = vec![mk(5, 1), mk(2, 2), mk(2, 3), mk(3, 4)];

        let loaded = Fragment::try_from(pb::DataFragment::from(&fragment)).unwrap();
        let versions: Vec<u64> = loaded
            .overlays
            .iter()
            .map(|o| o.committed_version)
            .collect();
        assert_eq!(versions, vec![2, 2, 3, 5]);
        // Stable: the two v2 overlays keep their original relative order (field 2
        // before field 3).
        assert_eq!(
            loaded.overlays[0].data_file.fields.as_ref(),
            [2i32].as_slice()
        );
        assert_eq!(
            loaded.overlays[1].data_file.fields.as_ref(),
            [3i32].as_slice()
        );
    }

    #[test]
    fn test_overlay_coverage_serde_json_roundtrip() {
        // The custom serde impl round-trips through JSON for dense/sparse,
        // including empty bitmaps and a zero-bitmap sparse coverage.
        for coverage in [
            OverlayCoverage::dense(RoaringBitmap::from_iter([1u32, 5, 100])),
            OverlayCoverage::dense(RoaringBitmap::new()),
            OverlayCoverage::sparse(vec![
                RoaringBitmap::from_iter([2u32, 3]),
                RoaringBitmap::new(),
            ]),
            OverlayCoverage::sparse(vec![]),
        ] {
            let json = serde_json::to_string(&coverage).unwrap();
            let back: OverlayCoverage = serde_json::from_str(&json).unwrap();
            assert_eq!(back, coverage);
        }
    }

    #[test]
    fn test_coverage_for_field_out_of_bounds() {
        let overlay = DataOverlayFile {
            data_file: DataFile::new_legacy_from_fields("o.lance", vec![2, 4], None),
            coverage: OverlayCoverage::sparse(vec![
                RoaringBitmap::from_iter([1u32]),
                RoaringBitmap::from_iter([2u32]),
            ]),
            committed_version: 1,
        };
        assert!(overlay.coverage_for_field(0).is_ok());
        assert!(overlay.coverage_for_field(1).is_ok());
        let err = overlay.coverage_for_field(5).unwrap_err();
        assert!(err.to_string().contains("field position"), "{err}");
    }

    #[test]
    fn test_new_fragment() {
        let path = "foobar.lance";

        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new(
                "s",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("si", DataType::Int32, false),
                    ArrowField::new("sb", DataType::Binary, true),
                ])),
                true,
            ),
            ArrowField::new("bool", DataType::Boolean, true),
        ]);
        let schema = Schema::try_from(&arrow_schema).unwrap();
        let fragment = Fragment::with_file_legacy(123, path, &schema, Some(10));

        assert_eq!(123, fragment.id);
        assert_eq!(
            fragment.files,
            vec![DataFile::new_legacy_from_fields(
                path.to_string(),
                vec![0, 1, 2, 3],
                None,
            )]
        )
    }

    #[test]
    fn test_roundtrip_fragment() {
        let mut fragment = Fragment::new(123);
        let schema = ArrowSchema::new(vec![ArrowField::new("x", DataType::Float16, true)]);
        fragment.add_file_legacy("foobar.lance", &Schema::try_from(&schema).unwrap());
        fragment.deletion_file = Some(DeletionFile {
            read_version: 123,
            id: 456,
            file_type: DeletionFileType::Array,
            num_deleted_rows: Some(10),
            base_id: None,
        });

        let proto = pb::DataFragment::from(&fragment);
        let fragment2 = Fragment::try_from(proto).unwrap();
        assert_eq!(fragment, fragment2);

        fragment.deletion_file = None;
        let proto = pb::DataFragment::from(&fragment);
        let fragment2 = Fragment::try_from(proto).unwrap();
        assert_eq!(fragment, fragment2);
    }

    #[test]
    fn test_to_json() {
        let mut fragment = Fragment::new(123);
        let schema = ArrowSchema::new(vec![ArrowField::new("x", DataType::Float16, true)]);
        fragment.add_file_legacy("foobar.lance", &Schema::try_from(&schema).unwrap());
        fragment.deletion_file = Some(DeletionFile {
            read_version: 123,
            id: 456,
            file_type: DeletionFileType::Array,
            num_deleted_rows: Some(10),
            base_id: None,
        });

        let json = serde_json::to_string(&fragment).unwrap();

        let value: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            value,
            json!({
                "id": 123,
                "files":[
                    {"path": "foobar.lance", "fields": [0], "column_indices": [], 
                     "file_major_version": MAJOR_VERSION, "file_minor_version": MINOR_VERSION,
                     "file_size_bytes": null, "base_id": null }
                ],
                "deletion_file": {"read_version": 123, "id": 456, "file_type": "array",
                                  "num_deleted_rows": 10, "base_id": null},
                "physical_rows": None::<usize>}),
        );

        let frag2 = Fragment::from_json(&json).unwrap();
        assert_eq!(fragment, frag2);
    }

    #[test]
    fn data_file_validate_allows_extra_columns() {
        let data_file = DataFile {
            path: "foo.lance".to_string(),
            fields: Arc::from([1, 2]),
            // One extra column without a field id mapping
            column_indices: Arc::from([0, 1, 2]),
            file_major_version: MAJOR_VERSION as u32,
            file_minor_version: MINOR_VERSION as u32,
            file_size_bytes: Default::default(),
            base_id: None,
        };

        let base_path = Path::from("base");
        data_file
            .validate(&base_path)
            .expect("validation should allow extra columns without field ids");
    }
}
