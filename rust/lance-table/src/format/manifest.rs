// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::prelude::*;
use lance_file::datatypes::{populate_schema_dictionary, Fields, FieldsWithMeta};
use lance_file::reader::FileReader;
use lance_io::traits::{ProtoStruct, Reader};
use object_store::path::Path;
use prost::Message;
use prost_types::Timestamp;

use super::Fragment;
use crate::feature_flags::FLAG_MOVE_STABLE_ROW_IDS;
use crate::format::pb;
use lance_core::cache::FileMetadataCache;
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use lance_io::utils::read_struct;
use snafu::{location, Location};

/// Manifest of a dataset
///
///  * Schema
///  * Version
///  * Fragments.
///  * Indices.
#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    /// Dataset schema.
    pub schema: Schema,

    /// Dataset version
    pub version: u64,

    /// Version of the writer library that wrote this manifest.
    pub writer_version: Option<WriterVersion>,

    /// Fragments, the pieces to build the dataset.
    pub fragments: Arc<Vec<Fragment>>,

    /// The file position of the version aux data.
    pub version_aux_data: usize,

    /// The file position of the index metadata.
    pub index_section: Option<usize>,

    /// The creation timestamp with nanosecond resolution as 128-bit integer
    pub timestamp_nanos: u128,

    /// An optional string tag for this version
    pub tag: Option<String>,

    /// The reader flags
    pub reader_feature_flags: u64,

    /// The writer flags
    pub writer_feature_flags: u64,

    /// The max fragment id used so far
    pub max_fragment_id: u32,

    /// The path to the transaction file, relative to the root of the dataset
    pub transaction_file: Option<String>,

    /// Precomputed logic offset of each fragment
    /// accelerating the fragment search using offset ranges.
    fragment_offsets: Vec<usize>,

    /// The max row id used so far.
    pub next_row_id: u64,
}

fn compute_fragment_offsets(fragments: &[Fragment]) -> Vec<usize> {
    fragments
        .iter()
        .map(|f| f.num_rows().unwrap_or_default())
        .chain([0]) // Make the last offset to be the full-length of the dataset.
        .scan(0_usize, |offset, len| {
            let start = *offset;
            *offset += len;
            Some(start)
        })
        .collect()
}

impl Manifest {
    pub fn new(schema: Schema, fragments: Arc<Vec<Fragment>>) -> Self {
        let fragment_offsets = compute_fragment_offsets(&fragments);
        Self {
            schema,
            version: 1,
            writer_version: Some(WriterVersion::default()),
            fragments,
            version_aux_data: 0,
            index_section: None,
            timestamp_nanos: 0,
            tag: None,
            reader_feature_flags: 0,
            writer_feature_flags: 0,
            max_fragment_id: 0,
            transaction_file: None,
            fragment_offsets,
            next_row_id: 0,
        }
    }

    pub fn new_from_previous(
        previous: &Self,
        schema: Schema,
        fragments: Arc<Vec<Fragment>>,
    ) -> Self {
        let fragment_offsets = compute_fragment_offsets(&fragments);

        Self {
            schema,
            version: previous.version + 1,
            writer_version: Some(WriterVersion::default()),
            fragments,
            version_aux_data: 0,
            index_section: None, // Caller should update index if they want to keep them.
            timestamp_nanos: 0,  // This will be set on commit
            tag: None,
            reader_feature_flags: 0, // These will be set on commit
            writer_feature_flags: 0, // These will be set on commit
            max_fragment_id: previous.max_fragment_id,
            transaction_file: None,
            fragment_offsets,
            next_row_id: previous.next_row_id,
        }
    }

    /// Return the `timestamp_nanos` value as a Utc DateTime
    pub fn timestamp(&self) -> DateTime<Utc> {
        let nanos = self.timestamp_nanos % 1_000_000_000;
        let seconds = ((self.timestamp_nanos - nanos) / 1_000_000_000) as i64;
        Utc.from_utc_datetime(
            &DateTime::from_timestamp(seconds, nanos as u32)
                .unwrap_or_default()
                .naive_utc(),
        )
    }

    /// Set the `timestamp_nanos` value from a Utc DateTime
    pub fn set_timestamp(&mut self, nanos: u128) {
        self.timestamp_nanos = nanos;
    }

    /// Check the current fragment list and update the high water mark
    pub fn update_max_fragment_id(&mut self) {
        let max_fragment_id = self
            .fragments
            .iter()
            .map(|f| f.id)
            .max()
            .unwrap_or_default()
            .try_into()
            .unwrap();

        if max_fragment_id > self.max_fragment_id {
            self.max_fragment_id = max_fragment_id;
        }
    }

    /// Return the max fragment id.
    /// Note this does not support recycling of fragment ids.
    ///
    /// This will return None if there are no fragments.
    pub fn max_fragment_id(&self) -> Option<u64> {
        if self.max_fragment_id == 0 {
            // It might not have been updated, so the best we can do is recompute
            // it from the fragment list.
            self.fragments.iter().map(|f| f.id).max()
        } else {
            Some(self.max_fragment_id.into())
        }
    }

    /// Get the max used field id
    ///
    /// This is different than [Schema::max_field_id] because it also considers
    /// the field ids in the data files that have been dropped from the schema.
    pub fn max_field_id(&self) -> i32 {
        let schema_max_id = self.schema.max_field_id().unwrap_or(-1);
        let fragment_max_id = self
            .fragments
            .iter()
            .flat_map(|f| f.files.iter().flat_map(|file| file.fields.as_slice()))
            .max()
            .copied();
        let fragment_max_id = fragment_max_id.unwrap_or(-1);
        schema_max_id.max(fragment_max_id)
    }

    /// Return the fragments that are newer than the given manifest.
    /// Note this does not support recycling of fragment ids.
    pub fn fragments_since(&self, since: &Self) -> Result<Vec<Fragment>> {
        if since.version >= self.version {
            return Err(Error::io(
                format!(
                    "fragments_since: given version {} is newer than manifest version {}",
                    since.version, self.version
                ),
                location!(),
            ));
        }
        let start = since.max_fragment_id();
        Ok(self
            .fragments
            .iter()
            .filter(|&f| start.map(|s| f.id > s).unwrap_or(true))
            .cloned()
            .collect())
    }

    /// Find the fragments that contain the rows, identified by the offset range.
    ///
    /// Note that the offsets are the logical offsets of rows, not row IDs.
    ///
    ///
    /// Parameters
    /// ----------
    /// range: Range<usize>
    ///     Offset range
    ///
    /// Returns
    /// -------
    /// Vec<(usize, Fragment)>
    ///    A vector of `(starting_offset_of_fragment, fragment)` pairs.
    ///
    pub fn fragments_by_offset_range(&self, range: Range<usize>) -> Vec<(usize, &Fragment)> {
        let start = range.start;
        let end = range.end;
        let idx = self
            .fragment_offsets
            .binary_search(&start)
            .unwrap_or_else(|idx| idx - 1);

        let mut fragments = vec![];
        for i in idx..self.fragments.len() {
            if self.fragment_offsets[i] >= end
                || self.fragment_offsets[i] + self.fragments[i].num_rows().unwrap_or_default()
                    <= start
            {
                break;
            }
            fragments.push((self.fragment_offsets[i], &self.fragments[i]));
        }

        fragments
    }

    /// Whether the dataset uses move-stable row ids.
    pub fn uses_move_stable_row_ids(&self) -> bool {
        self.reader_feature_flags & FLAG_MOVE_STABLE_ROW_IDS != 0
    }

    /// Creates a serialized copy of the manifest, suitable for IPC or temp storage
    /// and can be used to create a dataset
    pub fn serialized(&self) -> Vec<u8> {
        let pb_manifest: pb::Manifest = self.into();
        pb_manifest.encode_to_vec()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WriterVersion {
    pub library: String,
    pub version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionPart {
    Major,
    Minor,
    Patch,
}

impl WriterVersion {
    /// Try to parse the version string as a semver string. Returns None if
    /// not successful.
    pub fn semver(&self) -> Option<(u32, u32, u32, Option<&str>)> {
        let mut parts = self.version.split('.');
        let major = parts.next().unwrap_or("0").parse().ok()?;
        let minor = parts.next().unwrap_or("0").parse().ok()?;
        let patch = parts.next().unwrap_or("0").parse().ok()?;
        let tag = parts.next();
        Some((major, minor, patch, tag))
    }

    pub fn semver_or_panic(&self) -> (u32, u32, u32, Option<&str>) {
        self.semver()
            .unwrap_or_else(|| panic!("Invalid writer version: {}", self.version))
    }

    /// Return true if self is older than the given major/minor/patch
    pub fn older_than(&self, major: u32, minor: u32, patch: u32) -> bool {
        let version = self.semver_or_panic();
        (version.0, version.1, version.2) < (major, minor, patch)
    }

    pub fn bump(&self, part: VersionPart, keep_tag: bool) -> Self {
        let parts = self.semver_or_panic();
        let tag = if keep_tag { parts.3 } else { None };
        let new_parts = match part {
            VersionPart::Major => (parts.0 + 1, parts.1, parts.2, tag),
            VersionPart::Minor => (parts.0, parts.1 + 1, parts.2, tag),
            VersionPart::Patch => (parts.0, parts.1, parts.2 + 1, tag),
        };
        let new_version = if let Some(tag) = tag {
            format!("{}.{}.{}.{}", new_parts.0, new_parts.1, new_parts.2, tag)
        } else {
            format!("{}.{}.{}", new_parts.0, new_parts.1, new_parts.2)
        };
        Self {
            library: self.library.clone(),
            version: new_version,
        }
    }
}

impl Default for WriterVersion {
    #[cfg(not(test))]
    fn default() -> Self {
        Self {
            library: "lance".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    // Unit tests always run as if they are in the next version.
    #[cfg(test)]
    fn default() -> Self {
        Self {
            library: "lance".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
        .bump(VersionPart::Patch, true)
    }
}

impl ProtoStruct for Manifest {
    type Proto = pb::Manifest;
}

impl TryFrom<pb::Manifest> for Manifest {
    type Error = Error;

    fn try_from(p: pb::Manifest) -> Result<Self> {
        let timestamp_nanos = p.timestamp.map(|ts| {
            let sec = ts.seconds as u128 * 1e9 as u128;
            let nanos = ts.nanos as u128;
            sec + nanos
        });
        // We only use the writer version if it is fully set.
        let writer_version = match p.writer_version {
            Some(pb::manifest::WriterVersion { library, version }) => {
                Some(WriterVersion { library, version })
            }
            _ => None,
        };
        let fragments = Arc::new(
            p.fragments
                .into_iter()
                .map(Fragment::try_from)
                .collect::<Result<Vec<_>>>()?,
        );
        let fragment_offsets = compute_fragment_offsets(fragments.as_slice());
        let fields_with_meta = FieldsWithMeta {
            fields: Fields(p.fields),
            metadata: p.metadata,
        };

        if FLAG_MOVE_STABLE_ROW_IDS & p.reader_feature_flags != 0
            && !fragments.iter().all(|frag| frag.row_id_meta.is_some())
        {
            return Err(Error::Internal {
                message: "All fragments must have row ids".into(),
                location: location!(),
            });
        }

        Ok(Self {
            schema: Schema::from(fields_with_meta),
            version: p.version,
            writer_version,
            fragments,
            version_aux_data: p.version_aux_data as usize,
            index_section: p.index_section.map(|i| i as usize),
            timestamp_nanos: timestamp_nanos.unwrap_or(0),
            tag: if p.tag.is_empty() { None } else { Some(p.tag) },
            reader_feature_flags: p.reader_feature_flags,
            writer_feature_flags: p.writer_feature_flags,
            max_fragment_id: p.max_fragment_id,
            transaction_file: if p.transaction_file.is_empty() {
                None
            } else {
                Some(p.transaction_file)
            },
            fragment_offsets,
            next_row_id: p.next_row_id,
        })
    }
}

impl From<&Manifest> for pb::Manifest {
    fn from(m: &Manifest) -> Self {
        let timestamp_nanos = if m.timestamp_nanos == 0 {
            None
        } else {
            let nanos = m.timestamp_nanos % 1e9 as u128;
            let seconds = ((m.timestamp_nanos - nanos) / 1e9 as u128) as i64;
            Some(Timestamp {
                seconds,
                nanos: nanos as i32,
            })
        };
        let fields_with_meta: FieldsWithMeta = (&m.schema).into();
        Self {
            fields: fields_with_meta.fields.0,
            version: m.version,
            writer_version: m
                .writer_version
                .as_ref()
                .map(|wv| pb::manifest::WriterVersion {
                    library: wv.library.clone(),
                    version: wv.version.clone(),
                }),
            fragments: m.fragments.iter().map(pb::DataFragment::from).collect(),
            metadata: fields_with_meta.metadata,
            version_aux_data: m.version_aux_data as u64,
            index_section: m.index_section.map(|i| i as u64),
            timestamp: timestamp_nanos,
            tag: m.tag.clone().unwrap_or_default(),
            reader_feature_flags: m.reader_feature_flags,
            writer_feature_flags: m.writer_feature_flags,
            max_fragment_id: m.max_fragment_id,
            transaction_file: m.transaction_file.clone().unwrap_or_default(),
            next_row_id: m.next_row_id,
        }
    }
}

#[async_trait]
pub trait SelfDescribingFileReader {
    /// Open a file reader without any cached schema
    ///
    /// In this case the schema will first need to be loaded
    /// from the file itself.
    ///
    /// When loading files from a dataset it is preferable to use
    /// the fragment reader to avoid this overhead.
    async fn try_new_self_described(
        object_store: &ObjectStore,
        path: &Path,
        cache: Option<&FileMetadataCache>,
    ) -> Result<Self>
    where
        Self: Sized,
    {
        let reader = object_store.open(path).await?;
        Self::try_new_self_described_from_reader(reader.into(), cache).await
    }

    async fn try_new_self_described_from_reader(
        reader: Arc<dyn Reader>,
        cache: Option<&FileMetadataCache>,
    ) -> Result<Self>
    where
        Self: Sized;
}

#[async_trait]
impl SelfDescribingFileReader for FileReader {
    async fn try_new_self_described_from_reader(
        reader: Arc<dyn Reader>,
        cache: Option<&FileMetadataCache>,
    ) -> Result<Self> {
        let metadata = Self::read_metadata(reader.as_ref(), cache).await?;
        let manifest_position = metadata.manifest_position.ok_or(Error::Internal {
            message: format!(
                "Attempt to open file at {} as self-describing but it did not contain a manifest",
                reader.path(),
            ),
            location: location!(),
        })?;
        let mut manifest: Manifest = read_struct(reader.as_ref(), manifest_position).await?;
        populate_schema_dictionary(&mut manifest.schema, reader.as_ref()).await?;
        let schema = manifest.schema;
        let max_field_id = schema.max_field_id().unwrap_or_default();
        Self::try_new_from_reader(
            reader.path(),
            reader.clone(),
            Some(metadata),
            schema,
            0,
            0,
            max_field_id,
            cache,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use crate::format::DataFile;

    use super::*;

    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
    use lance_core::datatypes::Field;

    #[test]
    fn test_writer_version() {
        let wv = WriterVersion::default();
        assert_eq!(wv.library, "lance");
        let parts = wv.semver().unwrap();
        assert_eq!(
            parts,
            (
                env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
                env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
                // Unit tests run against (major,minor,patch + 1)
                env!("CARGO_PKG_VERSION_PATCH").parse::<u32>().unwrap() + 1,
                None
            )
        );
        assert_eq!(
            format!("{}.{}.{}", parts.0, parts.1, parts.2 - 1),
            env!("CARGO_PKG_VERSION")
        );
        for part in &[VersionPart::Major, VersionPart::Minor, VersionPart::Patch] {
            let bumped = wv.bump(*part, false);
            let bumped_parts = bumped.semver_or_panic();
            assert!(wv.older_than(bumped_parts.0, bumped_parts.1, bumped_parts.2));
        }
    }

    #[test]
    fn test_fragments_by_offset_range() {
        let arrow_schema = ArrowSchema::new(vec![ArrowField::new(
            "a",
            arrow_schema::DataType::Int64,
            false,
        )]);
        let schema = Schema::try_from(&arrow_schema).unwrap();
        let fragments = vec![
            Fragment::with_file_legacy(0, "path1", &schema, Some(10)),
            Fragment::with_file_legacy(1, "path2", &schema, Some(15)),
            Fragment::with_file_legacy(2, "path3", &schema, Some(20)),
        ];
        let manifest = Manifest::new(schema, Arc::new(fragments));

        let actual = manifest.fragments_by_offset_range(0..10);
        assert_eq!(actual.len(), 1);
        assert_eq!(actual[0].0, 0);
        assert_eq!(actual[0].1.id, 0);

        let actual = manifest.fragments_by_offset_range(5..15);
        assert_eq!(actual.len(), 2);
        assert_eq!(actual[0].0, 0);
        assert_eq!(actual[0].1.id, 0);
        assert_eq!(actual[1].0, 10);
        assert_eq!(actual[1].1.id, 1);

        let actual = manifest.fragments_by_offset_range(15..50);
        assert_eq!(actual.len(), 2);
        assert_eq!(actual[0].0, 10);
        assert_eq!(actual[0].1.id, 1);
        assert_eq!(actual[1].0, 25);
        assert_eq!(actual[1].1.id, 2);

        // Out of range
        let actual = manifest.fragments_by_offset_range(45..100);
        assert!(actual.is_empty());

        assert!(manifest.fragments_by_offset_range(200..400).is_empty());
    }

    #[test]
    fn test_max_field_id() {
        // Validate that max field id handles varying field ids by fragment.
        let mut field0 =
            Field::try_from(ArrowField::new("a", arrow_schema::DataType::Int64, false)).unwrap();
        field0.set_id(-1, &mut 0);
        let mut field2 =
            Field::try_from(ArrowField::new("b", arrow_schema::DataType::Int64, false)).unwrap();
        field2.set_id(-1, &mut 2);

        let schema = Schema {
            fields: vec![field0, field2],
            metadata: Default::default(),
        };
        let fragments = vec![
            Fragment {
                id: 0,
                files: vec![DataFile::new_legacy_from_fields("path1", vec![0, 1, 2])],
                deletion_file: None,
                row_id_meta: None,
                physical_rows: None,
            },
            Fragment {
                id: 1,
                files: vec![
                    DataFile::new_legacy_from_fields("path2", vec![0, 1, 43]),
                    DataFile::new_legacy_from_fields("path3", vec![2]),
                ],
                deletion_file: None,
                row_id_meta: None,
                physical_rows: None,
            },
        ];

        let manifest = Manifest::new(schema, Arc::new(fragments));

        assert_eq!(manifest.max_field_id(), 43);
    }
}
