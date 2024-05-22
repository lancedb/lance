// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::Error;
use lance_file::format::{MAJOR_VERSION, MINOR_VERSION_NEXT};
use object_store::path::Path;
use serde::{Deserialize, Serialize};
use snafu::{location, Location};

use crate::format::pb;

use lance_core::datatypes::Schema;
use lance_core::error::Result;

/// Lance Data File
///
/// A data file is one piece of file storing data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataFile {
    /// Relative path of the data file to dataset root.
    pub path: String,
    /// The ids of fields in this file.
    pub fields: Vec<i32>,
    /// The offsets of the fields listed in `fields`, empty in v1 files
    ///
    /// Note that -1 is a possibility and it indices that the field has
    /// no top-level column in the file.
    #[serde(default)]
    pub column_indices: Vec<i32>,
    /// The major version of the file format used to write this file.
    #[serde(default)]
    pub file_major_version: u32,
    /// The minor version of the file format used to write this file.
    #[serde(default)]
    pub file_minor_version: u32,
}

impl DataFile {
    pub fn new(
        path: impl Into<String>,
        fields: Vec<i32>,
        column_indices: Vec<i32>,
        file_major_version: u32,
        file_minor_version: u32,
    ) -> Self {
        Self {
            path: path.into(),
            fields,
            column_indices,
            file_major_version,
            file_minor_version,
        }
    }

    pub fn new_legacy_from_fields(path: impl Into<String>, fields: Vec<i32>) -> Self {
        Self::new(path, fields, vec![], 0, 0)
    }

    pub fn new_legacy(path: impl Into<String>, schema: &Schema) -> Self {
        let mut field_ids = schema.field_ids();
        field_ids.sort();
        Self::new(path, field_ids, vec![], 0, 0)
    }

    pub fn schema(&self, full_schema: &Schema) -> Schema {
        full_schema.project_by_ids(&self.fields)
    }

    pub fn is_legacy_file(&self) -> bool {
        self.file_major_version == 0 && self.file_minor_version < 3
    }

    pub fn validate(&self, base_path: &Path) -> Result<()> {
        if self.is_legacy_file() {
            if !self.fields.windows(2).all(|w| w[0] < w[1]) {
                return Err(Error::corrupt_file(
                    base_path.child(self.path.clone()),
                    "contained unsorted or duplicate field ids",
                    location!(),
                ));
            }
        } else if self.fields.len() != self.column_indices.len() {
            return Err(Error::corrupt_file(
                base_path.child(self.path.clone()),
                "contained an unequal number of fields / column_indices",
                location!(),
            ));
        }
        Ok(())
    }
}

impl From<&DataFile> for pb::DataFile {
    fn from(df: &DataFile) -> Self {
        Self {
            path: df.path.clone(),
            fields: df.fields.clone(),
            column_indices: df.column_indices.clone(),
            file_major_version: df.file_major_version,
            file_minor_version: df.file_minor_version,
        }
    }
}

impl TryFrom<pb::DataFile> for DataFile {
    type Error = Error;

    fn try_from(proto: pb::DataFile) -> Result<Self> {
        Ok(Self {
            path: proto.path,
            fields: proto.fields,
            column_indices: proto.column_indices,
            file_major_version: proto.file_major_version,
            file_minor_version: proto.file_minor_version,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeletionFile {
    pub read_version: u64,
    pub id: u64,
    pub file_type: DeletionFileType,
    /// Number of deleted rows in this file. If None, this is unknown.
    pub num_deleted_rows: Option<usize>,
}

impl TryFrom<pb::DeletionFile> for DeletionFile {
    type Error = Error;

    fn try_from(value: pb::DeletionFile) -> Result<Self> {
        let file_type = match value.file_type {
            0 => DeletionFileType::Array,
            1 => DeletionFileType::Bitmap,
            _ => {
                return Err(Error::NotSupported {
                    source: "Unknown deletion file type".into(),
                    location: location!(),
                })
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
        })
    }
}

/// Data fragment.
///
/// A fragment is a set of files which represent the different columns of the same rows.
/// If column exists in the schema, but the related file does not exist, treat this column as `nulls`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fragment {
    /// Fragment ID
    pub id: u64,

    /// Files within the fragment.
    pub files: Vec<DataFile>,

    /// Optional file with deleted row ids.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deletion_file: Option<DeletionFile>,

    /// Original number of rows in the fragment. If this is None, then it is
    /// unknown. This is only optional for legacy reasons. All new tables should
    /// have this set.
    pub physical_rows: Option<usize>,
}

impl Fragment {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            files: vec![],
            deletion_file: None,
            physical_rows: None,
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
            files: vec![DataFile::new_legacy(path, schema)],
            deletion_file: None,
            physical_rows,
        }
    }

    pub fn add_file(
        &mut self,
        path: impl Into<String>,
        field_ids: Vec<i32>,
        column_indices: Vec<i32>,
    ) {
        self.files.push(DataFile::new(
            path,
            field_ids,
            column_indices,
            MAJOR_VERSION as u32,
            MINOR_VERSION_NEXT as u32,
        ));
    }

    /// Add a new [`DataFile`] to this fragment.
    pub fn add_file_legacy(&mut self, path: &str, schema: &Schema) {
        self.files.push(DataFile::new_legacy(path, schema));
    }

    // True if this fragment is made up of legacy v1 files, false otherwise
    pub fn has_legacy_files(&self) -> bool {
        // If any file in a fragment is legacy then all files in the fragment must be
        self.files[0].is_legacy_file()
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
                .collect::<Result<Vec<_>>>()?,
            deletion_file: p.deletion_file.map(DeletionFile::try_from).transpose()?,
            physical_rows,
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
            }
        });
        Self {
            id: f.id,
            files: f.files.iter().map(pb::DataFile::from).collect(),
            deletion_file,
            physical_rows: f.physical_rows.unwrap_or_default() as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{
        DataType, Field as ArrowField, Fields as ArrowFields, Schema as ArrowSchema,
    };
    use serde_json::{json, Value};

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
        });

        let proto = pb::DataFragment::from(&fragment);
        let fragment2 = Fragment::try_from(proto.clone()).unwrap();
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
        });

        let json = serde_json::to_string(&fragment).unwrap();

        let value: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            value,
            json!({
                "id": 123,
                "files":[
                    {"path": "foobar.lance", "fields": [0], "column_indices": [], "file_major_version": 0, "file_minor_version": 0}],
                     "deletion_file": {"read_version": 123, "id": 456, "file_type": "array",
                                       "num_deleted_rows": 10},
                "physical_rows": None::<usize>}),
        );

        let frag2 = Fragment::from_json(&json).unwrap();
        assert_eq!(fragment, frag2);
    }
}
