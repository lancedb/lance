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

use std::collections::BTreeSet;
use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::datatypes::Schema;
use crate::error::Result;
use crate::format::pb;

/// Lance Data File
///
/// A data file is one piece of file storing data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataFile {
    /// Relative path of the data file to dataset root.
    pub path: String,
    /// The Ids of fields in this file.
    pub fields: Vec<i32>,
}

impl DataFile {
    pub(crate) fn new(path: &str, schema: &Schema) -> Self {
        Self {
            path: path.to_string(),
            fields: schema.field_ids(),
        }
    }

    pub fn schema(&self, full_schema: &Schema) -> Schema {
        full_schema.project_by_ids(&self.fields)
    }
}

impl From<&DataFile> for pb::DataFile {
    fn from(df: &DataFile) -> Self {
        Self {
            path: df.path.clone(),
            fields: df.fields.clone(),
        }
    }
}

impl From<&pb::DataFile> for DataFile {
    fn from(proto: &pb::DataFile) -> Self {
        Self {
            path: proto.path.clone(),
            fields: proto.fields.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeletionFileType {
    Array,
    Bitmap,
}

impl DeletionFileType {
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
    /// Number of deleted rows in this file. If 0, this is unknown.
    pub num_deleted_rows: usize,
}

// TODO: should we convert this to TryFrom and surface the error?
#[allow(clippy::fallible_impl_from)]
impl From<&pb::DeletionFile> for DeletionFile {
    fn from(value: &pb::DeletionFile) -> Self {
        let file_type = match value.file_type {
            0 => DeletionFileType::Array,
            1 => DeletionFileType::Bitmap,
            _ => panic!("Invalid deletion file type"),
        };
        Self {
            read_version: value.read_version,
            id: value.id,
            file_type,
            num_deleted_rows: value.num_deleted_rows as usize,
        }
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

    /// Original number of rows in the fragment. If this is zero, then it is
    /// unknown.
    pub physical_rows: usize,
}

impl Fragment {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            files: vec![],
            deletion_file: None,
            physical_rows: 0,
        }
    }

    pub fn num_rows(&self) -> Option<usize> {
        match (self.physical_rows, &self.deletion_file) {
            // Unknown fragment length
            (0, _) => None,
            // Known fragment length, no deletion file.
            (len, None) => Some(len),
            // Known fragment length, but don't know deletion file size.
            (_, Some(deletion_file)) if deletion_file.num_deleted_rows == 0 => None,
            (len, Some(deletion_file)) => Some(len - deletion_file.num_deleted_rows),
        }
    }

    pub fn from_json(json: &str) -> Result<Self> {
        let fragment: Self = serde_json::from_str(json)?;
        Ok(fragment)
    }

    /// Create a `Fragment` with one DataFile
    pub fn with_file(id: u64, path: &str, schema: &Schema, physical_rows: usize) -> Self {
        Self {
            id,
            files: vec![DataFile::new(path, schema)],
            deletion_file: None,
            physical_rows,
        }
    }

    /// Add a new [`DataFile`] to this fragment.
    pub fn add_file(&mut self, path: &str, schema: &Schema) {
        self.files.push(DataFile::new(path, schema));
    }

    /// Get all field IDs from this fragment, sorted.
    pub fn field_ids(&self) -> Vec<i32> {
        BTreeSet::from_iter(self.files.iter().flat_map(|f| f.fields.clone()))
            .into_iter()
            .collect()
    }
}

impl From<&pb::DataFragment> for Fragment {
    fn from(p: &pb::DataFragment) -> Self {
        Self {
            id: p.id,
            files: p.files.iter().map(DataFile::from).collect(),
            deletion_file: p.deletion_file.as_ref().map(DeletionFile::from),
            physical_rows: p.physical_rows as usize,
        }
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
                num_deleted_rows: f.num_deleted_rows as u64,
            }
        });
        Self {
            id: f.id,
            files: f.files.iter().map(pb::DataFile::from).collect(),
            deletion_file,
            physical_rows: f.physical_rows as u64,
        }
    }
}

pub struct RowAddress(u64);

impl RowAddress {
    pub const FRAGMENT_SIZE: u64 = 1 << 32;
    // A fragment id that will never be used
    pub const TOMBSTONE_FRAG: u32 = 0xffffffff;
    // A row id that will never be used
    pub const TOMBSTONE_ROW: u64 = 0xffffffffffffffff;

    pub fn new_from_id(row_id: u64) -> Self {
        Self(row_id)
    }

    pub fn new_from_parts(fragment_id: u32, row_id: u32) -> Self {
        Self(((fragment_id as u64) << 32) | row_id as u64)
    }

    pub fn first_row(fragment_id: u32) -> Self {
        Self::new_from_parts(fragment_id, 0)
    }

    pub fn fragment_range(fragment_id: u32) -> Range<u64> {
        u64::from(Self::first_row(fragment_id))..u64::from(Self::first_row(fragment_id + 1))
    }

    pub fn fragment_id(&self) -> u32 {
        (self.0 >> 32) as u32
    }

    pub fn row_id(&self) -> u32 {
        self.0 as u32
    }
}

impl From<RowAddress> for u64 {
    fn from(row_id: RowAddress) -> Self {
        row_id.0
    }
}

impl std::fmt::Debug for RowAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self) // use Display
    }
}

impl std::fmt::Display for RowAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.fragment_id(), self.row_id())
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
        let fragment = Fragment::with_file(123, path, &schema, 10);

        assert_eq!(123, fragment.id);
        assert_eq!(fragment.field_ids(), [0, 1, 2, 3]);
        assert_eq!(
            fragment.files,
            vec![DataFile {
                path: path.to_string(),
                fields: vec![0, 1, 2, 3]
            }]
        )
    }

    #[test]
    fn test_roundtrip_fragment() {
        let mut fragment = Fragment::new(123);
        let schema = ArrowSchema::new(vec![ArrowField::new("x", DataType::Float16, true)]);
        fragment.add_file("foobar.lance", &Schema::try_from(&schema).unwrap());
        fragment.deletion_file = Some(DeletionFile {
            read_version: 123,
            id: 456,
            file_type: DeletionFileType::Array,
            num_deleted_rows: 10,
        });

        let proto = pb::DataFragment::from(&fragment);
        let fragment2 = Fragment::from(&proto);
        assert_eq!(fragment, fragment2);

        fragment.deletion_file = None;
        let proto = pb::DataFragment::from(&fragment);
        let fragment2 = Fragment::from(&proto);
        assert_eq!(fragment, fragment2);
    }

    #[test]
    fn test_to_json() {
        let mut fragment = Fragment::new(123);
        let schema = ArrowSchema::new(vec![ArrowField::new("x", DataType::Float16, true)]);
        fragment.add_file("foobar.lance", &Schema::try_from(&schema).unwrap());
        fragment.deletion_file = Some(DeletionFile {
            read_version: 123,
            id: 456,
            file_type: DeletionFileType::Array,
            num_deleted_rows: 10,
        });

        let json = serde_json::to_string(&fragment).unwrap();

        let value: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            value,
            json!({
                "id": 123,
                "files":[
                    {"path": "foobar.lance", "fields": [0]}],
                     "deletion_file": {"read_version": 123, "id": 456, "file_type": "array",
                                       "num_deleted_rows": 10},
                "physical_rows": 0}),
        );

        let frag2 = Fragment::from_json(&json).unwrap();
        assert_eq!(fragment, frag2);
    }
}
