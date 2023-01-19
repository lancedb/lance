// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::collections::BTreeSet;

use crate::datatypes::Schema;
use crate::format::pb;

/// Lance Data File
///
/// A data file is one piece of file storing data.
#[derive(Debug, Clone, PartialEq)]
pub struct DataFile {
    /// Relative path of the data file to dataset root.
    pub path: String,
    /// The Ids of fields in this file.
    fields: Vec<i32>,
}

impl DataFile {
    pub fn new(path: &str, schema: &Schema) -> Self {
        Self {
            path: path.to_string(),
            fields: schema.field_ids(),
        }
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

/// Data fragment.
///
/// A fragment is a set of files which represent the different columns of the same rows.
/// If column exists in the schema, but the related file does not exist, treat this column as `nulls`.
#[derive(Debug, Clone)]
pub struct Fragment {
    /// Fragment ID
    pub id: u64,

    /// Files within the fragment.
    pub files: Vec<DataFile>,
}

impl Fragment {
    pub fn new(id: u64) -> Self {
        Self { id, files: vec![] }
    }

    /// Create a `Fragment` with one DataFile
    pub fn with_file(id: u64, path: &str, schema: &Schema) -> Self {
        Self {
            id,
            files: vec![DataFile::new(path, schema)],
        }
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
        }
    }
}

impl From<&Fragment> for pb::DataFragment {
    fn from(f: &Fragment) -> Self {
        Self {
            id: f.id,
            files: f.files.iter().map(pb::DataFile::from).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};

    #[test]
    fn test_new_fragment() {
        let path = "foobar.lance";

        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new(
                "s",
                DataType::Struct(vec![
                    ArrowField::new("si", DataType::Int32, false),
                    ArrowField::new("sb", DataType::Binary, true),
                ]),
                true,
            ),
            ArrowField::new("bool", DataType::Boolean, true),
        ]);
        let schema = Schema::try_from(&arrow_schema).unwrap();
        let fragment = Fragment::with_file(123, &path, &schema);

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
}
