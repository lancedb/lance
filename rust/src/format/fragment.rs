use crate::format::pb;
use crate::format::pb::DataFragment;

/// Lance Data File
///
/// A data file is one piece of file storing data.
#[derive(Debug)]
pub struct DataFile {
    /// Relative path of the data file to dataset root.
    pub path: String,
    /// The Ids of fields in this file.
    pub fields: Vec<i32>,
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
#[derive(Debug)]
pub struct Fragment {
    /// Fragment ID
    pub id: u64,

    /// Files within the fragment.
    pub files: Vec<DataFile>,
}

impl Fragment {
    /// Get all field IDs from this datafragment
    pub fn field_ids(&self) -> Vec<i32> {
        vec![]
    }
}

impl From<&pb::DataFragment> for Fragment {
    fn from(p: &DataFragment) -> Self {
        Self {
            id: p.id,
            files: p.files.iter().map(DataFile::from).collect(),
        }
    }
}
