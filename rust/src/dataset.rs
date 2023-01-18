//! Lance Dataset
//!

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchReader};
use chrono::prelude::*;
use futures::stream::{self, StreamExt, TryStreamExt};
use object_store::path::Path;
use uuid::Uuid;

pub mod scanner;
mod write;

use self::scanner::Scanner;
use crate::arrow::*;
use crate::datatypes::Schema;
use crate::format::{Fragment, Manifest};
use crate::io::{read_manifest, FileWriter};
use crate::io::{read_metadata_offset, ObjectStore};
use crate::{Error, Result};
pub use write::*;

const LATEST_MANIFEST_NAME: &str = "_latest.manifest";
const VERSIONS_DIR: &str = "_versions";
const DATA_DIR: &str = "data";

fn latest_manifest_path(base: &Path) -> Path {
    base.child(LATEST_MANIFEST_NAME)
}

/// Lance Dataset
#[derive(Debug)]
pub struct Dataset {
    object_store: Arc<ObjectStore>,
    base: Path,
    manifest: Arc<Manifest>,
}

/// Dataset Version
pub struct Version {
    /// version number
    pub version: u64,

    /// Timestamp of dataset creation in UTC.
    pub timestamp: DateTime<Utc>,

    /// Key-value pairs of metadata.
    pub metadata: BTreeMap<String, String>,
}

/// Convert Manifest to Data Version.
impl From<&Manifest> for Version {
    fn from(m: &Manifest) -> Self {
        Self {
            version: m.version,
            timestamp: Utc::now(),
            metadata: BTreeMap::default(),
        }
    }
}

impl Dataset {
    /// Open an existing dataset.
    pub async fn open(uri: &str) -> Result<Self> {
        let object_store = Arc::new(ObjectStore::new(uri)?);

        let latest_manifest_path = latest_manifest_path(object_store.base_path());

        let mut object_reader = object_store.open(&latest_manifest_path).await?;
        let bytes = object_store
            .inner
            .get(&latest_manifest_path)
            .await?
            .bytes()
            .await?;
        let offset = read_metadata_offset(&bytes)?;
        let mut manifest: Manifest = object_reader.read_struct(offset).await?;
        manifest.schema.load_dictionary(&object_reader).await?;
        Ok(Self {
            object_store,
            base: Path::from(uri),
            manifest: Arc::new(manifest),
        })
    }

    /// Create a new dataset with [RecordBatch]s.
    pub async fn create(
        batches: &mut dyn RecordBatchReader,
        uri: &str,
        params: Option<WriteParams>,
    ) -> Result<Self> {
        // 1. check the directory does not exist.
        let object_store = Arc::new(ObjectStore::new(uri)?);

        let latest_manifest_path = latest_manifest_path(object_store.base_path());
        match object_store.inner.head(&latest_manifest_path).await {
            Ok(_) => return Err(Error::IO(format!("Dataset already exists: {}", uri))),
            Err(object_store::Error::NotFound { path: _, source: _ }) => { /* we are good */ }
            Err(e) => return Err(Error::from(e)),
        }

        let params = params.unwrap_or_default();


        let mut peekable = batches.peekable();
        let schema: Schema;
        if let Some(batch) = peekable.peek() {
            if let Ok(b) = batch {
                schema = Schema::try_from(b.schema().as_ref())?;
            } else {
                return Err(Error::from(batch.as_ref().unwrap_err()));
            }
        } else {
            return Err(Error::IO(
                "Attempt to write empty record batches".to_string(),
            ));
        }

        let mut manifest = Manifest::new(&schema);

        let mut fragment = 
        let file_path = object_store
            .base_path()
            .child(DATA_DIR)
            .child(format!("{}.lance", Uuid::new_v4()));
        println!("Create file path: {}", file_path);
        let mut buffer = RecordBatchBuffer::empty();

        let object_writer = object_store.create(&file_path).await?;
        let mut file_writer = FileWriter::new(object_writer, &schema);
        for batch_result in peekable {
            let batch = batch_result?;
            buffer.batches.push(batch);
            if buffer.num_rows() >= params.max_rows_per_group {
                file_writer.write(&buffer.finish()?).await?;
                buffer = RecordBatchBuffer::empty();
            }
        }
        drop(file_writer);


        todo!()
    }

    /// Create a Scanner to scan the dataset.
    pub fn scan(&self) -> Scanner {
        Scanner::new(&self)
    }

    pub(crate) fn object_store(&self) -> &ObjectStore {
        &self.object_store
    }

    fn versions_dir(&self) -> Path {
        self.base.child(VERSIONS_DIR)
    }

    fn data_dir(&self) -> Path {
        self.base.child(DATA_DIR)
    }

    pub fn version(&self) -> Version {
        Version::from(self.manifest.as_ref())
    }

    /// Get all versions.
    pub async fn versions(&self) -> Result<Vec<Version>> {
        let paths: Vec<Path> = self
            .object_store
            .inner
            .list_with_delimiter(Some(&self.versions_dir()))
            .await?
            .objects
            .iter()
            .filter(|&obj| obj.location.as_ref().ends_with(".manifest"))
            .map(|o| o.location.clone())
            .collect();
        let mut versions = vec![];
        for path in paths.iter() {
            let manifest = read_manifest(&self.object_store, path).await?;
            versions.push(Version::from(&manifest));
        }
        Ok(versions)
    }

    pub fn schema(&self) -> &Schema {
        &self.manifest.schema
    }

    pub fn fragments(&self) -> &[Fragment] {
        &self.manifest.fragments
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Field, Schema};

    use tempfile::tempdir;

    #[tokio::test]
    async fn create_dataset() {
        let test_dir = tempdir().unwrap();

        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, false)]));
        let mut batches = RecordBatchBuffer::new(vec![RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from_iter_values(1..10))],
        )
        .unwrap()]);

        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = Dataset::create(&mut batches, test_uri, None).await.unwrap();
    }
}
