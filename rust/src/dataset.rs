//! Lance Dataset
//!

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::RecordBatchReader;
use chrono::prelude::*;
use object_store::path::Path;
use uuid::Uuid;

pub mod scanner;
mod write;

use self::scanner::Scanner;
use crate::arrow::*;
use crate::datatypes::Schema;
use crate::format::{Fragment, Manifest};
use crate::io::{read_manifest, write_manifest, FileWriter};
use crate::io::{read_metadata_offset, ObjectStore};
use crate::{Error, Result};
pub use write::*;

const LATEST_MANIFEST_NAME: &str = "_latest.manifest";
const VERSIONS_DIR: &str = "_versions";
const INDICES_DIR: &str = "_indices";
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

/// Create a new [FileWriter] with the related `data_file_path` under `<DATA_DIR>`.
async fn new_file_writer<'a>(
    object_store: &'a ObjectStore,
    data_file_path: &str,
    schema: &'a Schema,
) -> Result<FileWriter<'a>> {
    let full_path = object_store
        .base_path()
        .child(DATA_DIR)
        .child(data_file_path);
    Ok(FileWriter::try_new(&object_store, &full_path, schema).await?)
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

    /// Create a new dataset with RecordBatch.
    ///
    /// Returns the newly created dataset.
    ///
    /// Returns [Error] if the dataset already exists.
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
        let mut fragment_id = 0;

        macro_rules! new_writer {
            () => {{
                let file_path = format!("{}.lance", Uuid::new_v4());
                let fragment = Fragment::with_file(fragment_id, &file_path, &schema);
                manifest.fragments.push(fragment);
                fragment_id += 1;
                Some(new_file_writer(&object_store, &file_path, &schema).await?)
            }};
        }

        let mut writer = None;
        let mut buffer = RecordBatchBuffer::empty();
        for batch_result in peekable {
            let batch = batch_result?;
            buffer.batches.push(batch);
            if buffer.num_rows() >= params.max_rows_per_group {
                // TODO: the max rows per group boundry is not accurately calculated yet.
                if writer.is_none() {
                    writer = new_writer!();
                };
                writer.as_mut().unwrap().write(&buffer.finish()?).await?;
                buffer = RecordBatchBuffer::empty();
            }
            if let Some(w) = writer.as_mut() {
                if w.len() >= params.max_rows_per_file {
                    w.finish().await?;
                    writer = None;
                }
            }
        }
        if buffer.num_rows() > 0 {
            if writer.is_none() {
                writer = new_writer!();
            };
            writer.as_mut().unwrap().write(&buffer.finish()?).await?;
        }
        if let Some(w) = writer.as_mut() {
            w.finish().await?;
        };
        drop(writer);

        let manifest_file_path = object_store
            .base_path()
            .child(VERSIONS_DIR)
            .child(format!("{}.manifest", manifest.version));
        {
            let mut object_writer = object_store.create(&manifest_file_path).await?;
            let pos = write_manifest(&mut object_writer, &mut manifest).await?;
            object_writer.write_magics(pos).await?;
            object_writer.shutdown().await?;
        }
        let latest_manifest = object_store.base_path().child(LATEST_MANIFEST_NAME);
        object_store
            .inner
            .copy(&manifest_file_path, &latest_manifest)
            .await?;

        let base = object_store.base_path().clone();
        Ok(Dataset {
            object_store,
            base,
            manifest: Arc::new(manifest.clone()),
        })
    }

    /// Create a Scanner to scan the dataset.
    pub fn scan(&self) -> Scanner {
        Scanner::new(&self)
    }

    fn versions_dir(&self) -> Path {
        self.base.child(VERSIONS_DIR)
    }

    fn data_dir(&self) -> Path {
        self.base.child(DATA_DIR)
    }

    fn indices_dir(&self) -> Path {
        self.base.child(INDICES_DIR)
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

    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use futures::stream::TryStreamExt;

    use tempfile::tempdir;

    #[tokio::test]
    async fn create_dataset() {
        let test_dir = tempdir().unwrap();

        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, false)]));
        let mut batches = RecordBatchBuffer::new(
            (0..20)
                .map(|i| {
                    RecordBatch::try_new(
                        schema.clone(),
                        vec![Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20))],
                    )
                    .unwrap()
                })
                .collect(),
        );

        let test_uri = test_dir.path().to_str().unwrap();

        let mut write_params = WriteParams::default();
        write_params.max_rows_per_file = 40;
        write_params.max_rows_per_group = 10;
        Dataset::create(&mut batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let actual_ds = Dataset::open(test_uri).await.unwrap();
        assert_eq!(actual_ds.version().version, 1);
        let actual_schema = Schema::from(actual_ds.schema());
        assert_eq!(&actual_schema, schema.as_ref());

        let actual_batches = actual_ds
            .scan()
            .into_stream()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(batches.batches, actual_batches);

        // Each fragments has different fragment ID
        assert_eq!(
            actual_ds
                .fragments()
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            (0..10).collect::<Vec<_>>()
        )
    }
}
