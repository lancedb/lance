//! Lance Dataset
//!

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::cast::as_struct_array;
use arrow_array::{RecordBatch, RecordBatchReader, StructArray, UInt64Array};
use arrow_schema::Schema as ArrowSchema;
use arrow_select::{concat::concat_batches, take::take};
use chrono::prelude::*;
use futures::stream::{self, StreamExt, TryStreamExt};
use object_store::path::Path;
use uuid::Uuid;

pub mod scanner;
mod write;

use self::scanner::Scanner;
use crate::arrow::*;
use crate::datatypes::Schema;
use crate::format::{pb, Fragment, Index, Manifest};
use crate::index::vector::ivf::IvfPqIndexBuilder;
use crate::index::vector::VectorIndexParams;
use crate::index::{IndexBuilder, IndexParams, IndexType};
use crate::io::object_reader::read_message;
use crate::io::{object_reader::read_struct, read_metadata_offset, ObjectStore};
use crate::io::{read_manifest, write_manifest, FileReader, FileWriter};
use crate::{Error, Result};
pub use scanner::ROW_ID;
pub use write::*;

const LATEST_MANIFEST_NAME: &str = "_latest.manifest";
const VERSIONS_DIR: &str = "_versions";
const INDICES_DIR: &str = "_indices";
const DATA_DIR: &str = "data";

/// Lance Dataset
#[derive(Debug, Clone)]
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
    FileWriter::try_new(object_store, &full_path, schema).await
}

/// Get the manifest file path for a version.
fn manifest_path(base: &Path, version: u64) -> Path {
    base.child(VERSIONS_DIR)
        .child(format!("{version}.manifest"))
}

/// Get the latest manifest path
fn latest_manifest_path(base: &Path) -> Path {
    base.child(LATEST_MANIFEST_NAME)
}

impl Dataset {
    /// Open an existing dataset.
    pub async fn open(uri: &str) -> Result<Self> {
        let object_store = Arc::new(ObjectStore::new(uri)?);

        let base_path = object_store.base_path().clone();
        let latest_manifest_path = latest_manifest_path(&base_path);

        let object_reader = object_store.open(&latest_manifest_path).await?;
        let bytes = object_store
            .inner
            .get(&latest_manifest_path)
            .await?
            .bytes()
            .await?;
        let offset = read_metadata_offset(&bytes)?;
        let mut manifest: Manifest = read_struct(object_reader.as_ref(), offset).await?;
        manifest
            .schema
            .load_dictionary(object_reader.as_ref())
            .await?;
        Ok(Self {
            object_store,
            base: base_path,
            manifest: Arc::new(manifest),
        })
    }

    /// Create a new dataset with a stream of [RecordBatch]s.
    ///
    /// Returns the newly created [`Dataset`]. Returns [Error] if the dataset already exists.
    pub async fn create(
        batches: &mut Box<dyn RecordBatchReader>,
        uri: &str,
        params: Option<WriteParams>,
    ) -> Result<Self> {
        // 1. check the directory does not exist.
        let object_store = Arc::new(ObjectStore::new(uri)?);

        let latest_manifest_path = latest_manifest_path(object_store.base_path());
        match object_store.inner.head(&latest_manifest_path).await {
            Ok(_) => return Err(Error::IO(format!("Dataset already exists: {uri}"))),
            Err(object_store::Error::NotFound { path: _, source: _ }) => { /* we are good */ }
            Err(e) => return Err(Error::from(e)),
        }
        let params = params.unwrap_or_default();

        let mut peekable = batches.peekable();
        let mut schema: Schema;
        if let Some(batch) = peekable.peek() {
            if let Ok(b) = batch {
                schema = Schema::try_from(b.schema().as_ref())?;
                schema.set_dictionary(b)?;
            } else {
                return Err(Error::from(batch.as_ref().unwrap_err()));
            }
        } else {
            return Err(Error::IO(
                "Attempt to write empty record batches".to_string(),
            ));
        }

        let mut fragment_id = 0;
        let mut fragments: Vec<Fragment> = vec![];

        macro_rules! new_writer {
            () => {{
                let file_path = format!("{}.lance", Uuid::new_v4());
                let fragment = Fragment::with_file(fragment_id, &file_path, &schema);
                fragments.push(fragment);
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
                // TODO: the max rows per group boundary is not accurately calculated yet.
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
            // Drop the last writer.
            w.finish().await?;
            drop(writer);
        };

        let mut manifest = Manifest::new(&schema, Arc::new(fragments));
        write_manifest_file(&object_store, &mut manifest, None).await?;

        let base = object_store.base_path().clone();
        Ok(Self {
            object_store,
            base,
            manifest: Arc::new(manifest.clone()),
        })
    }

    /// Create a Scanner to scan the dataset.
    pub fn scan(&self) -> Scanner {
        Scanner::new(Arc::new(self.clone()))
    }

    /// Create indices on columns.
    ///
    /// Upon finish, a new dataset version is generated.
    ///
    /// Parameters:
    ///
    ///  - `columns`: the columns to build the indices on.
    ///  - `index_type`: specify [`IndexType`].
    ///  - `name`: optional index name. Must be unique in the dataset.
    ///            if not provided, it will auto-generate one.
    ///  - `params`: index parameters.
    pub async fn create_index(
        &self,
        columns: &[&str],
        index_type: IndexType,
        name: Option<String>,
        params: &dyn IndexParams,
    ) -> Result<Self> {
        if columns.len() != 1 {
            return Err(Error::Index(
                "Only support building index on 1 column at the moment".to_string(),
            ));
        }
        let column = columns[0];
        let Some(field) = self.schema().field(column) else {
            return Err(Error::Index(format!(
                "CreateIndex: column '{column}' does not exist"
            )));
        };

        // Load indices from the disk.
        let mut indices = self.load_indices().await?;

        let index_name = name.unwrap_or(format!("{column}_idx"));
        if indices.iter().any(|i| i.name == index_name) {
            return Err(Error::Index(format!(
                "Index name '{index_name} already exists'"
            )));
        }

        let index_id = Uuid::new_v4();
        match index_type {
            IndexType::Vector => {
                let vec_params = params
                    .as_any()
                    .downcast_ref::<VectorIndexParams>()
                    .ok_or_else(|| {
                        Error::Index("Vector index type must take a VectorIndexParams".to_string())
                    })?;

                let builder = IvfPqIndexBuilder::try_new(
                    self,
                    index_id,
                    &index_name,
                    column,
                    vec_params.num_partitions,
                    vec_params.num_sub_vectors,
                )?;
                builder.build().await?
            }
        }

        // Write index metadata down
        let new_idx = Index::new(index_id, &index_name, &[field.id]);
        indices.push(new_idx);

        let latest_manifest = self.latest_manifest().await?;
        let mut new_manifest = self.manifest.as_ref().clone();
        new_manifest.version = latest_manifest.version + 1;

        write_manifest_file(&self.object_store, &mut new_manifest, Some(indices)).await?;

        Ok(Self {
            object_store: self.object_store.clone(),
            base: self.base.clone(),
            manifest: Arc::new(new_manifest),
        })
    }

    /// Take rows by the internal ROW ids.
    pub(crate) async fn take_rows(
        &self,
        row_ids: &[u64],
        projection: &Schema,
    ) -> Result<RecordBatch> {
        let mut sorted_row_ids = Vec::from(row_ids);
        sorted_row_ids.sort();

        let mut row_ids_per_fragment: BTreeMap<u64, Vec<u32>> = BTreeMap::new();
        sorted_row_ids.iter().for_each(|row_id| {
            let fragment_id = row_id >> 32;
            let offset = (row_id - (fragment_id << 32)) as u32;
            row_ids_per_fragment
                .entry(fragment_id)
                .and_modify(|v| v.push(offset))
                .or_insert_with(|| vec![offset]);
        });

        let schema = Arc::new(ArrowSchema::from(projection));
        let object_store = &self.object_store;
        let batches = stream::iter(self.fragments().as_ref())
            .filter(|f| async { row_ids_per_fragment.contains_key(&f.id) })
            .then(|fragment| async {
                let path = self.data_dir().child(fragment.files[0].path.as_str());
                let mut reader = FileReader::try_new_with_fragment(
                    object_store,
                    &path,
                    fragment.id,
                    Some(self.manifest.as_ref()),
                )
                .await?;
                reader.set_projection(projection.clone());
                if let Some(indices) = row_ids_per_fragment.get(&fragment.id) {
                    reader.take(indices.as_slice()).await
                } else {
                    Ok(RecordBatch::new_empty(schema.clone()))
                }
            })
            .try_collect::<Vec<_>>()
            .await?;
        let one_batch = concat_batches(&schema, &batches)?;

        let original_indices: UInt64Array = row_ids
            .iter()
            .map(|o| {
                sorted_row_ids
                    .iter()
                    .position(|sorted_id| sorted_id == o)
                    .unwrap() as u64
            })
            .collect();
        let struct_arr: StructArray = one_batch.into();
        let reordered = take(&struct_arr, &original_indices, None)?;
        Ok(as_struct_array(&reordered).into())
    }

    pub(crate) fn object_store(&self) -> &ObjectStore {
        &self.object_store
    }

    fn versions_dir(&self) -> Path {
        self.base.child(VERSIONS_DIR)
    }

    fn manifest_file(&self, version: u64) -> Path {
        self.versions_dir().child(format!("{version}.manifest"))
    }

    fn latest_manifest_path(&self) -> Path {
        latest_manifest_path(&self.base)
    }

    async fn latest_manifest(&self) -> Result<Manifest> {
        read_manifest(&self.object_store, &self.latest_manifest_path()).await
    }

    fn data_dir(&self) -> Path {
        self.base.child(DATA_DIR)
    }

    pub(crate) fn indices_dir(&self) -> Path {
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

    pub fn fragments(&self) -> &Arc<Vec<Fragment>> {
        &self.manifest.fragments
    }

    /// Read all indices of this Dataset version.
    pub async fn load_indices(&self) -> Result<Vec<Index>> {
        if let Some(pos) = self.manifest.index_section.as_ref() {
            let manifest_file = self.manifest_file(self.version().version);

            let reader = self.object_store.open(&manifest_file).await?;
            let section: pb::IndexSection = read_message(reader.as_ref(), *pos).await?;

            Ok(section
                .indices
                .iter()
                .map(|pb| Index::try_from(pb))
                .collect::<Result<Vec<_>>>()?)
        } else {
            Ok(vec![])
        }
    }
}

/// Finish writing the manifest file, and commit the changes by linking the latest manifest file
/// to this version.
async fn write_manifest_file(
    object_store: &ObjectStore,
    manifest: &mut Manifest,
    indices: Option<Vec<Index>>,
) -> Result<()> {
    let path = manifest_path(object_store.base_path(), manifest.version);
    let mut object_writer = object_store.create(&path).await?;
    let pos = write_manifest(&mut object_writer, manifest, indices).await?;
    object_writer.write_magics(pos).await?;
    object_writer.shutdown().await?;

    // Link it to latest manifest, and COMMIT.
    let latest_manifest = latest_manifest_path(object_store.base_path());
    object_store.inner.copy(&path, &latest_manifest).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::testing::generate_random_array;

    use arrow_array::{
        cast::as_struct_array, DictionaryArray, FixedSizeListArray, Int32Array, RecordBatch,
        StringArray, UInt16Array,
    };
    use arrow_ord::sort::sort_to_indices;
    use arrow_schema::{DataType, Field, Schema};
    use arrow_select::take::take;
    use futures::stream::TryStreamExt;

    use tempfile::tempdir;

    #[tokio::test]
    async fn create_dataset() {
        let test_dir = tempdir().unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new(
                "dict",
                DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
                false,
            ),
        ]));
        let dict_values = StringArray::from_iter_values(["a", "b", "c", "d", "e"]);
        let batches = RecordBatchBuffer::new(
            (0..20)
                .map(|i| {
                    RecordBatch::try_new(
                        schema.clone(),
                        vec![
                            Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20)),
                            Arc::new(
                                DictionaryArray::try_new(
                                    &UInt16Array::from_iter_values((0_u16..20_u16).map(|v| v % 5)),
                                    &dict_values,
                                )
                                .unwrap(),
                            ),
                        ],
                    )
                    .unwrap()
                })
                .collect(),
        );
        let expected_batches = batches.batches.clone();

        let test_uri = test_dir.path().to_str().unwrap();

        let mut write_params = WriteParams::default();
        write_params.max_rows_per_file = 40;
        write_params.max_rows_per_group = 10;
        let mut reader: Box<dyn RecordBatchReader> = Box::new(batches);
        Dataset::create(&mut reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let actual_ds = Dataset::open(test_uri).await.unwrap();
        assert_eq!(actual_ds.version().version, 1);
        let actual_schema = Schema::from(actual_ds.schema());
        assert_eq!(&actual_schema, schema.as_ref());

        let actual_batches = actual_ds
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        // sort
        let actual_batch = concat_batches(&schema, &actual_batches).unwrap();
        let idx_arr = actual_batch.column_by_name("i").unwrap();
        let sorted_indices = sort_to_indices(idx_arr, None, None).unwrap();
        let struct_arr: StructArray = actual_batch.into();
        let sorted_arr = take(&struct_arr, &sorted_indices, None).unwrap();

        let expected_struct_arr: StructArray =
            concat_batches(&schema, &expected_batches).unwrap().into();
        assert_eq!(&expected_struct_arr, as_struct_array(sorted_arr.as_ref()));

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

    #[ignore]
    #[tokio::test]
    async fn test_create_index() {
        let test_dir = tempdir().unwrap();

        let dimension = 32;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "embeddings",
            DataType::FixedSizeList(
                Box::new(Field::new("item", DataType::Float32, true)),
                dimension,
            ),
            false,
        )]));

        let float_arr = generate_random_array(100 * dimension as usize);
        let vectors = Arc::new(FixedSizeListArray::try_new(float_arr, dimension).unwrap());
        let batches =
            RecordBatchBuffer::new(vec![
                RecordBatch::try_new(schema.clone(), vec![vectors]).unwrap()
            ]);

        let test_uri = test_dir.path().to_str().unwrap();

        let mut reader: Box<dyn RecordBatchReader> = Box::new(batches);
        let dataset = Dataset::create(&mut reader, test_uri, None).await.unwrap();

        let params = VectorIndexParams::default();
        dataset
            .create_index(&["embeddings"], IndexType::Vector, None, &params)
            .await
            .unwrap();
    }
}
