// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::{ops::Range, sync::Arc};

use arrow::{
    array::{AsArray, StringBuilder},
    datatypes::UInt64Type,
};
use arrow_array::{
    Array, ArrayRef, LargeListArray, ListArray, RecordBatch, StructArray, UInt64Array,
};
use arrow_schema::{DataType, Field as ArrowField, FieldRef, Fields, Schema as ArrowSchema};
use bytes::Bytes;
use futures::{stream, FutureExt, StreamExt, TryStreamExt};
use lance_core::{Error, Result};
use lance_io::{
    object_store::ObjectStore,
    traits::{Reader, Writer},
};
use lance_table::format::BlobFile;
use object_store::path::Path;
use snafu::{location, Location};
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

use crate::Dataset;

#[derive(Debug)]
pub struct Blob {
    reader: Arc<dyn Reader>,
    position: u64,
    size: u64,
}

impl Blob {
    /// Returns the size of the blob in bytes
    ///
    /// Note that there is also Reader::size but this is sync
    pub fn size_bytes(&self) -> u64 {
        self.size
    }
}

#[async_trait::async_trait]
impl Reader for Blob {
    fn path(&self) -> &Path {
        // We could return self.reader.path() but that seems misleading
        // since this is only a slice of that path
        unimplemented!("Blob::path()")
    }

    /// Suggest optimal I/O size per storage device.
    fn block_size(&self) -> usize {
        self.reader.block_size()
    }

    /// Object/File Size.
    async fn size(&self) -> Result<usize> {
        Ok(self.size as usize)
    }

    /// Read a range of bytes from the object.
    async fn get_range(&self, range: Range<usize>) -> Result<Bytes> {
        if range.end > self.size as usize {
            return Err(Error::InvalidInput {
                source: format!("range {:?} exceeds object size {}", range, self.size).into(),
                location: location!(),
            });
        }

        let start = self.position as usize + range.start;
        let end = self.position as usize + range.end;
        let data = self.reader.get_range(start..end).await?;
        Ok(data)
    }
}

#[async_trait::async_trait]
pub trait BlobExt {
    /// Given an array of blob descriptions, open blobs and return them.
    ///
    /// The returned blobs can be used to access the blob data
    ///
    /// A returned blob will be None if the underlying blob was null
    async fn open_blobs(&self, blobs: &dyn Array) -> Result<Vec<Option<Blob>>>;
}

fn check_blob_arr(arr: &dyn Array) -> Result<()> {
    // Check that the array is a struct array with the expected fields
    // and types.
    if let DataType::Struct(fields) = arr.data_type() {
        let expected_fields = ["path", "position", "size"];
        let expected_types = [DataType::Utf8, DataType::UInt64, DataType::UInt64];
        if fields.len() != expected_fields.len() {
            return Err(Error::InvalidInput {
                source: format!(
                    "open_blobs function expected struct array with {} fields but received struct array with {} fields",
                    expected_fields.len(),
                    fields.len()
                )
                .into(),
                location: location!(),
            });
        }
        for (field, expected_field) in fields.iter().zip(expected_fields.iter()) {
            if field.name() != expected_field {
                return Err(Error::InvalidInput {
                    source: format!(
                        "open_blobs function expected struct array with field named '{}' but received struct array with field named '{}'",
                        expected_field,
                        field.name()
                    )
                    .into(),
                    location: location!(),
                });
            }
        }
        for (field, expected_type) in fields.iter().zip(expected_types.iter()) {
            if field.data_type() != expected_type {
                return Err(Error::InvalidInput {
                    source: format!(
                        "open_blobs function expected struct array with field '{}' of type {:?} but received struct array with field of type {:?}",
                        field.name(),
                        expected_type,
                        field.data_type()
                    )
                    .into(),
                    location: location!(),
                });
            }
        }
    } else {
        return Err(Error::InvalidInput {
            source: format!(
                "open_blobs function expected struct array but received array of type {:?}",
                arr.data_type()
            )
            .into(),
            location: location!(),
        });
    }
    Ok(())
}

#[async_trait::async_trait]
impl BlobExt for Dataset {
    async fn open_blobs(&self, blobs: &dyn Array) -> Result<Vec<Option<Blob>>> {
        // Simple impl for today, open each blob individually.  In the future
        // if using local storage, we can coalesce blobs in the same file to
        // share file descriptor.  Not much advantage for remote storage.
        check_blob_arr(blobs)?;
        let descriptions = blobs.as_struct();
        let paths = descriptions.column(0).as_string::<i32>();
        let positions = descriptions.column(1).as_primitive::<UInt64Type>();
        let sizes = descriptions.column(2).as_primitive::<UInt64Type>();
        let reader_futs = paths
            .iter()
            .zip(positions.values().iter().zip(sizes.values()))
            .map(|(path, (position, size))| {
                if let Some(path) = path {
                    let path = self.blob_dir().child(path);
                    let position = *position;
                    let size = *size;
                    Ok(async move {
                        let reader = self.object_store.open(&path).await?;
                        Result::Ok(Some(Blob {
                            reader: reader.into(),
                            position,
                            size,
                        }))
                    }
                    .boxed())
                } else {
                    Ok(std::future::ready(Ok(None)).boxed())
                }
            })
            .collect::<Result<Vec<_>>>()?;
        stream::iter(reader_futs)
            .buffered(num_cpus::get())
            .try_collect()
            .await
    }
}

/// A blob writer writes blob data to the dataset
pub struct BlobWriter {
    object_store: Arc<ObjectStore>,
    basedir: Path,
    target_size_bytes: u64,
    blob_files_created: Vec<BlobFile>,
    current_writer: Option<Box<dyn Writer>>,
    current_size_bytes: u64,
    current_num_items: u64,
    current_path: String,
}

impl BlobWriter {
    /// Create a new blob writer for the dataset
    ///
    /// The target_size_bytes will be the size of each blob file created
    /// though this may be exceeded in some cases
    pub fn new(object_store: Arc<ObjectStore>, basedir: Path, target_size_bytes: u64) -> Self {
        Self {
            object_store,
            basedir,
            target_size_bytes,
            blob_files_created: Vec::new(),
            current_writer: None,
            current_size_bytes: 0,
            current_num_items: 0,
            current_path: String::default(),
        }
    }

    async fn get_current_writer(&mut self) -> Result<&mut Box<dyn Writer>> {
        if self.current_writer.is_none() {
            let uuid = Uuid::new_v4();
            let path = self.basedir.child(uuid.to_string());
            let writer = self.object_store.create(&path).await?;
            self.current_writer = Some(Box::new(writer));
            self.current_path = uuid.to_string();
        }
        Ok(self.current_writer.as_mut().unwrap())
    }

    async fn do_flush(&mut self) -> Result<()> {
        let path = self.current_path.clone();
        assert!(self.current_num_items < u32::MAX as u64);
        self.blob_files_created.push(BlobFile {
            path,
            num_items: self.current_num_items as u32,
        });
        self.current_size_bytes = 0;
        self.current_num_items = 0;
        self.current_path = String::default();
        let mut writer = self.current_writer.take().unwrap();
        writer.shutdown().await?;
        Ok(())
    }

    async fn flush_if_needed(&mut self) -> Result<()> {
        if self.current_size_bytes >= self.target_size_bytes {
            self.do_flush().await?;
        }
        Ok(())
    }

    /// Close any outstanding blob files and return the list of blob files created so far
    pub async fn flush(&mut self) -> Result<Vec<BlobFile>> {
        if self.current_size_bytes > 0 {
            self.do_flush().await?;
        }
        Ok(std::mem::take(&mut self.blob_files_created))
    }

    async fn write_blob_array(
        &mut self,
        field_name: &str,
        array: &dyn Array,
    ) -> Result<(FieldRef, ArrayRef)> {
        let bin_array = array.as_binary::<i64>();
        let position = self.current_size_bytes;
        let writer = self.get_current_writer().await?;
        writer.write_all(bin_array.value_data()).await?;
        self.current_size_bytes += bin_array.value_data().len() as u64;
        let sizes = bin_array
            .offsets()
            .windows(2)
            .map(|w| (w[1] - w[0]) as u64)
            .collect::<Vec<_>>();
        let mut pos = position;
        let positions = sizes
            .iter()
            .map(|l| {
                let p = pos;
                pos = p + l;
                p
            })
            .collect::<Vec<_>>();
        debug_assert_eq!(
            bin_array.value_data().len() as u64,
            sizes.iter().sum::<u64>()
        );
        let mut paths = StringBuilder::with_capacity(
            positions.len(),
            positions.len() * self.current_path.len(),
        );
        for _ in 0..positions.len() {
            paths.append_value(&self.current_path);
        }
        let paths = Arc::new(paths.finish());
        let sizes = Arc::new(UInt64Array::from(sizes)) as ArrayRef;
        let positions = Arc::new(UInt64Array::from(positions)) as ArrayRef;
        let path_field = ArrowField::new("path", DataType::Utf8, true);
        let size_field = ArrowField::new("size", DataType::UInt64, true);
        let position_field = ArrowField::new("position", DataType::UInt64, true);
        let descriptions = StructArray::new(
            Fields::from(vec![path_field, position_field, size_field]),
            vec![paths, positions, sizes],
            array.nulls().cloned(),
        );
        let descriptions = Arc::new(descriptions) as ArrayRef;
        let descriptions_field = Arc::new(ArrowField::new(
            field_name,
            descriptions.data_type().clone(),
            true,
        ));
        Ok((descriptions_field, descriptions))
    }

    /// Extract blobs from a record batch and write them into the object store.
    ///
    /// The arrays that used to be blobs will be replaced by a struct array containing the
    /// path, position, and size of the blobs.
    ///
    /// Blob files will be written to try and align with the target size.  If the target size
    /// is exceeded, a new blob file will be created.  Currently we do not split arrays.  This
    /// means a very large array could cause the target size to be exceeded.
    pub async fn write_blobs(&mut self, batch: RecordBatch) -> Result<RecordBatch> {
        struct WriteBlobsTransform<'a> {
            writer: &'a mut BlobWriter,
        }
        #[async_trait::async_trait]
        impl<'a> ArrayTransformer for WriteBlobsTransform<'a> {
            async fn transform(
                &mut self,
                array: &dyn Array,
                field: &ArrowField,
            ) -> Result<Option<(FieldRef, ArrayRef)>> {
                if *array.data_type() == DataType::LargeBinary {
                    let (field, array) = self.writer.write_blob_array(field.name(), array).await?;
                    Ok(Some((field, array)))
                } else {
                    Ok(None)
                }
            }
        }
        let mut transformer = WriteBlobsTransform { writer: self };
        let batch = transform_batch(&batch, &mut transformer).await?;
        self.flush_if_needed().await?;
        batch.ok_or_else(|| Error::InvalidInput {
            source: "Cannot write batch.  Expected one or more blob columns but didn't find any"
                .into(),
            location: location!(),
        })
    }
}

// These transformation utilities could exist in a common file at some point but
// they are only used for blob writing at the moment so keeping them here until
// they are needed elsewhere.

/// A transform function to be used as part of
/// [`transform_array`] or [`transform_batch`].\
#[async_trait::async_trait]
pub trait ArrayTransformer: Send {
    /// Potentially transform an array
    ///
    /// Return None to indicate the array was not transformed or return
    /// a new field and array to indicate the array was transformed
    async fn transform(
        &mut self,
        array: &dyn Array,
        field: &ArrowField,
    ) -> Result<Option<(FieldRef, ArrayRef)>>;
}

/// Applies a transformation to an array.  If the array is a nested array
/// (struct or list) then it will recursively apply the transformation to
/// the children
///
/// The transformation is not applied to transformed arrays, even if they
/// are nested.  First the transformation will be applied to an array.  If
/// the transform succeeds then the result is used.  If it does not succeed
/// then it attempts to transform the array's children.
///
/// First we attempt to apply the transformation to the array and then, if
/// that fails, we try and apply the transformation to the array's chilren
/// and then, if no child was modified, we return None
#[async_recursion::async_recursion]
pub async fn transform_array(
    arr: &dyn Array,
    field: &ArrowField,
    transformer: &mut dyn ArrayTransformer,
) -> Result<Option<(FieldRef, ArrayRef)>> {
    match arr.data_type() {
        DataType::Struct(fields) => {
            let struct_arr = arr.as_struct();
            let mut new_columns = Vec::new();
            let mut new_fields = Vec::new();
            let mut has_new = false;
            for (col, child_field) in struct_arr.columns().iter().zip(fields) {
                if let Some((transformed_field, transformed_arr)) =
                    transformer.transform(col, child_field.as_ref()).await?
                {
                    has_new = true;
                    new_fields.push(transformed_field);
                    new_columns.push(transformed_arr);
                } else if let Some((transformed_field, transitive_arr)) =
                    transform_array(&col, child_field, transformer).await?
                {
                    has_new = true;
                    new_fields.push(transformed_field);
                    new_columns.push(transitive_arr);
                } else {
                    new_fields.push(child_field.clone());
                    new_columns.push(col.clone());
                }
            }
            if has_new {
                let fields = Fields::from(new_fields);
                let new_arr = Arc::new(StructArray::new(
                    fields.clone(),
                    new_columns,
                    arr.nulls().cloned(),
                ));
                let new_field =
                    ArrowField::new(field.name(), DataType::Struct(fields), field.is_nullable());
                Ok(Some((Arc::new(new_field), new_arr as ArrayRef)))
            } else {
                Ok(None)
            }
        }
        DataType::List(items_field) => {
            let list_arr = arr.as_list::<i32>();
            let rewrap = |transformed_items_field: FieldRef, transformed_items_arr: ArrayRef| {
                let new_arr = ListArray::new(
                    transformed_items_field.clone(),
                    list_arr.offsets().clone(),
                    transformed_items_arr,
                    list_arr.nulls().cloned(),
                );
                let new_field = ArrowField::new(
                    field.name(),
                    DataType::List(transformed_items_field),
                    field.is_nullable(),
                );
                Ok(Some((Arc::new(new_field), Arc::new(new_arr) as ArrayRef)))
            };
            if let Some((transformed_items_field, transformed_items_arr)) = transformer
                .transform(list_arr.values().as_ref(), items_field.as_ref())
                .await?
            {
                rewrap(transformed_items_field, transformed_items_arr)
            } else if let Some((transformed_items_field, transformed_items_arr)) = transform_array(
                list_arr.values().as_ref(),
                items_field.as_ref(),
                transformer,
            )
            .await?
            {
                rewrap(transformed_items_field, transformed_items_arr)
            } else {
                Ok(None)
            }
        }
        DataType::LargeList(items_field) => {
            let list_arr = arr.as_list::<i64>();
            let rewrap = |transformed_items_field: FieldRef, transformed_items_arr: ArrayRef| {
                let new_arr = LargeListArray::new(
                    transformed_items_field.clone(),
                    list_arr.offsets().clone(),
                    transformed_items_arr,
                    list_arr.nulls().cloned(),
                );
                let new_field = ArrowField::new(
                    field.name(),
                    DataType::List(transformed_items_field),
                    field.is_nullable(),
                );
                Ok(Some((Arc::new(new_field), Arc::new(new_arr) as ArrayRef)))
            };
            if let Some((transformed_items_field, transformed_items_arr)) = transformer
                .transform(list_arr.values().as_ref(), items_field.as_ref())
                .await?
            {
                rewrap(transformed_items_field, transformed_items_arr)
            } else if let Some((transformed_items_field, transformed_items_arr)) = transform_array(
                list_arr.values().as_ref(),
                items_field.as_ref(),
                transformer,
            )
            .await?
            {
                rewrap(transformed_items_field, transformed_items_arr)
            } else {
                Ok(None)
            }
        }
        _ => transformer.transform(arr, field).await,
    }
}

/// Applies a transformation to a batch of data
///
/// This is similar to [`transform_array`]
pub async fn transform_batch(
    batch: &RecordBatch,
    transformer: &mut dyn ArrayTransformer,
) -> Result<Option<RecordBatch>> {
    let mut new_columns = Vec::new();
    let mut new_fields = Vec::new();
    let mut has_new = false;
    for (col, field) in batch.columns().iter().zip(batch.schema().fields()) {
        if let Some((transformed_field, transformed_arr)) =
            transformer.transform(col, field).await?
        {
            has_new = true;
            new_fields.push(transformed_field);
            new_columns.push(transformed_arr);
        } else if let Some((transformed_field, transitive_arr)) =
            transform_array(&col, field, transformer).await?
        {
            has_new = true;
            new_fields.push(transformed_field);
            new_columns.push(transitive_arr);
        } else {
            new_fields.push(field.clone());
            new_columns.push(col.clone());
        }
    }
    if has_new {
        let new_schema = ArrowSchema::new(new_fields);
        Ok(Some(RecordBatch::try_new(
            Arc::new(new_schema),
            new_columns,
        )?))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use arrow::{
        array::{AsArray, LargeBinaryBuilder},
        datatypes::UInt64Type,
    };
    use arrow_array::{
        Array, ArrayRef, Float32Array, Float64Array, Int32Array, ListArray, RecordBatch,
        StructArray,
    };
    use arrow_buffer::{OffsetBuffer, ScalarBuffer};
    use arrow_cast::cast;
    use arrow_schema::{DataType, Field as ArrowField, FieldRef, Fields, Schema as ArrowSchema};
    use lance_arrow::RecordBatchExt;
    use lance_io::object_store::ObjectStore;
    use object_store::path::Path;
    use rstest::{fixture, rstest};
    use tempfile::{tempdir, TempDir};

    use crate::dataset::blob::BlobWriter;

    use super::{transform_batch, ArrayTransformer, Result};

    struct TemporaryDir {
        _guard: TempDir,
        path_str: String,
        path: Path,
    }

    #[fixture]
    fn tmpdir() -> TemporaryDir {
        let _guard = tempdir().unwrap();
        let path_str = _guard.path().to_str().unwrap().to_string();
        let path = Path::from(path_str.clone());
        TemporaryDir {
            _guard,
            path_str,
            path,
        }
    }

    struct StoreFixture {
        _tmpdir: TemporaryDir,
        store: Arc<ObjectStore>,
    }

    #[fixture]
    async fn store_fixture(tmpdir: TemporaryDir) -> StoreFixture {
        let store = Arc::new(ObjectStore::from_path(&tmpdir.path_str).unwrap().0);
        StoreFixture {
            _tmpdir: tmpdir,
            store,
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_blob_writer(#[future] store_fixture: StoreFixture) {
        let store_fixture = store_fixture.await;
        let mut writer = BlobWriter::new(store_fixture.store, store_fixture._tmpdir.path, 100);

        let make_blob_batch = |num_rows| {
            let mut blobs_builder = LargeBinaryBuilder::new();
            for _ in 0..num_rows {
                blobs_builder.append_value([42; 17]);
            }
            let blobs_arr = blobs_builder.finish();
            let schema =
                ArrowSchema::new(vec![ArrowField::new("blobs", DataType::LargeBinary, true)]);
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(blobs_arr)]).unwrap()
        };

        // 101 values that are each 17 bytes should be 1717 bytes which should span 17 full
        // blob files and one partial blob file.
        for _ in 0..10 {
            let deblobbed = writer.write_blobs(make_blob_batch(10)).await.unwrap();
            let paths = deblobbed.column_by_qualified_name("blobs.path").unwrap();
            let paths = paths.as_string::<i32>();
            assert!((0..10).all(|idx| !paths.value(idx).is_empty()));

            let positions = deblobbed
                .column_by_qualified_name("blobs.position")
                .unwrap();
            let positions = positions.as_primitive::<UInt64Type>();
            assert!((0..10).all(|idx| positions.value(idx) == idx as u64 * 17));

            let sizes = deblobbed.column_by_qualified_name("blobs.size").unwrap();
            let sizes = sizes.as_primitive::<UInt64Type>();
            assert!((0..10).all(|idx| sizes.value(idx) == 17));
        }
        writer.write_blobs(make_blob_batch(1)).await.unwrap();

        let blob_files = writer.flush().await.unwrap();
        assert_eq!(blob_files.len(), 11);
    }

    #[tokio::test]
    async fn test_transform() {
        struct Fp32ToFp64;

        #[async_trait::async_trait]
        impl ArrayTransformer for Fp32ToFp64 {
            async fn transform(
                &mut self,
                array: &dyn Array,
                field: &ArrowField,
            ) -> Result<Option<(FieldRef, ArrayRef)>> {
                match field.data_type() {
                    DataType::Float32 => {
                        let new_field =
                            ArrowField::new(field.name(), DataType::Float64, field.is_nullable());
                        let fp64_arr = cast(array, &DataType::Float64)?;
                        Ok(Some((Arc::new(new_field), fp64_arr)))
                    }
                    _ => Ok(None),
                }
            }
        }

        // Creating:
        //
        // {
        //   "top_level_float": [1.0],
        //   "list": [
        //      [{"nested": [2.0], "unmodified": [3]}]
        //    ]
        // }
        //

        let tlf = Float32Array::from(vec![1.0]);
        let nested = Float32Array::from(vec![2.0]);
        let unmodified = Int32Array::from(vec![3]);
        let struc = StructArray::new(
            Fields::from(vec![
                ArrowField::new("nested", DataType::Float32, true),
                ArrowField::new("unmodified", DataType::Int32, true),
            ]),
            vec![Arc::new(nested), Arc::new(unmodified.clone())],
            None,
        );
        let offsets = OffsetBuffer::new(ScalarBuffer::<i32>::from(vec![0, 1]));
        let list = ListArray::new(
            Arc::new(ArrowField::new("item", struc.data_type().clone(), true)),
            offsets,
            Arc::new(struc) as ArrayRef,
            None,
        );
        let schema = ArrowSchema::new(vec![
            ArrowField::new("top_level_float", DataType::Float32, true),
            ArrowField::new("list", list.data_type().clone(), true),
        ]);
        let batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(tlf), Arc::new(list)]).unwrap();

        // Same thing but with f64 instead of f32
        let expected_tlf = Float64Array::from(vec![1.0]);
        let expected_nested = Float64Array::from(vec![2.0]);
        let expected_struc = StructArray::new(
            Fields::from(vec![
                ArrowField::new("nested", DataType::Float64, true),
                ArrowField::new("unmodified", DataType::Int32, true),
            ]),
            vec![Arc::new(expected_nested), Arc::new(unmodified)],
            None,
        );
        let expected_offsets = OffsetBuffer::new(ScalarBuffer::<i32>::from(vec![0, 1]));
        let expected_list = ListArray::new(
            Arc::new(ArrowField::new(
                "item",
                expected_struc.data_type().clone(),
                true,
            )),
            expected_offsets,
            Arc::new(expected_struc) as ArrayRef,
            None,
        );
        let expected_schema = ArrowSchema::new(vec![
            ArrowField::new("top_level_float", DataType::Float64, true),
            ArrowField::new("list", expected_list.data_type().clone(), true),
        ]);
        let expected = RecordBatch::try_new(
            Arc::new(expected_schema),
            vec![Arc::new(expected_tlf), Arc::new(expected_list)],
        )
        .unwrap();

        let actual = transform_batch(&batch, &mut Fp32ToFp64 {})
            .await
            .unwrap()
            .unwrap();
        assert_eq!(expected, actual);
    }
}
