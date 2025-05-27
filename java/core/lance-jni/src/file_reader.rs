use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::{error::{Error, Result}, traits::IntoJava, JNIEnvExt, RT};
use arrow::{array::RecordBatchReader, ffi::FFI_ArrowSchema, ffi_stream::FFI_ArrowArrayStream};
use arrow_schema::SchemaRef;
use jni::objects::JIntArray;
use jni::{
    objects::{JObject, JString},
    sys::{jint, jlong},
    JNIEnv,
};
use lance::io::ObjectStore;
use lance_core::cache::FileMetadataCache;
use lance_core::datatypes::Schema;
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::v2::reader::{FileReader, FileReaderOptions, ReaderProjection};
use lance_index::scalar::IndexReader;
use lance_io::{
    scheduler::{ScanScheduler, SchedulerConfig},
    utils::CachedFileSize,
    ReadBatchParams,
};
use object_store::path::Path;

pub const NATIVE_READER: &str = "nativeFileReaderHandle";

#[derive(Clone, Debug)]
pub struct BlockingFileReader {
    pub(crate) inner: Arc<FileReader>,
}

impl BlockingFileReader {
    pub fn create(file_reader: Arc<FileReader>) -> Self {
        Self { inner: file_reader }
    }

    pub fn open_stream(
        &self,
        batch_size: u32,
        read_batch_params: ReadBatchParams,
        reader_projection: Option<ReaderProjection>,
        filter_expression: FilterExpression
    ) -> Result<Box<dyn RecordBatchReader + Send + 'static>> {
        Ok(self.inner.read_stream_projected_blocking(
            read_batch_params,
            batch_size,
            reader_projection,
            filter_expression,
        )?)
    }

    pub fn schema(&self) -> Result<SchemaRef> {
        Ok(Arc::new(self.inner.schema().as_ref().into()))
    }

    pub fn num_rows(&self) -> u64 {
        self.inner.num_rows()
    }
}

impl IntoJava for BlockingFileReader {
    fn into_java<'local>(self, env: &mut JNIEnv<'local>) -> Result<JObject<'local>> {
        attach_native_reader(env, self)
    }
}

fn attach_native_reader<'local>(
    env: &mut JNIEnv<'local>,
    reader: BlockingFileReader,
) -> Result<JObject<'local>> {
    let j_reader = create_java_reader_object(env)?;
    unsafe { env.set_rust_field(&j_reader, NATIVE_READER, reader) }?;
    Ok(j_reader)
}

fn create_java_reader_object<'a>(env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
    let res = env.new_object("com/lancedb/lance/file/LanceFileReader", "()V", &[])?;
    Ok(res)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_file_LanceFileReader_openNative<'local>(
    mut env: JNIEnv<'local>,
    _reader_class: JObject,
    file_uri: JString,
) -> JObject<'local> {
    ok_or_throw!(env, inner_open(&mut env, file_uri,))
}

fn inner_open<'local>(env: &mut JNIEnv<'local>, file_uri: JString) -> Result<JObject<'local>> {
    let file_uri_str: String = env.get_string(&file_uri)?.into();

    let reader = RT.block_on(async move {
        let (obj_store, path) = ObjectStore::from_uri(&file_uri_str).await?;
        let config = SchedulerConfig::max_bandwidth(&obj_store);
        let scan_scheduler = ScanScheduler::new(obj_store, config);

        let file_scheduler = scan_scheduler
            .open_file(&Path::parse(&path)?, &CachedFileSize::unknown())
            .await?;
        FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &FileMetadataCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await
    })?;

    let reader = BlockingFileReader::create(Arc::new(reader));

    reader.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_file_LanceFileReader_closeNative<'local>(
    mut env: JNIEnv<'local>,
    reader: JObject,
) -> JObject<'local> {
    let maybe_err =
        unsafe { env.take_rust_field::<_, _, BlockingFileReader>(reader, NATIVE_READER) };
    match maybe_err {
        Ok(_) => {}
        // We were already closed, do nothing
        Err(jni::errors::Error::NullPtr(_)) => {}
        Err(err) => Error::from(err).throw(&mut env),
    }
    JObject::null()
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_file_LanceFileReader_numRowsNative(
    mut env: JNIEnv<'_>,
    reader: JObject,
) -> jlong {
    match inner_num_rows(&mut env, reader) {
        Ok(num_rows) => num_rows,
        Err(e) => {
            e.throw(&mut env);
            0
        }
    }
}

// If the reader is closed, the native handle will be null and we will get a JniError::NullPtr
// error when we call get_rust_field.  Translate that into a more meaningful error.
fn unwrap_reader<T>(val: std::result::Result<T, jni::errors::Error>) -> Result<T> {
    match val {
        Ok(val) => Ok(val),
        Err(jni::errors::Error::NullPtr(_)) => Err(Error::io_error(
            "FileReader has already been closed".to_string(),
        )),
        err => Ok(err?),
    }
}

fn inner_num_rows(env: &mut JNIEnv<'_>, reader: JObject) -> Result<jlong> {
    let reader = unsafe { env.get_rust_field::<_, _, BlockingFileReader>(reader, NATIVE_READER) };
    let reader = unwrap_reader(reader)?;
    Ok(reader.num_rows() as i64)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_file_LanceFileReader_populateSchemaNative(
    mut env: JNIEnv,
    reader: JObject,
    schema_addr: jlong,
) {
    ok_or_throw_without_return!(env, inner_populate_schema(&mut env, reader, schema_addr));
}

fn inner_populate_schema(env: &mut JNIEnv, reader: JObject, schema_addr: jlong) -> Result<()> {
    let reader = unsafe { env.get_rust_field::<_, _, BlockingFileReader>(reader, NATIVE_READER) };
    let reader = unwrap_reader(reader)?;
    let schema = reader.schema()?;
    let ffi_schema = FFI_ArrowSchema::try_from(schema.as_ref())?;
    unsafe { std::ptr::write_unaligned(schema_addr as *mut FFI_ArrowSchema, ffi_schema) }
    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_file_LanceFileReader_readAllNative(
    mut env: JNIEnv<'_>,
    reader: JObject,
    batch_size: jint,
    projected_names:JObject,
    selection_ranges: JObject,
    stream_addr: jlong,
) {
    let result = (|| -> Result<()>
    {
        let mut read_parameter = ReadBatchParams::default();
        let mut reader_projection: Option<ReaderProjection> = None;
        // We get reader here not from env.get_rust_field, because we need reader: MutexGuard<BlockingFileReader> has no relationship with the env lifecycle.
        // If we get reader from env.get_rust_field, we can't use env (can't borrow again) until we drop the reader.
        let reader = unsafe {
            let reader_ref = reader.as_ref();
            let ptr = env.get_field(reader_ref, NATIVE_READER, "J")?.j()? as *mut Mutex<BlockingFileReader>;
            let guard = env.lock_obj(reader_ref)?;
            if ptr.is_null() {
                return Err(Error::io_error("FileReader has already been closed".to_string()));
            }
            (*ptr).lock().unwrap()
        };
        
        if !projected_names.is_null() {
            let schema = Schema::try_from(reader.schema()?.as_ref())?;
            let column_names: Vec<String> = env
                .get_strings(&projected_names)?;
            let names: Vec<&str> = column_names.iter().map(|s| s.as_str()).collect();
            reader_projection = Some(ReaderProjection::from_column_names(&schema, names.as_slice())?);
        }

        if !selection_ranges.is_null() {
            let jarry: JIntArray = selection_ranges.into();
            let array_length = env.get_array_length(&jarry)?;
            if (array_length > 0) {
                let mut ranges: Vec<Range<u64>> = Vec::with_capacity((array_length / 2) as usize);
                let mut rust_buffer: Vec<i32> = vec![0; array_length as usize];
                env.get_int_array_region(&jarry, 0, &mut rust_buffer)?;
                for i in (0..array_length as usize).step_by(2) {
                    let start_i32 = rust_buffer[i];
                    let end_i32 = rust_buffer[i + 1];
                    if start_i32 < 0 || end_i32 < 0 {
                        return Err(Error::input_error("Range values must be non-negative".to_string()))
                    }
                    if start_i32 > end_i32 {
                        return Err(Error::input_error("Range start must be less than or equal to end".to_string()))
                    }
                    let start: u64 = start_i32 as u64;
                    let end: u64 = end_i32 as u64;
                    ranges.push(Range { start, end });
                }
                read_parameter = ReadBatchParams::Ranges(ranges.into_boxed_slice().into());
            }
        }
        inner_read_all(&reader, batch_size, read_parameter, reader_projection, FilterExpression::no_filter(), stream_addr)
    })();
    if let Err(e) = result {
        e.throw(&mut env);
    }
}

fn inner_read_all(
    reader: &BlockingFileReader,
    batch_size: jint,
    read_batch_params: ReadBatchParams,
    reader_projection: Option<ReaderProjection>,
    filter_expression: FilterExpression,
    stream_addr: jlong,
) -> Result<()> {
    let arrow_stream = reader.open_stream(batch_size as u32, read_batch_params, reader_projection, filter_expression)?;
    let ffi_stream = FFI_ArrowArrayStream::new(arrow_stream);
    unsafe { std::ptr::write_unaligned(stream_addr as *mut FFI_ArrowArrayStream, ffi_stream) }
    Ok(())
}
