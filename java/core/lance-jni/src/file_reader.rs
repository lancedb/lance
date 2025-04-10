use std::sync::Arc;

use crate::{
    error::{Error, Result},
    traits::IntoJava,
    RT,
};
use arrow::{array::RecordBatchReader, ffi::FFI_ArrowSchema, ffi_stream::FFI_ArrowArrayStream};
use arrow_schema::SchemaRef;
use jni::{
    objects::{JObject, JString},
    sys::{jint, jlong},
    JNIEnv,
};
use lance::io::ObjectStore;
use lance_core::cache::LanceCache;
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::v2::reader::{FileReader, FileReaderOptions};
use lance_io::{
    scheduler::{ScanScheduler, SchedulerConfig},
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
    ) -> Result<Box<dyn RecordBatchReader + Send + 'static>> {
        Ok(self.inner.read_stream_projected_blocking(
            ReadBatchParams::RangeFull,
            batch_size,
            None,
            FilterExpression::no_filter(),
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
        let obj_store = Arc::new(obj_store);
        let config = SchedulerConfig::max_bandwidth(&obj_store);
        let scan_scheduler = ScanScheduler::new(obj_store, config);

        let file_scheduler = scan_scheduler.open_file(&Path::parse(&path)?).await?;
        FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
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
    stream_addr: jlong,
) {
    if let Err(e) = inner_read_all(&mut env, reader, batch_size, stream_addr) {
        e.throw(&mut env);
    }
}

fn inner_read_all(
    env: &mut JNIEnv<'_>,
    reader: JObject,
    batch_size: jint,
    stream_addr: jlong,
) -> Result<()> {
    let reader = unsafe { env.get_rust_field::<_, _, BlockingFileReader>(reader, NATIVE_READER) };
    let reader = unwrap_reader(reader)?;
    let arrow_stream = reader.open_stream(batch_size as u32)?;
    let ffi_stream = FFI_ArrowArrayStream::new(arrow_stream);
    unsafe { std::ptr::write_unaligned(stream_addr as *mut FFI_ArrowArrayStream, ffi_stream) }
    Ok(())
}
