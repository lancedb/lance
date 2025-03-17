use std::sync::{Arc, Mutex};

use crate::{
    error::{Error, Result},
    traits::IntoJava,
    RT,
};
use arrow::{
    array::{RecordBatch, StructArray},
    ffi::{from_ffi_and_data_type, FFI_ArrowArray, FFI_ArrowSchema},
};
use arrow_schema::DataType;
use jni::{
    objects::{JObject, JString},
    sys::jlong,
    JNIEnv,
};
use lance::io::ObjectStore;
use lance_file::{
    v2::writer::{FileWriter, FileWriterOptions},
    version::LanceFileVersion,
};

pub const NATIVE_WRITER: &str = "nativeFileWriterHandle";

#[derive(Clone)]
pub struct BlockingFileWriter {
    pub(crate) inner: Arc<Mutex<FileWriter>>,
}

impl BlockingFileWriter {
    pub fn create(file_writer: FileWriter) -> Self {
        Self {
            inner: Arc::new(Mutex::new(file_writer)),
        }
    }
}

impl IntoJava for BlockingFileWriter {
    fn into_java<'local>(self, env: &mut JNIEnv<'local>) -> Result<JObject<'local>> {
        attach_native_writer(env, self)
    }
}

fn attach_native_writer<'local>(
    env: &mut JNIEnv<'local>,
    writer: BlockingFileWriter,
) -> Result<JObject<'local>> {
    let j_writer = create_java_writer_object(env)?;
    unsafe { env.set_rust_field(&j_writer, NATIVE_WRITER, writer) }?;
    Ok(j_writer)
}

fn create_java_writer_object<'a>(env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
    let res = env.new_object("com/lancedb/lance/file/LanceFileWriter", "()V", &[])?;
    Ok(res)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_file_LanceFileWriter_openNative<'local>(
    mut env: JNIEnv<'local>,
    _writer_class: JObject,
    file_uri: JString,
) -> JObject<'local> {
    ok_or_throw!(env, inner_open(&mut env, file_uri,))
}

fn inner_open<'local>(env: &mut JNIEnv<'local>, file_uri: JString) -> Result<JObject<'local>> {
    let file_uri_str: String = env.get_string(&file_uri)?.into();

    let writer = RT.block_on(async move {
        let (obj_store, path) = ObjectStore::from_uri(&file_uri_str).await?;
        let obj_store = Arc::new(obj_store);
        let obj_writer = obj_store.create(&path).await?;

        Result::Ok(FileWriter::new_lazy(
            obj_writer,
            FileWriterOptions {
                format_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            },
        ))
    })?;

    let writer = BlockingFileWriter::create(writer);

    writer.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_file_LanceFileWriter_closeNative<'local>(
    mut env: JNIEnv<'local>,
    writer: JObject,
) -> JObject<'local> {
    let maybe_err =
        unsafe { env.take_rust_field::<_, _, BlockingFileWriter>(writer, NATIVE_WRITER) };
    let writer = match maybe_err {
        Ok(writer) => Some(writer),
        // We were already closed, do nothing
        Err(jni::errors::Error::NullPtr(_)) => None,
        Err(err) => {
            Error::from(err).throw(&mut env);
            None
        }
    };
    if let Some(writer) = writer {
        match RT.block_on(writer.inner.lock().unwrap().finish()) {
            Ok(_) => {}
            Err(e) => {
                Error::from(e).throw(&mut env);
            }
        }
    }
    JObject::null()
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_file_LanceFileWriter_writeNative<'local>(
    mut env: JNIEnv<'local>,
    writer: JObject,
    batch_address: jlong,
    schema_address: jlong,
) -> JObject<'local> {
    if let Err(e) = inner_write_batch(&mut env, writer, batch_address, schema_address) {
        e.throw(&mut env);
        return JObject::null();
    }
    JObject::null()
}

fn inner_write_batch<'local>(
    env: &mut JNIEnv<'local>,
    writer: JObject,
    batch_address: jlong,
    schema_address: jlong,
) -> Result<()> {
    let c_array_ptr = batch_address as *mut FFI_ArrowArray;
    let c_schema_ptr = schema_address as *mut FFI_ArrowSchema;

    let c_array = unsafe { FFI_ArrowArray::from_raw(c_array_ptr) };
    let c_schema = unsafe { FFI_ArrowSchema::from_raw(c_schema_ptr) };

    let data_type = DataType::try_from(&c_schema)?;
    let array_data = unsafe { from_ffi_and_data_type(c_array, data_type) }?;
    let record_batch = RecordBatch::from(StructArray::from(array_data));

    let writer = unsafe { env.get_rust_field::<_, _, BlockingFileWriter>(writer, NATIVE_WRITER) }?;

    let mut writer = writer.inner.lock().unwrap();
    RT.block_on(writer.write_batch(&record_batch))?;
    Ok(())
}
