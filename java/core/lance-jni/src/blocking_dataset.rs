// Copyright 2024 Lance Developers.
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

use crate::ffi::JNIEnvExt;
use crate::{traits::IntoJava, Error, Result, RT};
use arrow::array::RecordBatchReader;
use arrow::ffi::FFI_ArrowSchema;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow_schema::Schema;
use jni::sys::jlong;
use jni::{objects::JObject, JNIEnv};
use lance::dataset::fragment::FileFragment;
use lance::dataset::transaction::Operation;
use lance::dataset::{scanner::Scanner, Dataset, WriteParams};
use lance_io::ffi::to_ffi_arrow_array_stream;
use snafu::{location, Location};
pub const NATIVE_DATASET: &str = "nativeDatasetHandle";

#[derive(Clone)]
pub struct BlockingDataset {
    pub(crate) inner: Dataset,
}

impl BlockingDataset {
    pub fn write(
        reader: impl RecordBatchReader + Send + 'static,
        uri: &str,
        params: Option<WriteParams>,
    ) -> Result<Self> {
        let inner = RT.block_on(Dataset::write(reader, uri, params))?;
        Ok(Self { inner })
    }

    pub fn open(uri: &str) -> Result<Self> {
        let inner = RT.block_on(Dataset::open(uri))?;
        Ok(Self { inner })
    }

    pub fn commit(uri: &str, operation: Operation, read_version: Option<u64>) -> Result<Self> {
        let inner = RT.block_on(Dataset::commit(uri, operation, read_version, None, None))?;
        Ok(Self { inner })
    }

    pub fn latest_version(&self) -> Result<u64> {
        Ok(RT.block_on(self.inner.latest_version_id())?)
    }

    pub fn count_rows(&self, filter: Option<String>) -> Result<usize> {
        Ok(RT.block_on(self.inner.count_rows(filter))?)
    }

    pub fn close(&self) {}
}

impl IntoJava for BlockingDataset {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> JObject<'a> {
        attach_native_dataset(env, self)
    }
}

fn attach_native_dataset<'local>(
    env: &mut JNIEnv<'local>,
    dataset: BlockingDataset,
) -> JObject<'local> {
    let j_dataset = create_java_dataset_object(env);
    // This block sets a native Rust object (dataset) as a field in the Java object (j_dataset).
    // Caution: This creates a potential for memory leaks. The Rust object (dataset) is not
    // automatically garbage-collected by Java, and its memory will not be freed unless
    // explicitly handled.
    //
    // To prevent memory leaks, ensure the following:
    // 1. The Java object (`j_dataset`) should implement the `java.io.Closeable` interface.
    // 2. Users of this Java object should be instructed to always use it within a try-with-resources
    //    statement (or manually call the `close()` method) to ensure that `self.close()` is invoked.
    match unsafe { env.set_rust_field(&j_dataset, NATIVE_DATASET, dataset) } {
        Ok(_) => j_dataset,
        Err(err) => {
            env.throw_new(
                "java/lang/RuntimeException",
                format!("Failed to set native handle: {}", err),
            )
            .expect("Error throwing exception");
            JObject::null()
        }
    }
}

fn create_java_dataset_object<'a>(env: &mut JNIEnv<'a>) -> JObject<'a> {
    env.new_object("com/lancedb/lance/Dataset", "()V", &[])
        .expect("Failed to create Java Dataset instance")
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_releaseNativeDataset(
    mut env: JNIEnv,
    obj: JObject,
) {
    let dataset: BlockingDataset = unsafe {
        env.take_rust_field(obj, "nativeDatasetHandle")
            .expect("Failed to take native dataset handle")
    };
    dataset.close()
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_getJsonFragments<'a>(
    mut env: JNIEnv<'a>,
    jdataset: JObject,
) -> JObject<'a> {
    let fragments = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }
                .expect("Failed to get native dataset handle");
        dataset.inner.get_fragments()
    };

    ok_or_throw!(env, create_json_fragment_list(&mut env, fragments))
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_importFfiSchema(
    mut env: JNIEnv,
    jdataset: JObject,
    arrow_schema_addr: jlong,
) {
    let schema = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }
                .expect("Failed to get native dataset handle");
        Schema::from(dataset.inner.schema())
    };
    let out_c_schema = arrow_schema_addr as *mut FFI_ArrowSchema;
    let c_schema = ok_or_throw_without_return!(env, FFI_ArrowSchema::try_from(&schema));

    unsafe {
        std::ptr::copy(std::ptr::addr_of!(c_schema), out_c_schema, 1);
        std::mem::forget(c_schema);
    };
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_ipc_DatasetScanner_openStream(
    mut env: JNIEnv,
    _reader: JObject,
    jdataset: JObject,
    columns: JObject,       // Optional<String[]>
    substrait_obj: JObject, // Optional<ByteBuffer>
    batch_size: jlong,
    stream_addr: jlong,
) {
    let columns = ok_or_throw_without_return!(env, env.get_strings_opt(&columns));
    let dataset = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }
                .expect("Dataset handle not set");
        dataset.clone()
    };
    let mut scanner: Scanner = dataset.inner.scan();
    if let Some(cols) = columns {
        ok_or_throw_without_return!(env, scanner.project(&cols));
    };
    let substrait = ok_or_throw_without_return!(env, env.get_bytes_opt(&substrait_obj));
    if let Some(substrait) = substrait {
        ok_or_throw_without_return!(
            env,
            RT.block_on(async { scanner.filter_substrait(substrait).await })
        );
    }
    scanner.batch_size(batch_size as usize);

    let stream =
        ok_or_throw_without_return!(env, RT.block_on(async { scanner.try_into_stream().await }));
    let ffi_stream = to_ffi_arrow_array_stream(stream, RT.handle().clone()).unwrap();
    unsafe { std::ptr::write_unaligned(stream_addr as *mut FFI_ArrowArrayStream, ffi_stream) }
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_ipc_DatasetScanner_getSchema(
    mut env: JNIEnv,
    _scanner: JObject,
    jdataset: JObject,
    columns: JObject, // Optional<String[]>
) -> jlong {
    let columns = ok_or_throw_with_return!(env, env.get_strings_opt(&columns), -1);

    let res = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }
                .expect("Dataset handle not set");
        dataset.clone()
    };
    let ds_schema = res.inner.schema();
    let schema = if let Some(columns) = columns {
        ok_or_throw_with_return!(env, ds_schema.project(&columns), -1)
    } else {
        ds_schema.clone()
    };
    let arrow_schema: arrow::datatypes::Schema = (&schema).into();
    let ffi_schema =
        Box::new(FFI_ArrowSchema::try_from(&arrow_schema).expect("Failed to convert schema"));
    Box::into_raw(ffi_schema) as jlong
}

fn create_json_fragment_list<'a>(
    env: &mut JNIEnv<'a>,
    fragments: Vec<FileFragment>,
) -> Result<JObject<'a>> {
    let array_list_class = env.find_class("java/util/ArrayList")?;

    let array_list = env.new_object(array_list_class, "()V", &[])?;

    for fragment in fragments {
        let json_string = serde_json::to_string(fragment.metadata()).map_err(|e| Error::JSON {
            message: e.to_string(),
            location: location!(),
        })?;
        let jstring = env.new_string(json_string)?;
        env.call_method(
            &array_list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[(&jstring).into()],
        )?;
    }
    Ok(array_list)
}
