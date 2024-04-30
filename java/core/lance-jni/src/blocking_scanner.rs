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

use std::sync::Arc;

use crate::ffi::JNIEnvExt;
use arrow::{ffi::FFI_ArrowSchema, ffi_stream::FFI_ArrowArrayStream};
use arrow_schema::SchemaRef;
use jni::{objects::JObject, sys::jlong, JNIEnv};
use lance::dataset::scanner::{DatasetRecordBatchStream, Scanner};
use lance_io::ffi::to_ffi_arrow_array_stream;

use crate::{
    blocking_dataset::{BlockingDataset, NATIVE_DATASET},
    traits::IntoJava,
    Error, Result, RT,
};

pub const NATIVE_SCANNER: &str = "nativeScannerHandle";

#[derive(Clone)]
pub struct BlockingScanner {
    pub(crate) inner: Arc<Scanner>,
}

impl BlockingScanner {
    pub fn create(scanner: Scanner) -> Self {
        Self {
            inner: Arc::new(scanner),
        }
    }

    pub fn open_stream(&self) -> Result<DatasetRecordBatchStream> {
        Ok(RT.block_on(self.inner.try_into_stream())?)
    }

    pub fn schema(&self) -> Result<SchemaRef> {
        Ok(RT.block_on(self.inner.schema())?)
    }
}

impl IntoJava for BlockingScanner {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> JObject<'a> {
        attach_native_scanner(env, self)
    }
}

fn attach_native_scanner<'local>(
    env: &mut JNIEnv<'local>,
    scanner: BlockingScanner,
) -> JObject<'local> {
    let j_scanner = create_java_scanner_object(env);
    // This block sets a native Rust object (scanner) as a field in the Java object (j_scanner).
    // Caution: This creates a potential for memory leaks. The Rust object (scanner) is not
    // automatically garbage-collected by Java, and its memory will not be freed unless
    // explicitly handled.
    //
    // To prevent memory leaks, ensure the following:
    // 1. The Java object (`j_scanner`) should implement the `java.io.Closeable` interface.
    // 2. Users of this Java object should be instructed to always use it within a try-with-resources
    //    statement (or manually call the `close()` method) to ensure that `self.close()` is invoked.
    match unsafe { env.set_rust_field(&j_scanner, NATIVE_SCANNER, scanner) } {
        Ok(_) => j_scanner,
        Err(err) => {
            env.throw_new(
                "java/lang/RuntimeException",
                format!("Failed to set native handle for scanner: {}", err),
            )
            .expect("Error throwing exception");
            JObject::null()
        }
    }
}

fn create_java_scanner_object<'a>(env: &mut JNIEnv<'a>) -> JObject<'a> {
    env.new_object("com/lancedb/lance/ipc/Scanner", "()V", &[])
        .expect("Failed to create Java Scanner instance")
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_ipc_Scanner_createScanner<'local>(
    mut env: JNIEnv<'local>,
    _reader: JObject,
    jdataset: JObject,
    fragment_id_obj: JObject,      // Optional<Integer>
    columns_obj: JObject,          // Optional<List<String>>
    substrait_filter_obj: JObject, // Optional<ByteBuffer>
    filter_obj: JObject,           // Optional<String>
    batch_size_obj: JObject,       // Optional<Long>
) -> JObject<'local> {
    let dataset = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }
                .expect("Dataset handle not set");
        dataset.clone()
    };
    let mut scanner = dataset.inner.scan();
    let fragment_id_opt = ok_or_throw!(env, env.get_int_opt(&fragment_id_obj));
    if let Some(fragment_id) = fragment_id_opt {
        let Some(fragment) = dataset.inner.get_fragment(fragment_id as usize) else {
            env.throw_new(
                "java/lang/RuntimeException",
                format!("fragment id {fragment_id} not found"),
            )
            .expect("failed to throw java exception");
            return JObject::null();
        };
        scanner.with_fragments(vec![fragment.metadata().clone()]);
    }
    let columns_opt = ok_or_throw!(env, env.get_strings_opt(&columns_obj));
    if let Some(columns) = columns_opt {
        ok_or_throw!(env, scanner.project(&columns));
    };
    let substrait_opt = ok_or_throw!(env, env.get_bytes_opt(&substrait_filter_obj));
    if let Some(substrait) = substrait_opt {
        ok_or_throw!(
            env,
            RT.block_on(async { scanner.filter_substrait(substrait).await })
        );
    }
    let filter_opt = ok_or_throw!(env, env.get_string_opt(&filter_obj));
    if let Some(filter) = filter_opt {
        ok_or_throw!(env, scanner.filter(filter.as_str()));
    }
    let batch_size_opt = ok_or_throw!(env, env.get_long_opt(&batch_size_obj));
    if let Some(batch_size) = batch_size_opt {
        scanner.batch_size(batch_size as usize);
    }
    let scanner = BlockingScanner::create(scanner);
    scanner.into_java(&mut env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_ipc_Scanner_releaseNativeScanner(
    mut env: JNIEnv,
    j_scanner: JObject,
) {
    let _: BlockingScanner = unsafe {
        env.take_rust_field(j_scanner, NATIVE_SCANNER)
            .expect("Failed to take native scanner handle")
    };
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_ipc_Scanner_openStream(
    mut env: JNIEnv,
    j_scanner: JObject,
    stream_addr: jlong,
) {
    let scanner = {
        let scanner_guard =
            unsafe { env.get_rust_field::<_, _, BlockingScanner>(j_scanner, NATIVE_SCANNER) }
                .expect("Failed to get native scanner handle");
        scanner_guard.clone()
    };
    let record_batch_stream = ok_or_throw_without_return!(env, scanner.open_stream());
    let ffi_stream = to_ffi_arrow_array_stream(record_batch_stream, RT.handle().clone()).unwrap();
    unsafe { std::ptr::write_unaligned(stream_addr as *mut FFI_ArrowArrayStream, ffi_stream) }
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_ipc_Scanner_importFfiSchema(
    mut env: JNIEnv,
    j_scanner: JObject,
    schema_addr: jlong,
) {
    let scanner = {
        let scanner_guard =
            unsafe { env.get_rust_field::<_, _, BlockingScanner>(j_scanner, NATIVE_SCANNER) }
                .expect("Failed to get native scanner handle");
        scanner_guard.clone()
    };
    let schema = ok_or_throw_without_return!(env, scanner.schema());
    let ffi_schema = ok_or_throw_without_return!(env, FFI_ArrowSchema::try_from(&*schema));
    unsafe { std::ptr::write_unaligned(schema_addr as *mut FFI_ArrowSchema, ffi_schema) }
}
