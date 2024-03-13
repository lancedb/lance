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

use arrow::{ffi::FFI_ArrowSchema, ffi_stream::FFI_ArrowArrayStream};
use arrow_schema::Schema;
use datafusion::execution::SendableRecordBatchStream;
use jni::{
    objects::JObject,
    sys::{jint, jlong},
    JNIEnv,
};
use lance::dataset::{fragment::FileFragment, scanner::Scanner};
use snafu::{location, Location};

use crate::{
    blocking_dataset::{BlockingDataset, NATIVE_DATASET},
    error::{Error, Result},
    ffi::JNIEnvExt,
    RT,
};

fn fragment_count_rows(dataset: &BlockingDataset, fragment_id: jlong) -> Result<jint> {
    let Some(fragment) = dataset.inner.get_fragment(fragment_id as usize) else {
        return Err(Error::InvalidArgument {
            message: format!("Fragment not found: {}", fragment_id),
        });
    };
    Ok(RT.block_on(fragment.count_rows())? as jint)
}

struct FragmentScanner {
    fragment: FileFragment,
}

impl FragmentScanner {
    async fn try_open(dataset: &BlockingDataset, fragment_id: usize) -> Result<Self> {
        let fragment = dataset
            .inner
            .get_fragment(fragment_id)
            .ok_or_else(|| Error::IO {
                message: format!("Fragment not found: {}", fragment_id),
                location: location!(),
            })?;
        Ok(Self { fragment })
    }

    /// Returns the schema of the scanner, with optional columns.
    fn schema(&self, columns: Option<Vec<String>>) -> Result<Schema> {
        let schema = self.fragment.schema();
        let schema = if let Some(columns) = columns {
            schema
                .project(&columns)
                .map_err(|e| Error::InvalidArgument {
                    message: format!("Failed to select columns: {}", e),
                })?
        } else {
            schema.clone()
        };
        Ok((&schema).into())
    }
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Fragment_countRowsNative(
    mut env: JNIEnv,
    _jfragment: JObject,
    jdataset: JObject,
    fragment_id: jlong,
) -> jint {
    let res = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }
                .expect("Dataset handle not set");
        fragment_count_rows(&dataset, fragment_id)
    };
    match res {
        Ok(r) => r,
        Err(e) => {
            e.throw(&mut env);
            -1
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_ipc_FragmentScanner_getSchema(
    mut env: JNIEnv,
    _scanner: JObject,
    jdataset: JObject,
    fragment_id: jint,
    columns: JObject, // Optional<String[]>
) -> jlong {
    let columns = match env.get_strings_opt(&columns) {
        Ok(c) => c,
        Err(e) => {
            env.throw(e.to_string()).expect("Failed to throw exception");
            return -1;
        }
    };

    let res = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }
                .expect("Dataset handle not set");
        dataset.clone()
    };
    let scanner = ok_or_throw_with_return!(
        env,
        RT.block_on(async { FragmentScanner::try_open(&res, fragment_id as usize).await }),
        0
    );

    let schema = ok_or_throw_with_return!(env, scanner.schema(columns), 0);
    let ffi_schema =
        Box::new(FFI_ArrowSchema::try_from(&schema).expect("Failed to convert schema"));
    Box::into_raw(ffi_schema) as jlong
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_ipc_FragmentArrowReader_openStream(
    mut env: JNIEnv,
    jreader: JObject,
    jdataset: JObject,
    fragment_id: jint,
    columns: JObject, // Optional<String[]>
    batch_size: jlong,
) -> jlong {
    let columns = match env.get_strings_opt(&columns) {
        Ok(c) => c,
        Err(e) => {
            env.throw(e.to_string()).expect("Failed to throw exception");
            return -1;
        }
    };
    let dataset = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }
                .expect("Dataset handle not set");
        dataset.clone()
    };
    let fragment = match dataset.inner.get_fragment(fragment_id as usize) {
        Some(f) => f,
        None => {
            env.throw("Fragment not found")
                .expect("Throw exception failed");
            return -1;
        }
    };
    let mut scanner: Scanner = fragment.scan();
    if let Some(cols) = columns {
        scanner.project(&cols);
    };
    scanner.batch_size(batch_size as usize);

    let stream = match RT.block_on(async { scanner.try_into_stream().await }) {
        Ok(s) => s,
        Err(e) => {
            env.throw(e.to_string()).expect("Throw exception failed");
            return -1;
        }
    };
    let stream = stream.map(|r| match r {
        Ok(b) => Ok(Box::new(b) as SendableRecordBatchStream),
        Err(e) => Err(e),
    });

    let mut ffi_stream = FFI_ArrowArrayStream::new(Box::new(stream));

    -1
}
