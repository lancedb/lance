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

use arrow::ffi::FFI_ArrowSchema;
use arrow_schema::Schema;
use jni::objects::JObject;
use jni::sys::jlong;
use jni::JNIEnv;
use lance::dataset::{WriteMode, WriteParams};

use crate::blocking_dataset::{BlockingDataset, NATIVE_DATASET};
use crate::ffi::JNIEnvExt;
use crate::{Error, Result};

pub fn extract_write_params(
    env: &mut JNIEnv,
    max_rows_per_file: &JObject,
    max_rows_per_group: &JObject,
    max_bytes_per_file: &JObject,
    mode: &JObject,
) -> Result<WriteParams> {
    let mut write_params = WriteParams::default();

    if let Some(max_rows_per_file_val) = env.get_int_opt(max_rows_per_file)? {
        write_params.max_rows_per_file = max_rows_per_file_val as usize;
    }
    if let Some(max_rows_per_group_val) = env.get_int_opt(max_rows_per_group)? {
        write_params.max_rows_per_group = max_rows_per_group_val as usize;
    }
    if let Some(max_bytes_per_file_val) = env.get_long_opt(max_bytes_per_file)? {
        write_params.max_bytes_per_file = max_bytes_per_file_val as usize;
    }
    if let Some(mode_val) = env.get_string_opt(mode)? {
        write_params.mode = WriteMode::try_from(mode_val.as_str())?;
    }
    Ok(write_params)
}

pub fn import_ffi_schema(
    mut env: JNIEnv,
    jdataset: JObject,
    arrow_schema_addr: jlong,
    columns: Option<Vec<String>>,
) {
    let dataset = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }
                .expect("Failed to get native dataset handle");
        dataset.clone()
    };
    let schema = if let Some(columns) = columns {
        let ds_schema = ok_or_throw_without_return!(env, dataset.inner.schema().project(&columns));
        Schema::from(&ds_schema)
    } else {
        Schema::from(dataset.inner.schema())
    };

    let c_schema = ok_or_throw_without_return!(env, FFI_ArrowSchema::try_from(&schema));
    let out_c_schema = unsafe { &mut *(arrow_schema_addr as *mut FFI_ArrowSchema) };
    let _old = std::mem::replace(out_c_schema, c_schema);
}
