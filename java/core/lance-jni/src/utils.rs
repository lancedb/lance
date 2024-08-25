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

use arrow::array::{ArrayRef, Float32Array};
use jni::objects::{JFloatArray, JObject, JString};
use jni::JNIEnv;
use lance::dataset::{WriteMode, WriteParams};
use lance_linalg::distance::DistanceType;

use crate::error::Result;
use crate::ffi::JNIEnvExt;

use lance_index::vector::Query;

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

// Convert from Java Optional<Query> to Rust Option<Query>
pub fn get_query(env: &mut JNIEnv, query_obj: JObject) -> Result<Option<Query>> {
    let query = env.get_optional(&query_obj, |env, obj| {
        let java_obj_gen = env.call_method(obj, "get", "()Ljava/lang/Object;", &[])?;
        let java_obj = java_obj_gen.l()?;

        let column = env.get_string_from_method(&java_obj, "getColumn")?;
        let key_array = env.get_vec_f32_from_method(&java_obj, "getKey")?;
        let key = Arc::new(Float32Array::from(key_array));

        let k = env.get_int_as_usize_from_method(&java_obj, "getK")?;
        let nprobes = env.get_int_as_usize_from_method(&java_obj, "getNprobes")?;

        let ef = env.get_optional_usize_from_method(&java_obj, "getEf")?;

        let refine_factor = env.get_optional_u32_from_method(&java_obj, "getRefineFactor")?;

        let distance_type: JString = env
            .call_method(&java_obj, "getMetricType", "()Ljava/lang/String;", &[])?
            .l()?
            .into();
        let distance_type: String = env.get_string(&distance_type)?.into();
        let metric_type = DistanceType::try_from(distance_type.as_str())?;

        let use_index = env.get_boolean_from_method(&java_obj, "isUseIndex")?;

        Ok(Query {
            column,
            key,
            k,
            nprobes,
            ef,
            refine_factor,
            metric_type,
            use_index,
        })
    })?;

    Ok(query)
}
