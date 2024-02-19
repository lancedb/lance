// Copyright 2023 Lance Developers.
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

use jni::objects::{JMap, JObject, JString};
use jni::JNIEnv;

pub fn throw_java_exception(env: &mut JNIEnv, err_msg: &str) {
    env.throw_new("java/lang/RuntimeException", err_msg)
        .expect("Error throwing exception");
}

pub fn extract_path_str(env: &mut JNIEnv, path: &JString) -> Result<String, String> {
    match env.get_string(path) {
        Ok(path) => Ok(path.into()),
        Err(err) => Err(format!("Invalid path string: {}", err)),
    }
}

pub trait ExtractJniValue {
    // extract JNI JObject to rust type
    fn extract(env: &mut JNIEnv, obj: JObject) -> Result<Self, String>
    where
        Self: Sized;
}

impl ExtractJniValue for i32 {
    fn extract(env: &mut JNIEnv, obj: JObject) -> Result<Self, String> {
        env.call_method(obj, "intValue", "()I", &[])
            .and_then(|jvalue| jvalue.i())
            .map_err(|e| e.to_string())
    }
}

impl ExtractJniValue for i64 {
    fn extract(env: &mut JNIEnv, obj: JObject) -> Result<Self, String> {
        env.call_method(obj, "longValue", "()J", &[])
            .and_then(|jvalue| jvalue.j())
            .map_err(|e| e.to_string())
    }
}

impl ExtractJniValue for String {
    fn extract(env: &mut JNIEnv, obj: JObject) -> Result<Self, String> {
        env.get_string(&JString::from(obj))
            .map(|jstr| jstr.into())
            .map_err(|e| e.to_string())
    }
}

pub fn extract_and_process_jni_map_value<T, F>(
    env: &mut JNIEnv,
    map: &JMap,
    key: &str,
    mut process: F,
) -> Result<(), String>
where
    T: ExtractJniValue,
    F: FnMut(T) -> Result<(), String>, // func to apply to return value
{
    let key_obj = env
        .new_string(key)
        .map(JObject::from)
        .map_err(|e| e.to_string())?;

    match map.get(env, &key_obj) {
        Ok(Some(value_obj)) => {
            if !value_obj.is_null() {
                let value = T::extract(env, value_obj)?;
                process(value)?;
            }
            Ok(())
        }
        Ok(None) => Ok(()), // Key is not present in the map, so we do nothing
        Err(e) => Err(e.to_string()),
    }
}
