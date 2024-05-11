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

use jni::objects::{JMap, JObject, JString, JValue};
use jni::JNIEnv;

use crate::error::{JavaErrorExt, JavaResult};

pub trait FromJObject<T> {
    fn extract(&self) -> JavaResult<T>;
}

/// Convert a Rust type into a Java Object.
pub trait IntoJava {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> JavaResult<JObject<'a>>;
}

impl FromJObject<i32> for JObject<'_> {
    fn extract(&self) -> JavaResult<i32> {
        JValue::from(self).i().infer_error()
    }
}

impl FromJObject<i64> for JObject<'_> {
    fn extract(&self) -> JavaResult<i64> {
        JValue::from(self).j().infer_error()
    }
}

impl FromJObject<f32> for JObject<'_> {
    fn extract(&self) -> JavaResult<f32> {
        JValue::from(self).f().infer_error()
    }
}

impl FromJObject<f64> for JObject<'_> {
    fn extract(&self) -> JavaResult<f64> {
        JValue::from(self).d().infer_error()
    }
}

pub trait FromJString {
    fn extract(&self, env: &mut JNIEnv) -> JavaResult<String>;
}

impl FromJString for JString<'_> {
    fn extract(&self, env: &mut JNIEnv) -> JavaResult<String> {
        Ok(env.get_string(self).infer_error()?.into())
    }
}

pub trait JMapExt {
    #[allow(dead_code)]
    fn get_string(&self, env: &mut JNIEnv, key: &str) -> JavaResult<Option<String>>;

    #[allow(dead_code)]
    fn get_i32(&self, env: &mut JNIEnv, key: &str) -> JavaResult<Option<i32>>;

    #[allow(dead_code)]
    fn get_i64(&self, env: &mut JNIEnv, key: &str) -> JavaResult<Option<i64>>;

    #[allow(dead_code)]
    fn get_f32(&self, env: &mut JNIEnv, key: &str) -> JavaResult<Option<f32>>;

    #[allow(dead_code)]
    fn get_f64(&self, env: &mut JNIEnv, key: &str) -> JavaResult<Option<f64>>;
}

fn get_map_value<T>(env: &mut JNIEnv, map: &JMap, key: &str) -> JavaResult<Option<T>>
where
    for<'a> JObject<'a>: FromJObject<T>,
{
    let key_obj: JObject = env.new_string(key).infer_error()?.into();
    if let Some(value) = map.get(env, &key_obj).infer_error()? {
        if value.is_null() {
            Ok(None)
        } else {
            Ok(Some(value.extract()?))
        }
    } else {
        Ok(None)
    }
}

impl JMapExt for JMap<'_, '_, '_> {
    fn get_string(&self, env: &mut JNIEnv, key: &str) -> JavaResult<Option<String>> {
        let key_obj: JObject = env.new_string(key).infer_error()?.into();
        if let Some(value) = self.get(env, &key_obj).infer_error()? {
            let value_str: JString = value.into();
            Ok(Some(value_str.extract(env)?))
        } else {
            Ok(None)
        }
    }

    fn get_i32(&self, env: &mut JNIEnv, key: &str) -> JavaResult<Option<i32>> {
        get_map_value(env, self, key)
    }

    fn get_i64(&self, env: &mut JNIEnv, key: &str) -> JavaResult<Option<i64>> {
        get_map_value(env, self, key)
    }

    fn get_f32(&self, env: &mut JNIEnv, key: &str) -> JavaResult<Option<f32>> {
        get_map_value(env, self, key)
    }

    fn get_f64(&self, env: &mut JNIEnv, key: &str) -> JavaResult<Option<f64>> {
        get_map_value(env, self, key)
    }
}
