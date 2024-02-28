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
use jni::strings::JavaStr;
use jni::JNIEnv;

use crate::{Error, Result};

pub fn throw_java_exception(env: &mut JNIEnv, err_msg: &str) {
    env.throw_new("java/lang/RuntimeException", err_msg)
        .expect("Error throwing exception");
}

pub trait FromJObject<T> {
    fn extract(&self) -> Result<T>;
}

impl FromJObject<i32> for JObject<'_> {
    fn extract(&self) -> Result<i32> {
        Ok(JValue::from(self).i()?)
    }
}

impl FromJObject<i64> for JObject<'_> {
    fn extract(&self) -> Result<i64> {
        Ok(JValue::from(self).j()?)
    }
}

impl FromJObject<f32> for JObject<'_> {
    fn extract(&self) -> Result<f32> {
        Ok(JValue::from(self).f()?)
    }
}

impl FromJObject<f64> for JObject<'_> {
    fn extract(&self) -> Result<f64> {
        Ok(JValue::from(self).d()?)
    }
}

pub trait JMapExt {
    fn get_string(&self, env: &mut JNIEnv, key: &str) -> Result<Option<String>>;

    fn get_i32(&self, env: &mut JNIEnv, key: &str) -> Result<Option<i32>>;

    fn get_i64(&self, env: &mut JNIEnv, key: &str) -> Result<Option<i64>>;
}

impl JMapExt for JMap<'_, '_, '_> {
    fn get_string(&self, env: &mut JNIEnv, key: &str) -> Result<Option<String>> {
        let key_obj: JObject = env.new_string(key)?.into();
        if let Some(value) = self.get(env, &key_obj)? {
            let value_str: JString = value.into();
            Ok(Some(env.get_string(&value_str)?.into()))
        } else {
            Ok(None)
        }
    }

    fn get_i32(&self, env: &mut JNIEnv, key: &str) -> Result<Option<i32>> {
        let key_obj: JObject = env.new_string(key)?.into();
        if let Some(value) = self.get(env, &key_obj)? {
            Ok(Some(value.extract()?))
        } else {
            Ok(None)
        }
    }

    fn get_i64(&self, env: &mut JNIEnv, key: &str) -> Result<Option<i64>> {
        let key_obj: JObject = env.new_string(key)?.into();
        if let Some(value) = self.get(env, &key_obj)? {
            Ok(Some(value.extract()?))
        } else {
            Ok(None)
        }
    }
}
