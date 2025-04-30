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

use jni::objects::{JIntArray, JMap, JObject, JString, JValue, JValueGen};
use jni::JNIEnv;

use crate::error::Result;

pub trait FromJObject<T> {
    fn extract(&self) -> Result<T>;
}

pub trait FromJObjectWithEnv<T> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<T>;
}

/// Convert a Rust type into a Java Object.
pub trait IntoJava {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>>;
}

impl FromJObject<i32> for JObject<'_> {
    fn extract(&self) -> Result<i32> {
        let res = JValue::from(self).i()?;
        Ok(res)
    }
}

impl FromJObject<i64> for JObject<'_> {
    fn extract(&self) -> Result<i64> {
        let res = JValue::from(self).j()?;
        Ok(res)
    }
}

impl FromJObject<f32> for JObject<'_> {
    fn extract(&self) -> Result<f32> {
        let res = JValue::from(self).f()?;
        Ok(res)
    }
}

impl FromJObject<f64> for JObject<'_> {
    fn extract(&self) -> Result<f64> {
        let res = JValue::from(self).d()?;
        Ok(res)
    }
}

pub trait FromJString {
    fn extract(&self, env: &mut JNIEnv) -> Result<String>;
}

impl FromJString for JString<'_> {
    fn extract(&self, env: &mut JNIEnv) -> Result<String> {
        Ok(env.get_string(self)?.into())
    }
}

pub trait JMapExt {
    #[allow(dead_code)]
    fn get_string(&self, env: &mut JNIEnv, key: &str) -> Result<Option<String>>;

    #[allow(dead_code)]
    fn get_i32(&self, env: &mut JNIEnv, key: &str) -> Result<Option<i32>>;

    #[allow(dead_code)]
    fn get_i64(&self, env: &mut JNIEnv, key: &str) -> Result<Option<i64>>;

    #[allow(dead_code)]
    fn get_f32(&self, env: &mut JNIEnv, key: &str) -> Result<Option<f32>>;

    #[allow(dead_code)]
    fn get_f64(&self, env: &mut JNIEnv, key: &str) -> Result<Option<f64>>;
}

fn get_map_value<T>(env: &mut JNIEnv, map: &JMap, key: &str) -> Result<Option<T>>
where
    for<'a> JObject<'a>: FromJObject<T>,
{
    let key_obj: JObject = env.new_string(key)?.into();
    if let Some(value) = map.get(env, &key_obj)? {
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
    fn get_string(&self, env: &mut JNIEnv, key: &str) -> Result<Option<String>> {
        let key_obj: JObject = env.new_string(key)?.into();
        if let Some(value) = self.get(env, &key_obj)? {
            let value_str: JString = value.into();
            Ok(Some(value_str.extract(env)?))
        } else {
            Ok(None)
        }
    }

    fn get_i32(&self, env: &mut JNIEnv, key: &str) -> Result<Option<i32>> {
        get_map_value(env, self, key)
    }

    fn get_i64(&self, env: &mut JNIEnv, key: &str) -> Result<Option<i64>> {
        get_map_value(env, self, key)
    }

    fn get_f32(&self, env: &mut JNIEnv, key: &str) -> Result<Option<f32>> {
        get_map_value(env, self, key)
    }

    fn get_f64(&self, env: &mut JNIEnv, key: &str) -> Result<Option<f64>> {
        get_map_value(env, self, key)
    }
}

pub fn export_vec<'a, 'b, T>(env: &mut JNIEnv<'a>, vec: &'b [T]) -> Result<JObject<'a>>
where
    &'b T: IntoJava,
{
    let array_list_class = env.find_class("java/util/ArrayList")?;
    let array_list = env.new_object(array_list_class, "()V", &[])?;
    for e in vec {
        let obj = &e.into_java(env)?;
        env.call_method(
            &array_list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValueGen::Object(obj)],
        )?;
    }
    Ok(array_list)
}

pub fn import_vec<'local>(env: &mut JNIEnv<'local>, obj: &JObject) -> Result<Vec<JObject<'local>>> {
    let size = env.call_method(obj, "size", "()I", &[])?.i()?;
    let mut ret = Vec::with_capacity(size as usize);
    for i in 0..size {
        let elem = env.call_method(obj, "get", "(I)Ljava/lang/Object;", &[JValueGen::Int(i)])?;
        ret.push(elem.l()?);
    }
    Ok(ret)
}

pub struct JLance<T>(pub T);

impl IntoJava for JLance<Vec<i32>> {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let arr = env.new_int_array(self.0.len() as i32)?;
        env.set_int_array_region(&arr, 0, &self.0)?;
        Ok(arr.into())
    }
}

impl IntoJava for JLance<usize> {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        Ok(env.new_object("java/lang/Long", "(J)V", &[JValueGen::Long(self.0 as i64)])?)
    }
}

impl IntoJava for JLance<Option<usize>> {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let obj = match self.0 {
            Some(v) => env.new_object("java/lang/Long", "(J)V", &[JValueGen::Long(v as i64)])?,
            None => JObject::null(),
        };
        Ok(obj)
    }
}

impl FromJObjectWithEnv<Option<i64>> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<Option<i64>> {
        let ret = if self.is_null() {
            None
        } else {
            let v = env.call_method(self, "longValue", "()J", &[])?.j()?;
            Some(v)
        };
        Ok(ret)
    }
}

impl FromJObjectWithEnv<Vec<i32>> for JIntArray<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<Vec<i32>> {
        let len = env.get_array_length(self)?;
        let mut ret: Vec<i32> = vec![0; len as usize];
        env.get_int_array_region(self, 0, ret.as_mut_slice())?;
        Ok(ret)
    }
}
