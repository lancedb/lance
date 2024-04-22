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

use jni::objects::JString;
use jni::{objects::JObject, JNIEnv};

use crate::Result;

/// Extend JNIEnv with helper functions.
pub trait JNIEnvExt {
    /// Get integers from Java List<Integer> object.
    fn get_integers(&mut self, obj: &JObject) -> Result<Vec<i32>>;

    /// Get strings from Java List<String> object.
    fn get_strings(&mut self, obj: &JObject) -> Result<Vec<String>>;

    /// Get Option<String> from Java Optional<String>.
    fn get_string_opt(&mut self, obj: &JObject) -> Result<Option<String>>;

    /// Get Option<Vec<String>> from Java Optional<List<String>>.
    fn get_strings_opt(&mut self, obj: &JObject) -> Result<Option<Vec<String>>>;

    /// Get Option<i32> from Java Optional<Integer>.
    fn get_int_opt(&mut self, obj: &JObject) -> Result<Option<i32>>;

    /// Get Option<i64> from Java Optional<Long>.
    fn get_long_opt(&mut self, obj: &JObject) -> Result<Option<i64>>;

    /// Get Option<u64> from Java Optional<Long>.
    fn get_u64_opt(&mut self, obj: &JObject) -> Result<Option<u64>>;

    fn get_optional<T, F>(&mut self, obj: &JObject, f: F) -> Result<Option<T>>
    where
        F: FnOnce(&mut JNIEnv, &JObject) -> Result<T>;
}

impl JNIEnvExt for JNIEnv<'_> {
    fn get_integers(&mut self, obj: &JObject) -> Result<Vec<i32>> {
        let list = self.get_list(obj)?;
        let mut iter = list.iter(self)?;
        let mut results = Vec::with_capacity(list.size(self)? as usize);
        while let Some(elem) = iter.next(self)? {
            let int_obj = self.call_method(elem, "intValue", "()I", &[])?;
            let int_value = int_obj.i()?;
            results.push(int_value);
        }
        Ok(results)
    }

    fn get_strings(&mut self, obj: &JObject) -> Result<Vec<String>> {
        let list = self.get_list(obj)?;
        let mut iter = list.iter(self)?;
        let mut results = Vec::with_capacity(list.size(self)? as usize);
        while let Some(elem) = iter.next(self)? {
            let jstr = JString::from(elem);
            let val = self.get_string(&jstr)?;
            results.push(val.to_str()?.to_string())
        }
        Ok(results)
    }

    fn get_string_opt(&mut self, obj: &JObject) -> Result<Option<String>> {
        self.get_optional(obj, |env, inner_obj| {
            let java_obj_gen = env.call_method(inner_obj, "get", "()Ljava/lang/Object;", &[])?;
            let java_string_obj = java_obj_gen.l()?;
            let jstr = JString::from(java_string_obj);
            let val = env.get_string(&jstr)?;
            Ok(val.to_str()?.to_string())
        })
    }

    fn get_strings_opt(&mut self, obj: &JObject) -> Result<Option<Vec<String>>> {
        self.get_optional(obj, |env, inner_obj| {
            let java_obj_gen = env.call_method(inner_obj, "get", "()Ljava/util/List;", &[])?;
            let java_list_obj = java_obj_gen.l()?;
            env.get_strings(&java_list_obj)
        })
    }

    fn get_int_opt(&mut self, obj: &JObject) -> Result<Option<i32>> {
        self.get_optional(obj, |env, inner_obj| {
            let java_obj_gen = env.call_method(inner_obj, "get", "()Ljava/lang/Object;", &[])?;
            let java_int_obj = java_obj_gen.l()?;
            let int_obj = env.call_method(java_int_obj, "intValue", "()I", &[])?;
            let int_value = int_obj.i()?;
            Ok(int_value)
        })
    }

    fn get_long_opt(&mut self, obj: &JObject) -> Result<Option<i64>> {
        self.get_optional(obj, |env, inner_obj| {
            let java_obj_gen = env.call_method(inner_obj, "get", "()Ljava/lang/Object;", &[])?;
            let java_long_obj = java_obj_gen.l()?;
            let long_obj = env.call_method(java_long_obj, "longValue", "()J", &[])?;
            let long_value = long_obj.j()?;
            Ok(long_value)
        })
    }

    fn get_u64_opt(&mut self, obj: &JObject) -> Result<Option<u64>> {
        self.get_optional(obj, |env, inner_obj| {
            let java_obj_gen = env.call_method(inner_obj, "get", "()Ljava/lang/Object;", &[])?;
            let java_long_obj = java_obj_gen.l()?;
            let long_obj = env.call_method(java_long_obj, "longValue", "()J", &[])?;
            let long_value = long_obj.j()?;
            Ok(long_value as u64)
        })
    }

    fn get_optional<T, F>(&mut self, obj: &JObject, f: F) -> Result<Option<T>>
    where
        F: FnOnce(&mut JNIEnv, &JObject) -> Result<T>,
    {
        if obj.is_null() {
            return Ok(None);
        }
        let is_empty = self.call_method(obj, "isEmpty", "()Z", &[])?;
        if is_empty.z()? {
            Ok(None)
        } else {
            f(self, obj).map(Some)
        }
    }
}
