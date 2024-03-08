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

use jni::{JNIEnv, objects::JObject};
use jni::objects::JString;

use crate::Result;

/// Extend JNIEnv with helper functions.
pub trait JNIEnvExt {
    /// Get strings from Java List<String> object.
    fn get_strings(&mut self, obj: &JObject) -> Result<Vec<String>>;

    /// Get Option<Vec<String>> from Java Optional<List<String>>.
    fn get_strings_opt(&mut self, obj: &JObject) -> Result<Option<Vec<String>>>;
}

impl JNIEnvExt for JNIEnv<'_> {
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

    fn get_strings_opt(&mut self, obj: &JObject) -> Result<Option<Vec<String>>> {
        let is_empty = self.call_method(obj, "java/util/Optional/isEmpty", "()Z", &[])?;
        if !is_empty.z()? {
            Ok(None)
        } else {
            let inner = self.call_method(obj, "java/util/Optional/get", "()Ljava/util/List;", &[])?;
            let inner_obj = inner.l()?;
            Ok(Some(self.get_strings(&inner_obj)?))
        }
    }
}

#[cfg(test)]
mod tests {}
