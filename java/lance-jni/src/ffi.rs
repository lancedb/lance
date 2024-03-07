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

use jni::{objects::JObject, JNIEnv};
use jni::objects::JString;
use jni::signature::{ReturnType, TypeSignature};

use crate::{Result, Error};

/// Extend JNIEnv with helper functions.
pub trait JNIEnvExt {

    /// Get Option<JObject> from Java [java.util.Optional<..>].
    fn get_option(&mut self, obj: &JObject) -> Result<Option<JObject>>;

    /// Get strings from Java List<String> object.
    fn get_strings(&mut self, obj: &JObject) -> Result<Vec<String>>;

    /// Get Option<Vec<String>> from Java Optional<List<String>>.
    fn get_strings_opt(&mut self, obj: &JObject) -> Result<Option<Vec<String>>>;
}

impl JNIEnvExt for JNIEnv<'_> {

    fn get_option(&mut self, obj: &JObject) -> Result<Option<JObject>> {
        let is_empty = self.call_method(obj, "java/util/Optional/isEmpty", "()Z", &[])?;
        if !is_empty.z()? {
            return Ok(None);
        } else {
            let signature = TypeSignature {
              args: vec![],
                ret: ReturnType::Object,
            };
            let inner = self.call_method(obj, "java/util/Optional/get", signature.to_string(), &[])?;
            Ok(Some(inner.l()?))
        }
    }

    fn get_strings(&mut self, obj: &JObject) -> Result<Vec<String>> {
        let list = self.get_list(obj)?;
        let mut iter = list.iter(self)?;
        let mut results = vec![];
        while let Some(elem) = iter.next(self)? {
            let jstr = JString::from(elem);
            let val = self.get_string(&jstr)?;
            results.push(val.to_str()?.to_string())
        }
        Ok(results)
    }

    fn get_strings_opt(&mut self, obj: &JObject) -> Result<Option<Vec<String>>> {
        todo!()
    }
}