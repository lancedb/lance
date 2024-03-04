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

use jni::{
    objects::{JObject, JString, JValue},
    JNIEnv,
};
use lance::dataset::fragment::FileFragment;

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_ipc_FragmentArrowReader_open<'local>(
    mut _env: JNIEnv<'local>,
    _path: JString,
) -> JObject<'local> {
    todo!("Implement FragmentArrowReader::open")
}

pub(crate) fn new_java_fragment<'a>(env: &mut JNIEnv<'a>, fragment: FileFragment) -> JObject<'a> {
    let j_fragment = env
        .new_object(
            "com/lancedb/lance/Fragment",
            "(j)V",
            &[JValue::Long(fragment.id() as i64)],
        )
        .expect("Failed to allocate Fragment object");

    j_fragment
}
