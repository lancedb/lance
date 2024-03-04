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
    sys::{jint, jlong},
    JNIEnv,
};
use lance::dataset::fragment::FileFragment;

use crate::{
    blocking_dataset::{BlockingDataset, NATIVE_DATASET},
    error::{Error, Result},
    RT,
};

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_ipc_FragmentArrowReader_open<'local>(
    mut _env: JNIEnv<'local>,
    _path: JString,
) -> JObject<'local> {
    todo!("Implement FragmentArrowReader::open")
}

pub(crate) fn new_java_fragment<'a>(env: &mut JNIEnv<'a>, fragment: FileFragment) -> JObject<'a> {
    let j_fragment = match env.new_object(
        "com/lancedb/lance/Fragment",
        "(J)V",
        &[JValue::Long(fragment.id() as i64)],
    ) {
        Ok(f) => f,
        Err(e) => {
            env.throw(e.to_string()).unwrap();
            return JObject::null();
        }
    };

    j_fragment
}

fn fragment_count_rows(dataset: &BlockingDataset, fragment_id: i64) -> Result<jint> {
    let fragment = match dataset.inner.get_fragment(fragment_id as usize) {
        Some(f) => f,
        None => {
            return Err(Error::InvalidArgument {
                message: format!("Fragment not found: {}", fragment_id),
            });
        }
    };
    Ok(RT.block_on(fragment.count_rows())? as jint)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Fragment_countRowsNative<'a>(
    mut env: JNIEnv<'a>,
    jdataset: JObject,
    fragment_id: jlong,
) -> jint {
    match {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }
                .expect("Dataset handle not set");
        fragment_count_rows(&dataset, fragment_id)
    } {
        Ok(r) => r,
        Err(e) => {
            env.throw(e.to_string()).expect("failed to throw exception");
            -1
        }
    }
}
