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
    objects::JObject,
    sys::{jint, jlong},
    JNIEnv,
};

use crate::{
    blocking_dataset::{BlockingDataset, NATIVE_DATASET},
    error::{Error, Result},
    RT,
};

fn fragment_count_rows(dataset: &BlockingDataset, fragment_id: jlong) -> Result<jint> {
    let Some(fragment) = dataset.inner.get_fragment(fragment_id as usize) else {
        return Err(Error::InvalidArgument {
            message: format!("Fragment not found: {}", fragment_id),
        });
    };
    Ok(RT.block_on(fragment.count_rows())? as jint)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Fragment_countRowsNative(
    mut env: JNIEnv,
    _jfragment: JObject,
    jdataset: JObject,
    fragment_id: jlong,
) -> jint {
    let res = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }
                .expect("Dataset handle not set");
        fragment_count_rows(&dataset, fragment_id)
    };
    match res {
        Ok(r) => r,
        Err(e) => {
            e.throw(&mut env);
            -1
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_ipc_FragmentScanner_getSchema(
    mut env: JNIEnv,
    _scanner: JObject,
    jdataset: JObject,
    fragment_id: jint,
    columns: JObject, // Optional<List[String]>
    schema: jlong
) {
    let res = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }
                .expect("Dataset handle not set");
        let Some(fragment) = dataset.inner.get_fragment(fragment_id as usize) else {
            return Err(Error::InvalidArgument {
                message: format!("Fragment not found: {}", fragment_id),
            });
        };

    };
}