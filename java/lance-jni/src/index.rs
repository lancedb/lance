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

//! Dataset Indexing.

use jni::objects::{JObject, JString};
use jni::sys::jboolean;
use jni::{sys::jint, JNIEnv};
use lance::index::vector::VectorIndexParams;
use lance_index::{DatasetIndexExt, IndexType};
use lance_linalg::distance::DistanceType;
use snafu::{location, Location};

use crate::blocking_dataset::NATIVE_DATASET;
use crate::error::{Error, Result};
use crate::{BlockingDataset, RT};

fn parse_distance_type(val: i32) -> Result<DistanceType> {
    match val {
        1 => Ok(DistanceType::L2),
        2 => Ok(DistanceType::Cosine),
        3 => Ok(DistanceType::Dot),
        _ => Err(Error::Index {
            message: format!("invalid distance type: {val}"),
            location: location!(),
        }),
    }
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_index_IndexBuilder_createIvfPQ(
    mut env: JNIEnv,
    _builder: JObject,
    jdataset: JObject,
    column: JString,
    num_partitions: jint,
    num_sub_vectors: jint,
    distance_type: jint,
    replace: jboolean,
) {
    let mut dataset = {
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }
            .expect("Failed to get native dataset handle")
            .clone()
    };

    let column: String = if let Ok(js) = env.get_string(&column) {
        js.into()
    } else {
        env.throw_new("java/lang/IllegalArgumentException", "Invalid column name")
            .expect("Failed to throw exception");
        return;
    };

    let Ok(distance_type) = parse_distance_type(distance_type) else {
        env.throw_new(
            "java/lang/IllegalArgumentException",
            format!("Invalid distance type: {distance_type}"),
        )
        .expect("Failed to throw exception");
        return;
    };

    let params = VectorIndexParams::ivf_pq(
        num_partitions as usize,
        8,
        num_sub_vectors as usize,
        false,
        distance_type,
        50,
    );

    let res = RT
        .block_on(dataset.inner.create_index(
            &[&column],
            IndexType::Vector,
            None,
            &params,
            replace == 1,
        ))
        .map_err(|e| Error::Index {
            message: e.to_string(),
            location: location!(),
        });
    if let Err(e) = res {
        e.throw(&mut env);
    }
}
