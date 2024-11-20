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

use std::sync::Arc;

use arrow::array::Float32Array;
use jni::objects::{JMap, JObject, JString};
use jni::JNIEnv;
use lance::dataset::{WriteMode, WriteParams};
use lance::index::vector::{IndexFileVersion, StageParams, VectorIndexParams};
use lance::io::ObjectStoreParams;
use lance_encoding::version::LanceFileVersion;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_index::IndexParams;
use lance_linalg::distance::DistanceType;

use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;

use lance_index::vector::Query;
use std::collections::HashMap;

pub fn extract_write_params(
    env: &mut JNIEnv,
    max_rows_per_file: &JObject,
    max_rows_per_group: &JObject,
    max_bytes_per_file: &JObject,
    mode: &JObject,
    storage_options_obj: &JObject,
) -> Result<WriteParams> {
    let mut write_params = WriteParams::default();

    if let Some(max_rows_per_file_val) = env.get_int_opt(max_rows_per_file)? {
        write_params.max_rows_per_file = max_rows_per_file_val as usize;
    }
    if let Some(max_rows_per_group_val) = env.get_int_opt(max_rows_per_group)? {
        write_params.max_rows_per_group = max_rows_per_group_val as usize;
    }
    if let Some(max_bytes_per_file_val) = env.get_long_opt(max_bytes_per_file)? {
        write_params.max_bytes_per_file = max_bytes_per_file_val as usize;
    }
    if let Some(mode_val) = env.get_string_opt(mode)? {
        write_params.mode = WriteMode::try_from(mode_val.as_str())?;
    }
    // Java code always sets the data storage version to Legacy for now
    write_params.data_storage_version = Some(LanceFileVersion::Legacy);
    let jmap = JMap::from_env(env, storage_options_obj)?;
    let storage_options: HashMap<String, String> = env.with_local_frame(16, |env| {
        let mut map = HashMap::new();
        let mut iter = jmap.iter(env)?;
        while let Some((key, value)) = iter.next(env)? {
            let key_jstring = JString::from(key);
            let value_jstring = JString::from(value);
            let key_string: String = env.get_string(&key_jstring)?.into();
            let value_string: String = env.get_string(&value_jstring)?.into();
            map.insert(key_string, value_string);
        }
        Ok::<_, Error>(map)
    })?;

    write_params.store_params = Some(ObjectStoreParams {
        storage_options: Some(storage_options),
        ..Default::default()
    });
    Ok(write_params)
}

// Convert from Java Optional<Query> to Rust Option<Query>
pub fn get_query(env: &mut JNIEnv, query_obj: JObject) -> Result<Option<Query>> {
    let query = env.get_optional(&query_obj, |env, obj| {
        let java_obj_gen = env.call_method(obj, "get", "()Ljava/lang/Object;", &[])?;
        let java_obj = java_obj_gen.l()?;

        let column = env.get_string_from_method(&java_obj, "getColumn")?;
        let key_array = env.get_vec_f32_from_method(&java_obj, "getKey")?;
        let key = Arc::new(Float32Array::from(key_array));

        let k = env.get_int_as_usize_from_method(&java_obj, "getK")?;
        let nprobes = env.get_int_as_usize_from_method(&java_obj, "getNprobes")?;

        let ef = env.get_optional_usize_from_method(&java_obj, "getEf")?;

        let refine_factor = env.get_optional_u32_from_method(&java_obj, "getRefineFactor")?;

        let distance_type_jstr: JString = env
            .call_method(&java_obj, "getDistanceType", "()Ljava/lang/String;", &[])?
            .l()?
            .into();
        let distance_type_str: String = env.get_string(&distance_type_jstr)?.into();
        let distance_type = DistanceType::try_from(distance_type_str.as_str())?;

        let use_index = env.get_boolean_from_method(&java_obj, "isUseIndex")?;

        Ok(Query {
            column,
            key,
            k,
            nprobes,
            ef,
            refine_factor,
            metric_type: distance_type,
            use_index,
        })
    })?;

    Ok(query)
}

pub fn get_index_params(
    env: &mut JNIEnv,
    index_params_obj: JObject,
) -> Result<Box<dyn IndexParams>> {
    let distance_type_obj: JString = env
        .call_method(
            &index_params_obj,
            "getDistanceType",
            "()Ljava/lang/String;",
            &[],
        )?
        .l()?
        .into();
    let distance_type_str: String = env.get_string(&distance_type_obj)?.into();
    let distance_type = DistanceType::try_from(distance_type_str.as_str())?;

    let vector_index_params_option_object = env
        .call_method(
            index_params_obj,
            "getVectorIndexParams",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;

    let vector_index_params_option = if env
        .call_method(&vector_index_params_option_object, "isPresent", "()Z", &[])?
        .z()?
    {
        let vector_index_params_obj = env
            .call_method(
                &vector_index_params_option_object,
                "get",
                "()Ljava/lang/Object;",
                &[],
            )?
            .l()?;

        let ivf_params_obj = env
            .call_method(
                &vector_index_params_obj,
                "getIvfParams",
                "()Lcom/lancedb/lance/index/vector/IvfBuildParams;",
                &[],
            )?
            .l()?;

        let mut stages = Vec::new();

        // Parse IvfBuildParams
        let num_partitions =
            env.get_int_as_usize_from_method(&ivf_params_obj, "getNumPartitions")?;
        let max_iters = env.get_int_as_usize_from_method(&ivf_params_obj, "getMaxIters")?;
        let sample_rate = env.get_int_as_usize_from_method(&ivf_params_obj, "getSampleRate")?;
        let shuffle_partition_batches =
            env.get_int_as_usize_from_method(&ivf_params_obj, "getShufflePartitionBatches")?;
        let shuffle_partition_concurrency =
            env.get_int_as_usize_from_method(&ivf_params_obj, "getShufflePartitionConcurrency")?;
        let use_residual = env.get_boolean_from_method(&ivf_params_obj, "useResidual")?;

        let ivf_params = IvfBuildParams {
            num_partitions,
            max_iters,
            sample_rate,
            shuffle_partition_batches,
            shuffle_partition_concurrency,
            use_residual,
            ..Default::default()
        };
        stages.push(StageParams::Ivf(ivf_params));

        // Parse HnswBuildParams
        let hnsw_params = env.get_optional_from_method(
            &vector_index_params_obj,
            "getHnswParams",
            |env, hnsw_obj| {
                let max_level = env.call_method(&hnsw_obj, "getMaxLevel", "()S", &[])?.s()? as u16;
                let m = env.get_int_as_usize_from_method(&hnsw_obj, "getM")?;
                let ef_construction =
                    env.get_int_as_usize_from_method(&hnsw_obj, "getEfConstruction")?;
                let prefetch_distance =
                    env.get_optional_usize_from_method(&hnsw_obj, "getPrefetchDistance")?;

                Ok(HnswBuildParams {
                    max_level,
                    m,
                    ef_construction,
                    prefetch_distance,
                })
            },
        )?;

        if let Some(hnsw_params) = hnsw_params {
            stages.push(StageParams::Hnsw(hnsw_params));
        }

        // Parse PQBuildParams
        let pq_params = env.get_optional_from_method(
            &vector_index_params_obj,
            "getPqParams",
            |env, pq_obj| {
                let num_sub_vectors =
                    env.get_int_as_usize_from_method(&pq_obj, "getNumSubVectors")?;
                let num_bits = env.get_int_as_usize_from_method(&pq_obj, "getNumBits")?;
                let max_iters = env.get_int_as_usize_from_method(&pq_obj, "getMaxIters")?;
                let kmeans_redos = env.get_int_as_usize_from_method(&pq_obj, "getKmeansRedos")?;
                let sample_rate = env.get_int_as_usize_from_method(&pq_obj, "getSampleRate")?;

                Ok(PQBuildParams {
                    num_sub_vectors,
                    num_bits,
                    max_iters,
                    kmeans_redos,
                    sample_rate,
                    ..Default::default()
                })
            },
        )?;

        if let Some(pq_params) = pq_params {
            stages.push(StageParams::PQ(pq_params));
        }

        // Parse SQBuildParams
        let sq_params = env.get_optional_from_method(
            &vector_index_params_obj,
            "getSqParams",
            |env, sq_obj| {
                let num_bits = env.call_method(&sq_obj, "getNumBits", "()S", &[])?.s()? as u16;
                let sample_rate = env.get_int_as_usize_from_method(&sq_obj, "getSampleRate")?;

                Ok(SQBuildParams {
                    num_bits,
                    sample_rate,
                })
            },
        )?;

        if let Some(sq_params) = sq_params {
            stages.push(StageParams::SQ(sq_params));
        }

        Some(VectorIndexParams {
            metric_type: distance_type,
            stages,
            version: IndexFileVersion::V3,
        })
    } else {
        None
    };

    match vector_index_params_option {
        Some(params) => Ok(Box::new(params) as Box<dyn IndexParams>),
        None => Err(Error::input_error(
            "VectorIndexParams not present".to_string(),
        )),
    }
}
