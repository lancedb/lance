// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array};
use arrow_schema::{DataType, Field};
use jni::objects::{JFloatArray, JMap, JObject, JString, JValue, JValueGen};
use jni::sys::{jboolean, jfloat, jlong};
use jni::JNIEnv;
use lance::dataset::optimize::CompactionOptions;
use lance::dataset::{WriteMode, WriteParams};
use lance::index::vector::{IndexFileVersion, StageParams, VectorIndexParams};
use lance::io::ObjectStoreParams;
use lance_encoding::version::LanceFileVersion;
use lance_index::vector::bq::RQBuildParams;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_index::IndexParams;
use lance_linalg::distance::DistanceType;

use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;
use crate::storage_options::JavaStorageOptionsProvider;

use crate::traits::FromJObjectWithEnv;
use lance_index::vector::Query;
use lance_io::object_store::StorageOptionsProvider;
use std::collections::HashMap;
use std::str::FromStr;

pub fn extract_storage_options(
    env: &mut JNIEnv,
    storage_options_obj: &JObject,
) -> Result<HashMap<String, String>> {
    let jmap = JMap::from_env(env, storage_options_obj)?;
    let storage_options: HashMap<String, String> = to_rust_map(env, &jmap)?;
    Ok(storage_options)
}

#[allow(clippy::too_many_arguments)]
pub fn extract_write_params(
    env: &mut JNIEnv,
    max_rows_per_file: &JObject,
    max_rows_per_group: &JObject,
    max_bytes_per_file: &JObject,
    mode: &JObject,
    enable_stable_row_ids: &JObject,
    data_storage_version: &JObject,
    enable_v2_manifest_paths: Option<&JObject>,
    storage_options_obj: &JObject,
    storage_options_provider_obj: &JObject, // Optional<StorageOptionsProvider>
    initial_bases: &JObject,                // Optional<BasePath>
    target_bases: &JObject,                 // Optional<String>
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
    if let Some(enable_stable_row_ids_val) = env.get_boolean_opt(enable_stable_row_ids)? {
        write_params.enable_stable_row_ids = enable_stable_row_ids_val;
    }
    if let Some(data_storage_version_val) = env.get_string_opt(data_storage_version)? {
        write_params.data_storage_version = Some(LanceFileVersion::from_str(
            data_storage_version_val.as_str(),
        )?);
    }

    // Enable v2 manifest paths by default.
    write_params.enable_v2_manifest_paths =
        if let Some(enable_v2_manifest_paths) = enable_v2_manifest_paths {
            env.get_boolean_opt(enable_v2_manifest_paths)?
                .unwrap_or(true)
        } else {
            true
        };

    let storage_options: HashMap<String, String> =
        extract_storage_options(env, storage_options_obj)?;

    // Extract storage options provider if present
    let storage_options_provider: Option<Arc<dyn StorageOptionsProvider>> = env
        .get_optional(storage_options_provider_obj, |env, provider_obj| {
            JavaStorageOptionsProvider::new(env, provider_obj)
        })?
        .map(|p| Arc::new(p) as Arc<dyn StorageOptionsProvider>);

    if let Some(initial_bases) =
        env.get_list_opt(initial_bases, |env, elem| elem.extract_object(env))?
    {
        write_params.initial_bases = Some(initial_bases);
    }

    if let Some(names) = env.get_strings_opt(target_bases)? {
        write_params.target_base_names_or_paths = Some(names);
    }

    // Create storage options accessor from storage_options and provider
    let accessor = match (storage_options.is_empty(), storage_options_provider) {
        (false, Some(provider)) => Some(Arc::new(
            lance::io::StorageOptionsAccessor::with_initial_and_provider(storage_options, provider),
        )),
        (false, None) => Some(Arc::new(
            lance::io::StorageOptionsAccessor::with_static_options(storage_options),
        )),
        (true, Some(provider)) => Some(Arc::new(lance::io::StorageOptionsAccessor::with_provider(
            provider,
        ))),
        (true, None) => None,
    };

    write_params.store_params = Some(ObjectStoreParams {
        storage_options_accessor: accessor,
        ..Default::default()
    });
    Ok(write_params)
}

#[allow(clippy::too_many_arguments)]
pub fn build_compaction_options(
    env: &mut JNIEnv,
    target_rows_per_fragment: &JObject,        // Optional<Long>
    max_rows_per_group: &JObject,              // Optional<Long>
    max_bytes_per_file: &JObject,              // Optional<Long>
    materialize_deletions: &JObject,           // Optional<Boolean>
    materialize_deletions_threshold: &JObject, // Optional<Float>
    num_threads: &JObject,                     // Optional<Long>
    batch_size: &JObject,                      // Optional<Long>
    defer_index_remap: &JObject,               // Optional<Boolean>
) -> Result<CompactionOptions> {
    let mut compaction_options = CompactionOptions::default();

    if let Some(target_rows_per_fragment_val) = env.get_long_opt(target_rows_per_fragment)? {
        compaction_options.target_rows_per_fragment = target_rows_per_fragment_val as usize;
    }
    if let Some(max_rows_per_group_val) = env.get_long_opt(max_rows_per_group)? {
        compaction_options.max_rows_per_group = max_rows_per_group_val as usize;
    }
    if let Some(max_bytes_per_file_val) = env.get_long_opt(max_bytes_per_file)? {
        compaction_options.max_bytes_per_file = Some(max_bytes_per_file_val as usize);
    }
    if let Some(materialize_deletions_val) = env.get_boolean_opt(materialize_deletions)? {
        compaction_options.materialize_deletions = materialize_deletions_val;
    }
    if let Some(materialize_deletions_threshold_val) =
        env.get_f32_opt(materialize_deletions_threshold)?
    {
        compaction_options.materialize_deletions_threshold = materialize_deletions_threshold_val;
    }
    if let Some(num_threads_val) = env.get_long_opt(num_threads)? {
        compaction_options.num_threads = Some(num_threads_val as usize);
    }
    if let Some(batch_size_val) = env.get_long_opt(batch_size)? {
        compaction_options.batch_size = Some(batch_size_val as usize);
    }
    if let Some(defer_index_remap_val) = env.get_boolean_opt(defer_index_remap)? {
        compaction_options.defer_index_remap = defer_index_remap_val;
    }

    Ok(compaction_options)
}

// Convert from Java Optional<Query> to Rust Option<Query>
pub fn get_query(env: &mut JNIEnv, query_obj: JObject) -> Result<Option<Query>> {
    let query = env.get_optional(&query_obj, |env, java_obj| {
        let column = env.get_string_from_method(&java_obj, "getColumn")?;
        let key_array = env.get_vec_f32_from_method(&java_obj, "getKey")?;
        let key = Arc::new(Float32Array::from(key_array));

        let k = env.get_int_as_usize_from_method(&java_obj, "getK")?;
        let minimum_nprobes = env.get_int_as_usize_from_method(&java_obj, "getMinimumNprobes")?;
        let maximum_nprobes = env.get_optional_usize_from_method(&java_obj, "getMaximumNprobes")?;

        let ef = env.get_optional_usize_from_method(&java_obj, "getEf")?;

        let refine_factor = env.get_optional_u32_from_method(&java_obj, "getRefineFactor")?;

        let distance_type = if let Some(distance_type_str) =
            env.get_optional_string_from_method(&java_obj, "getDistanceTypeString")?
        {
            Some(DistanceType::try_from(distance_type_str.as_str())?)
        } else {
            None
        };

        let use_index = env.get_boolean_from_method(&java_obj, "isUseIndex")?;

        Ok(Query {
            column,
            key,
            k,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes,
            maximum_nprobes,
            ef,
            refine_factor,
            metric_type: distance_type,
            use_index,
            dist_q_c: 0.0,
        })
    })?;

    Ok(query)
}

pub fn get_vector_index_params(
    env: &mut JNIEnv,
    index_params_obj: JObject,
) -> Result<Box<dyn IndexParams>> {
    let vector_index_params_option = env.get_optional_from_method(
        &index_params_obj,
        "getVectorIndexParams",
        |env, vector_index_params_obj| {
            // Get distance type from VectorIndexParams
            let distance_type_obj: JString = env
                .call_method(
                    &vector_index_params_obj,
                    "getDistanceTypeString",
                    "()Ljava/lang/String;",
                    &[],
                )?
                .l()?
                .into();
            let distance_type_str: String = env.get_string(&distance_type_obj)?.into();
            let distance_type = DistanceType::try_from(distance_type_str.as_str())?;

            let ivf_params_obj = env
                .call_method(
                    &vector_index_params_obj,
                    "getIvfParams",
                    "()Lorg/lance/index/vector/IvfBuildParams;",
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
            let shuffle_partition_concurrency = env
                .get_int_as_usize_from_method(&ivf_params_obj, "getShufflePartitionConcurrency")?;

            let mut ivf_params = IvfBuildParams {
                num_partitions: Some(num_partitions),
                max_iters,
                sample_rate,
                shuffle_partition_batches,
                shuffle_partition_concurrency,
                ..Default::default()
            };

            // Optional pre-trained IVF centroids from Java IvfBuildParams
            // Method signature: float[] getCentroids()
            let centroids_obj = env
                .call_method(&ivf_params_obj, "getCentroids", "()[F", &[])?
                .l()?;

            if !centroids_obj.is_null() {
                let jarray: JFloatArray = centroids_obj.into();
                let length = env.get_array_length(&jarray)?;
                if length > 0 {
                    if !(length as usize).is_multiple_of(num_partitions) {
                        return Err(Error::input_error(format!(
                            "Invalid IVF centroids: length {} is not divisible by num_partitions {}",
                            length, num_partitions
                        )));
                    }
                    let mut buffer = vec![0.0f32; length as usize];
                    env.get_float_array_region(&jarray, 0, &mut buffer)?;
                    let dimension = buffer.len() / num_partitions;

                    let values = Float32Array::from(buffer);
                    let fsl = FixedSizeListArray::try_new(
                        Arc::new(Field::new("item", DataType::Float32, false)),
                        dimension as i32,
                        Arc::new(values) as ArrayRef,
                        None,
                    )
                    .map_err(|e| {
                        Error::input_error(format!(
                            "Failed to construct FixedSizeListArray for IVF centroids: {e}"
                        ))
                    })?;

                    ivf_params.centroids = Some(Arc::new(fsl));
                }
            }

            stages.push(StageParams::Ivf(ivf_params));

            // Parse HnswBuildParams
            let hnsw_params = env.get_optional_from_method(
                &vector_index_params_obj,
                "getHnswParams",
                |env, hnsw_obj| {
                    let max_level =
                        env.call_method(&hnsw_obj, "getMaxLevel", "()S", &[])?.s()? as u16;
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
                    let kmeans_redos =
                        env.get_int_as_usize_from_method(&pq_obj, "getKmeansRedos")?;
                    let sample_rate = env.get_int_as_usize_from_method(&pq_obj, "getSampleRate")?;

                    // Optional pre-trained PQ codebook from Java PQBuildParams
                    // Method signature: float[] getCodebook()
                    let codebook_obj = env
                        .call_method(&pq_obj, "getCodebook", "()[F", &[])?
                        .l()?;

                    let codebook = if !codebook_obj.is_null() {
                        let jarray: JFloatArray = codebook_obj.into();
                        let length = env.get_array_length(&jarray)?;
                        if length > 0 {
                            let mut buffer = vec![0.0f32; length as usize];
                            env.get_float_array_region(&jarray, 0, &mut buffer)?;
                            let values = Float32Array::from(buffer);
                            Some(Arc::new(values) as _)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    Ok(PQBuildParams {
                        num_sub_vectors,
                        num_bits,
                        max_iters,
                        kmeans_redos,
                        codebook,
                        sample_rate,
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

            // Parse RQBuildParams
            let rq_params = env.get_optional_from_method(
                &vector_index_params_obj,
                "getRqParams",
                |env, rq_obj| {
                    let num_bits = env.call_method(&rq_obj, "getNumBits", "()B", &[])?.b()? as u8;
                    Ok(RQBuildParams::new(num_bits))
                },
            )?;

            if let Some(rq_params) = rq_params {
                stages.push(StageParams::RQ(rq_params));
            }

            Ok(VectorIndexParams {
                metric_type: distance_type,
                stages,
                version: IndexFileVersion::V3,
            })
        },
    )?;

    match vector_index_params_option {
        Some(params) => Ok(Box::new(params) as Box<dyn IndexParams>),
        None => Err(Error::input_error(
            "VectorIndexParams not present".to_string(),
        )),
    }
}

pub fn get_scalar_index_params(
    env: &mut JNIEnv,
    index_params_obj: JObject,
) -> Result<(String, Option<String>)> {
    env.get_optional_from_method(
        &index_params_obj,
        "getScalarIndexParams",
        |env, scalar_params_obj| {
            let index_type = env.get_string_from_method(&scalar_params_obj, "getIndexType")?;

            let params = env.get_optional_from_method(
                &scalar_params_obj,
                "getJsonParams",
                |env, params_obj| {
                    let params_str: JString = params_obj.into();
                    let params_string: String = env.get_string(&params_str)?.into();
                    Ok(params_string)
                },
            )?;

            Ok((index_type, params))
        },
    )?
    .ok_or_else(|| Error::input_error("ScalarIndexParams not present".to_string()))
}

pub fn to_rust_map(env: &mut JNIEnv, jmap: &JMap) -> Result<HashMap<String, String>> {
    env.with_local_frame(16, |env| {
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
    })
}

pub fn to_java_map<'local>(
    env: &mut JNIEnv<'local>,
    map: &HashMap<String, String>,
) -> Result<JObject<'local>> {
    let java_map = env.new_object("java/util/HashMap", "()V", &[])?;
    for (k, v) in map {
        let jkey = env.new_string(k)?;
        let jval = env.new_string(v)?;
        env.call_method(
            &java_map,
            "put",
            "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
            &[JValue::Object(&jkey), JValue::Object(&jval)],
        )?;
    }
    Ok(java_map)
}

pub fn to_java_list<'local>(
    env: &mut JNIEnv<'local>,
    list: &Vec<JObject>,
) -> Result<JObject<'local>> {
    let java_list = env.new_object("java/util/ArrayList", "()V", &[])?;
    for item in list {
        env.call_method(
            &java_list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValue::Object(item)],
        )?;
    }
    Ok(java_list)
}

pub fn to_java_optional<'local>(
    env: &mut JNIEnv<'local>,
    value: JObject,
) -> Result<JObject<'local>> {
    Ok(env
        .call_static_method(
            "java/util/Optional",
            "ofNullable",
            "(Ljava/lang/Object;)Ljava/util/Optional;",
            &[JValueGen::Object(&value)],
        )?
        .l()?)
}

pub fn to_java_long_obj<'local>(
    env: &mut JNIEnv<'local>,
    value: Option<i64>,
) -> Result<JObject<'local>> {
    match value {
        Some(base_index) => Ok(env.new_object(
            "java/lang/Long",
            "(J)V",
            &[JValue::Long(base_index as jlong)],
        )?),
        None => Ok(JObject::null()),
    }
}

pub fn to_java_boolean_obj<'local>(
    env: &mut JNIEnv<'local>,
    value: Option<bool>,
) -> Result<JObject<'local>> {
    match value {
        Some(base_index) => Ok(env.new_object(
            "java/lang/Boolean",
            "(Z)V",
            &[JValue::Bool(base_index as jboolean)],
        )?),
        None => Ok(JObject::null()),
    }
}

pub fn to_java_float_obj<'local>(
    env: &mut JNIEnv<'local>,
    value: Option<f32>,
) -> Result<JObject<'local>> {
    match value {
        Some(base_index) => Ok(env.new_object(
            "java/lang/Float",
            "(F)V",
            &[JValue::Float(base_index as jfloat)],
        )?),
        None => Ok(JObject::null()),
    }
}
