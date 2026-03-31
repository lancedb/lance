// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use crate::RT;
use crate::blocking_dataset::{BlockingDataset, NATIVE_DATASET};
use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;

use arrow::array::{FixedSizeListArray, Float32Array};
use jni::JNIEnv;
use jni::objects::{JClass, JFloatArray, JObject, JString};
use jni::sys::jfloatArray;
use lance::index::NoopIndexBuildProgress;
use lance::index::vector::utils::get_vector_dim;
use lance_arrow::FixedSizeListArrayExt;
use lance_index::vector::ivf::builder::IvfBuildParams as RustIvfBuildParams;
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::pq::builder::PQBuildParams as RustPQBuildParams;
use lance_linalg::distance::MetricType;

/// Flatten a FixedSizeList<Float32> into a contiguous Vec<f32>.
fn flatten_fixed_size_list_to_f32(arr: &FixedSizeListArray) -> Result<Vec<f32>> {
    let values = arr
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| {
            Error::input_error(format!(
                "Expected FixedSizeList<Float32>, got value type {}",
                arr.value_type()
            ))
        })?;

    Ok(values.values().to_vec())
}

fn build_ivf_params_from_java(
    env: &mut JNIEnv,
    ivf_params_obj: &JObject,
) -> Result<RustIvfBuildParams> {
    let num_partitions = env.get_int_as_usize_from_method(ivf_params_obj, "getNumPartitions")?;
    let max_iters = env.get_int_as_usize_from_method(ivf_params_obj, "getMaxIters")?;
    let sample_rate = env.get_int_as_usize_from_method(ivf_params_obj, "getSampleRate")?;
    let shuffle_partition_batches =
        env.get_int_as_usize_from_method(ivf_params_obj, "getShufflePartitionBatches")?;
    let shuffle_partition_concurrency =
        env.get_int_as_usize_from_method(ivf_params_obj, "getShufflePartitionConcurrency")?;

    Ok(RustIvfBuildParams {
        num_partitions: Some(num_partitions),
        max_iters,
        sample_rate,
        shuffle_partition_batches,
        shuffle_partition_concurrency,
        ..Default::default()
    })
}

fn build_pq_params_from_java(
    env: &mut JNIEnv,
    pq_params_obj: &JObject,
) -> Result<RustPQBuildParams> {
    let num_sub_vectors = env.get_int_as_usize_from_method(pq_params_obj, "getNumSubVectors")?;
    let num_bits = env.get_int_as_usize_from_method(pq_params_obj, "getNumBits")?;
    let max_iters = env.get_int_as_usize_from_method(pq_params_obj, "getMaxIters")?;
    let kmeans_redos = env.get_int_as_usize_from_method(pq_params_obj, "getKmeansRedos")?;
    let sample_rate = env.get_int_as_usize_from_method(pq_params_obj, "getSampleRate")?;

    Ok(RustPQBuildParams {
        num_sub_vectors,
        num_bits,
        max_iters,
        kmeans_redos,
        codebook: None,
        sample_rate,
    })
}

/// Extract a nullable Java `List<Integer>` into `Option<Vec<u32>>`.
fn get_nullable_fragment_ids(env: &mut JNIEnv, obj: &JObject) -> Result<Option<Vec<u32>>> {
    if obj.is_null() {
        return Ok(None);
    }
    let ints = env.get_integers(obj)?;
    Ok(Some(ints.into_iter().map(|i| i as u32).collect()))
}

/// Extract a nullable Java `float[]` into `Option<Vec<f32>>`.
fn get_nullable_ivf_centroids(
    env: &mut JNIEnv,
    centroids_obj: &JObject,
) -> Result<Option<Vec<f32>>> {
    if centroids_obj.is_null() {
        return Ok(None);
    }
    let jarray = unsafe { JFloatArray::from_raw(centroids_obj.as_raw()) };
    let length = env.get_array_length(&jarray)?;
    let mut buffer = vec![0.0f32; length as usize];
    env.get_float_array_region(&jarray, 0, &mut buffer)?;
    Ok(Some(buffer))
}

/// Build an `IvfModel` from a flat float array and known dimension.
fn build_ivf_model_from_centroids(centroids: &[f32], dim: usize) -> Result<IvfModel> {
    let centroids_array = FixedSizeListArray::try_new_from_values(
        Float32Array::from(centroids.to_vec()),
        dim as i32,
    )?;
    Ok(IvfModel::new(centroids_array, None))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_vector_VectorTrainer_nativeTrainIvfCentroids<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    dataset_obj: JObject<'local>,      // org.lance.Dataset
    column_jstr: JString<'local>,      // java.lang.String
    ivf_params_obj: JObject<'local>,   // org.lance.index.vector.IvfBuildParams
    fragment_ids_obj: JObject<'local>, // List<Integer>, nullable
) -> jfloatArray {
    ok_or_throw_with_return!(
        env,
        inner_train_ivf_centroids(
            &mut env,
            dataset_obj,
            column_jstr,
            ivf_params_obj,
            fragment_ids_obj
        )
        .map(|arr| arr.into_raw()),
        JFloatArray::default().into_raw()
    )
}

fn inner_train_ivf_centroids<'local>(
    env: &mut JNIEnv<'local>,
    dataset_obj: JObject<'local>,
    column_jstr: JString<'local>,
    ivf_params_obj: JObject<'local>,
    fragment_ids_obj: JObject<'local>,
) -> Result<JFloatArray<'local>> {
    let column: String = env.get_string(&column_jstr)?.into();
    let ivf_params = build_ivf_params_from_java(env, &ivf_params_obj)?;
    let fragment_ids = get_nullable_fragment_ids(env, &fragment_ids_obj)?;

    let flattened: Vec<f32> = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(dataset_obj, NATIVE_DATASET) }?;
        let dataset = &dataset_guard.inner;

        let dim = get_vector_dim(dataset.schema(), &column)?;
        let metric_type = MetricType::L2;

        let ivf_model = RT.block_on(lance::index::vector::ivf::build_ivf_model(
            dataset,
            &column,
            dim,
            metric_type,
            &ivf_params,
            fragment_ids.as_deref(),
            Arc::new(NoopIndexBuildProgress),
        ))?;

        let centroids = ivf_model
            .centroids
            .ok_or_else(|| Error::runtime_error("IVF model missing centroids".to_string()))?;

        flatten_fixed_size_list_to_f32(&centroids)?
    };

    let jarray = env.new_float_array(flattened.len() as i32)?;
    env.set_float_array_region(&jarray, 0, &flattened)?;
    Ok(jarray)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_vector_VectorTrainer_nativeTrainPqCodebook<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    dataset_obj: JObject<'local>,       // org.lance.Dataset
    column_jstr: JString<'local>,       // java.lang.String
    pq_params_obj: JObject<'local>,     // org.lance.index.vector.PQBuildParams
    ivf_centroids_obj: JObject<'local>, // float[], nullable
    fragment_ids_obj: JObject<'local>,  // List<Integer>, nullable
) -> jfloatArray {
    ok_or_throw_with_return!(
        env,
        inner_train_pq_codebook(
            &mut env,
            dataset_obj,
            column_jstr,
            pq_params_obj,
            ivf_centroids_obj,
            fragment_ids_obj
        )
        .map(|arr| arr.into_raw()),
        JFloatArray::default().into_raw()
    )
}

fn inner_train_pq_codebook<'local>(
    env: &mut JNIEnv<'local>,
    dataset_obj: JObject<'local>,
    column_jstr: JString<'local>,
    pq_params_obj: JObject<'local>,
    ivf_centroids_obj: JObject<'local>,
    fragment_ids_obj: JObject<'local>,
) -> Result<JFloatArray<'local>> {
    let column: String = env.get_string(&column_jstr)?.into();
    let pq_params = build_pq_params_from_java(env, &pq_params_obj)?;
    let fragment_ids = get_nullable_fragment_ids(env, &fragment_ids_obj)?;
    let ivf_centroids_data = get_nullable_ivf_centroids(env, &ivf_centroids_obj)?;

    let flattened: Vec<f32> = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(dataset_obj, NATIVE_DATASET) }?;
        let dataset = &dataset_guard.inner;

        let dim = get_vector_dim(dataset.schema(), &column)?;
        let metric_type = MetricType::L2;

        let ivf_model = ivf_centroids_data
            .as_deref()
            .map(|data| build_ivf_model_from_centroids(data, dim))
            .transpose()?;

        let pq = RT.block_on(lance::index::vector::pq::build_pq_model_in_fragments(
            dataset,
            &column,
            dim,
            metric_type,
            &pq_params,
            ivf_model.as_ref(),
            fragment_ids.as_deref(),
        ))?;

        flatten_fixed_size_list_to_f32(&pq.codebook)?
    };

    let jarray = env.new_float_array(flattened.len() as i32)?;
    env.set_float_array_region(&jarray, 0, &flattened)?;
    Ok(jarray)
}
