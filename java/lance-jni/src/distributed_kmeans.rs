// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! JNI shim for the distributed IVF centroid-training primitives.
//!
//! Mirrors `python/src/indices.rs`. Native methods take Arrow-IPC `byte[]` for
//! `RecordBatch` arguments (centroids, partial stats, samples) and return
//! either an IPC `byte[]` (for stats) or a flat `float[]` (for centroids).

use crate::RT;
use crate::blocking_dataset::{BlockingDataset, NATIVE_DATASET};
use crate::error::{Error, Result};

use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, RecordBatch};
use jni::JNIEnv;
use jni::objects::{JByteArray, JClass, JIntArray, JObject, JObjectArray, JString};
use jni::sys::{jbyteArray, jfloatArray};

use lance::index::vector::ivf::distributed as l2;
use lance_index::vector::kmeans::distributed as l1;
use lance_linalg::distance::DistanceType;

fn parse_distance_type(env: &mut JNIEnv, s: &JString) -> Result<DistanceType> {
    let raw: String = env.get_string(s)?.into();
    DistanceType::try_from(raw.as_str()).map_err(|e| Error::input_error(e.to_string()))
}

fn arrow_err(e: arrow::error::ArrowError) -> Error {
    Error::input_error(e.to_string())
}

fn record_batch_to_ipc(batch: &RecordBatch) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &batch.schema()).map_err(arrow_err)?;
        writer.write(batch).map_err(arrow_err)?;
        writer.finish().map_err(arrow_err)?;
    }
    Ok(buf)
}

fn ipc_to_record_batch(env: &mut JNIEnv, jba: &JByteArray) -> Result<RecordBatch> {
    let bytes = env.convert_byte_array(jba)?;
    let mut reader = StreamReader::try_new(std::io::Cursor::new(bytes), None).map_err(arrow_err)?;
    reader
        .next()
        .ok_or_else(|| Error::input_error("empty IPC stream".to_string()))?
        .map_err(arrow_err)
}

fn ipc_to_centroids_fsl(env: &mut JNIEnv, jba: &JByteArray) -> Result<FixedSizeListArray> {
    let batch = ipc_to_record_batch(env, jba)?;
    if batch.num_columns() != 1 {
        return Err(Error::input_error(format!(
            "centroids IPC must have a single column, got {}",
            batch.num_columns()
        )));
    }
    Ok(batch
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| Error::input_error("centroids column must be FixedSizeList".to_string()))?
        .clone())
}

fn fsl_to_jfloat_array<'a>(
    env: &mut JNIEnv<'a>,
    fsl: &FixedSizeListArray,
) -> Result<jni::objects::JFloatArray<'a>> {
    let values = fsl.values().as_primitive::<Float32Type>().values().to_vec();
    let arr = env.new_float_array(values.len() as i32)?;
    env.set_float_array_region(&arr, 0, &values)?;
    Ok(arr)
}

fn read_optional_fragment_ids(env: &mut JNIEnv, arr: &JIntArray) -> Result<Option<Vec<u32>>> {
    if arr.is_null() {
        return Ok(None);
    }
    let len = env.get_array_length(arr)? as usize;
    let mut buf = vec![0i32; len];
    env.get_int_array_region(arr, 0, &mut buf)?;
    Ok(Some(buf.into_iter().map(|x| x as u32).collect()))
}

fn read_byte_array_2d(env: &mut JNIEnv, arr: &JObjectArray) -> Result<Vec<RecordBatch>> {
    let len = env.get_array_length(arr)? as usize;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let element = env.get_object_array_element(arr, i as i32)?;
        let jba: JByteArray = JByteArray::from(element);
        out.push(ipc_to_record_batch(env, &jba)?);
    }
    Ok(out)
}

trait Pipe: Sized {
    fn pipe<R, F: FnOnce(Self) -> R>(self, f: F) -> R {
        f(self)
    }
}
impl<T> Pipe for T {}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_vector_DistributedKMeans_nativeSampleRound0<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    dataset_obj: JObject<'local>,
    column_jstr: JString<'local>,
    target: i64,
    distance_type_jstr: JString<'local>,
    rng_seed: i64,
    fragment_ids_arr: JIntArray<'local>,
) -> jbyteArray {
    let mut inner = || -> Result<Vec<u8>> {
        let column: String = env.get_string(&column_jstr)?.into();
        let dt = parse_distance_type(&mut env, &distance_type_jstr)?;
        let frags = read_optional_fragment_ids(&mut env, &fragment_ids_arr)?;
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&dataset_obj, NATIVE_DATASET) }?;
        let batch = RT.block_on(l2::sample_round_0(
            &dataset_guard.inner,
            &column,
            frags.as_deref(),
            target as usize,
            dt,
            rng_seed as u64,
        ))?;
        record_batch_to_ipc(&batch)
    };
    crate::ok_or_throw_with_return!(env, inner(), JByteArray::default().into_raw()).pipe(|bytes| {
        match env.byte_array_from_slice(&bytes) {
            Ok(arr) => arr.into_raw(),
            Err(e) => {
                let _ = env.throw_new("java/lang/RuntimeException", e.to_string());
                JByteArray::default().into_raw()
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_vector_DistributedKMeans_nativeComputePartialStats<
    'local,
>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    dataset_obj: JObject<'local>,
    column_jstr: JString<'local>,
    centroids_ipc: JByteArray<'local>,
    distance_type_jstr: JString<'local>,
    fragment_ids_arr: JIntArray<'local>,
) -> jbyteArray {
    let mut inner = || -> Result<Vec<u8>> {
        let column: String = env.get_string(&column_jstr)?.into();
        let dt = parse_distance_type(&mut env, &distance_type_jstr)?;
        let frags = read_optional_fragment_ids(&mut env, &fragment_ids_arr)?;
        let centroids = ipc_to_centroids_fsl(&mut env, &centroids_ipc)?;
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&dataset_obj, NATIVE_DATASET) }?;
        let stats = RT.block_on(l2::compute_partial_stats(
            &dataset_guard.inner,
            &column,
            frags.as_deref(),
            &centroids,
            dt,
        ))?;
        record_batch_to_ipc(stats.record_batch())
    };
    crate::ok_or_throw_with_return!(env, inner(), JByteArray::default().into_raw()).pipe(|bytes| {
        match env.byte_array_from_slice(&bytes) {
            Ok(arr) => arr.into_raw(),
            Err(e) => {
                let _ = env.throw_new("java/lang/RuntimeException", e.to_string());
                JByteArray::default().into_raw()
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_vector_DistributedKMeans_nativeMergePartialStats<
    'local,
>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    a_ipc: JByteArray<'local>,
    b_ipc: JByteArray<'local>,
) -> jbyteArray {
    let mut inner = || -> Result<Vec<u8>> {
        let a = l1::PartialStats::from_record_batch(ipc_to_record_batch(&mut env, &a_ipc)?)?;
        let b = l1::PartialStats::from_record_batch(ipc_to_record_batch(&mut env, &b_ipc)?)?;
        let merged = l1::merge_partial_stats(a, b)?;
        record_batch_to_ipc(merged.record_batch())
    };
    crate::ok_or_throw_with_return!(env, inner(), JByteArray::default().into_raw()).pipe(|bytes| {
        match env.byte_array_from_slice(&bytes) {
            Ok(arr) => arr.into_raw(),
            Err(e) => {
                let _ = env.throw_new("java/lang/RuntimeException", e.to_string());
                JByteArray::default().into_raw()
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_vector_DistributedKMeans_nativeReducePartialStats<
    'local,
>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    stats_arr: JObjectArray<'local>,
) -> jbyteArray {
    let mut inner = || -> Result<Vec<u8>> {
        let batches = read_byte_array_2d(&mut env, &stats_arr)?;
        let mut parsed = Vec::with_capacity(batches.len());
        for b in batches {
            parsed.push(l1::PartialStats::from_record_batch(b)?);
        }
        let merged = l1::reduce_partial_stats(parsed)?;
        record_batch_to_ipc(merged.record_batch())
    };
    crate::ok_or_throw_with_return!(env, inner(), JByteArray::default().into_raw()).pipe(|bytes| {
        match env.byte_array_from_slice(&bytes) {
            Ok(arr) => arr.into_raw(),
            Err(e) => {
                let _ = env.throw_new("java/lang/RuntimeException", e.to_string());
                JByteArray::default().into_raw()
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_vector_DistributedKMeans_nativeFinalizeCentroids<
    'local,
>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    stats_ipc: JByteArray<'local>,
    prev_ipc: JByteArray<'local>,
) -> jfloatArray {
    let mut inner = || -> Result<FixedSizeListArray> {
        let stats =
            l1::PartialStats::from_record_batch(ipc_to_record_batch(&mut env, &stats_ipc)?)?;
        let prev = ipc_to_centroids_fsl(&mut env, &prev_ipc)?;
        Ok(l1::finalize_centroids(&stats, &prev)?)
    };
    let fsl = crate::ok_or_throw_with_return!(
        env,
        inner(),
        jni::objects::JFloatArray::default().into_raw()
    );
    match fsl_to_jfloat_array(&mut env, &fsl) {
        Ok(arr) => arr.into_raw(),
        Err(e) => {
            e.throw(&mut env);
            jni::objects::JFloatArray::default().into_raw()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_vector_DistributedKMeans_nativeSelectInitialCentroids<
    'local,
>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    samples_arr: JObjectArray<'local>,
    k: i64,
    rng_seed: i64,
) -> jfloatArray {
    let mut inner = || -> Result<FixedSizeListArray> {
        let batches = read_byte_array_2d(&mut env, &samples_arr)?;
        Ok(l1::select_initial_centroids(
            batches,
            k as usize,
            rng_seed as u64,
        )?)
    };
    let fsl = crate::ok_or_throw_with_return!(
        env,
        inner(),
        jni::objects::JFloatArray::default().into_raw()
    );
    match fsl_to_jfloat_array(&mut env, &fsl) {
        Ok(arr) => arr.into_raw(),
        Err(e) => {
            e.throw(&mut env);
            jni::objects::JFloatArray::default().into_raw()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_vector_DistributedKMeans_nativeBootstrapCentroids<
    'local,
>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    samples_arr: JObjectArray<'local>,
    k: i64,
    distance_type_jstr: JString<'local>,
    rng_seed: i64,
) -> jfloatArray {
    let mut inner = || -> Result<FixedSizeListArray> {
        let dt = parse_distance_type(&mut env, &distance_type_jstr)?;
        let batches = read_byte_array_2d(&mut env, &samples_arr)?;
        Ok(l1::bootstrap_centroids(
            batches,
            k as usize,
            dt,
            rng_seed as u64,
        )?)
    };
    let fsl = crate::ok_or_throw_with_return!(
        env,
        inner(),
        jni::objects::JFloatArray::default().into_raw()
    );
    match fsl_to_jfloat_array(&mut env, &fsl) {
        Ok(arr) => arr.into_raw(),
        Err(e) => {
            e.throw(&mut env);
            jni::objects::JFloatArray::default().into_raw()
        }
    }
}
