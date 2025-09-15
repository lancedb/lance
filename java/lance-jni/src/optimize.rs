// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, sync::Arc};

use jni::{
    objects::{JByteArray, JMap, JObject, JValue, JValueGen},
    sys::jlong,
    JNIEnv,
};
use lance::dataset::{
    index::DatasetIndexRemapperOptions,
    optimize::{
        commit_compaction, plan_compaction, CompactionMetrics, CompactionOptions, CompactionPlan,
        CompactionTask, IndexRemapperOptions, RewriteResult, TaskData,
    },
};

use crate::{
    blocking_dataset::{BlockingDataset, NATIVE_DATASET},
    traits::{
        export_vec, import_vec_from_method, import_vec_to_rust, FromJObjectWithEnv, IntoJava,
    },
    utils::{
        build_compaction_options, to_java_boolean_obj, to_java_float_obj, to_java_long_obj,
        to_java_optional,
    },
    RT,
};

use crate::error::Result;

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_compaction_Compaction_nativePlanCompaction<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    java_dataset: JObject,                    // Dataset
    target_rows_per_fragment: JObject,        // Optional<Long>
    max_rows_per_group: JObject,              // Optional<Long>
    max_bytes_per_file: JObject,              // Optional<Long>
    materialize_deletions: JObject,           // Optional<Boolean>
    materialize_deletions_threshold: JObject, // Optional<Float>
    num_threads: JObject,                     // Optional<Long>
    batch_size: JObject,                      // Optional<Long>
    defer_index_remap: JObject,               // Optional<Boolean>
) -> JObject<'local> {
    ok_or_throw_with_return!(
        env,
        inner_plan_compaction(
            &mut env,
            java_dataset,
            target_rows_per_fragment,
            max_rows_per_group,
            max_bytes_per_file,
            materialize_deletions,
            materialize_deletions_threshold,
            num_threads,
            batch_size,
            defer_index_remap
        ),
        JObject::null()
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_plan_compaction<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,                    // Dataset
    target_rows_per_fragment: JObject,        // Optional<Long>
    max_rows_per_group: JObject,              // Optional<Long>
    max_bytes_per_file: JObject,              // Optional<Long>
    materialize_deletions: JObject,           // Optional<Boolean>
    materialize_deletions_threshold: JObject, // Optional<Float>
    num_threads: JObject,                     // Optional<Long>
    batch_size: JObject,                      // Optional<Long>
    defer_index_remap: JObject,               // Optional<Boolean>
) -> Result<JObject<'local>> {
    let compaction_options = build_compaction_options(
        env,
        &target_rows_per_fragment,
        &max_rows_per_group,
        &max_bytes_per_file,
        &materialize_deletions,
        &materialize_deletions_threshold,
        &num_threads,
        &batch_size,
        &defer_index_remap,
    )?;

    let plan = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        RT.block_on(plan_compaction(&dataset.inner, &compaction_options))?
    };
    plan.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_compaction_Compaction_nativeCommitCompaction<
    'local,
>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    java_dataset: JObject,                    // Dataset
    rewrite_results: JObject,                 // List<RewriteResult>
    target_rows_per_fragment: JObject,        // Optional<Long>
    max_rows_per_group: JObject,              // Optional<Long>
    max_bytes_per_file: JObject,              // Optional<Long>
    materialize_deletions: JObject,           // Optional<Boolean>
    materialize_deletions_threshold: JObject, // Optional<Float>
    num_threads: JObject,                     // Optional<Long>
    batch_size: JObject,                      // Optional<Long>
    defer_index_remap: JObject,               // Optional<Boolean>
) -> JObject<'local> {
    ok_or_throw_with_return!(
        env,
        inner_commit_compaction(
            &mut env,
            java_dataset,
            rewrite_results,
            target_rows_per_fragment,
            max_rows_per_group,
            max_bytes_per_file,
            materialize_deletions,
            materialize_deletions_threshold,
            num_threads,
            batch_size,
            defer_index_remap,
        ),
        JObject::null()
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_commit_compaction<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,                    // Dataset
    rewrite_results: JObject,                 // List<RewriteResult>
    target_rows_per_fragment: JObject,        // Optional<Long>
    max_rows_per_group: JObject,              // Optional<Long>
    max_bytes_per_file: JObject,              // Optional<Long>
    materialize_deletions: JObject,           // Optional<Boolean>
    materialize_deletions_threshold: JObject, // Optional<Float>
    num_threads: JObject,                     // Optional<Long>
    batch_size: JObject,                      // Optional<Long>
    defer_index_remap: JObject,               // Optional<Boolean>
) -> Result<JObject<'local>> {
    let compaction_options = build_compaction_options(
        env,
        &target_rows_per_fragment,
        &max_rows_per_group,
        &max_bytes_per_file,
        &materialize_deletions,
        &materialize_deletions_threshold,
        &num_threads,
        &batch_size,
        &defer_index_remap,
    )?;
    let completed_tasks = import_vec_to_rust(env, &rewrite_results, |env, rewrite_result| {
        rewrite_result.extract_object(env)
    })?;
    let remap_options: Arc<dyn IndexRemapperOptions> = Arc::new(DatasetIndexRemapperOptions {});
    let committed_metrics = {
        let mut dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        RT.block_on(commit_compaction(
            &mut dataset.inner,
            completed_tasks,
            remap_options,
            &compaction_options,
        ))?
    };
    committed_metrics.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_compaction_CompactionTask_nativeExecute<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,                            // CompactionTask itself
    java_dataset: JObject,                    // Dataset
    task_data: JObject,                       // TaskData
    read_version: jlong,                      // readVersion
    target_rows_per_fragment: JObject,        // Optional<Long>
    max_rows_per_group: JObject,              // Optional<Long>
    max_bytes_per_file: JObject,              // Optional<Long>
    materialize_deletions: JObject,           // Optional<Boolean>
    materialize_deletions_threshold: JObject, // Optional<Float>
    num_threads: JObject,                     // Optional<Long>
    batch_size: JObject,                      // Optional<Long>
    defer_index_remap: JObject,               // Optional<Boolean>
) -> JObject<'local> {
    ok_or_throw_with_return!(
        env,
        inner_execute_task(
            &mut env,
            java_dataset,
            task_data,
            read_version,
            target_rows_per_fragment,
            max_rows_per_group,
            max_bytes_per_file,
            materialize_deletions,
            materialize_deletions_threshold,
            num_threads,
            batch_size,
            defer_index_remap
        ),
        JObject::null()
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_execute_task<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,                    // Dataset
    task_data: JObject,                       // TaskData
    read_version: jlong,                      // readVersion
    target_rows_per_fragment: JObject,        // Optional<Long>
    max_rows_per_group: JObject,              // Optional<Long>
    max_bytes_per_file: JObject,              // Optional<Long>
    materialize_deletions: JObject,           // Optional<Boolean>
    materialize_deletions_threshold: JObject, // Optional<Float>
    num_threads: JObject,                     // Optional<Long>
    batch_size: JObject,                      // Optional<Long>
    defer_index_remap: JObject,               // Optional<Boolean>
) -> Result<JObject<'local>> {
    let task_data: TaskData = task_data.extract_object(env)?;
    let compaction_options = build_compaction_options(
        env,
        &target_rows_per_fragment,
        &max_rows_per_group,
        &max_bytes_per_file,
        &materialize_deletions,
        &materialize_deletions_threshold,
        &num_threads,
        &batch_size,
        &defer_index_remap,
    )?;
    let compaction_task = CompactionTask {
        task: task_data,
        read_version: read_version as u64,
        options: compaction_options,
    };
    let rewrite_result = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        RT.block_on(compaction_task.execute(&dataset.inner))?
    };
    rewrite_result.into_java(env)
}

const TASK_DATA_CLASS: &str = "com/lancedb/lance/compaction/TaskData";
const TASK_DATA_CONSTRUCTOR_SIG: &str = "(Ljava/util/List;)V";
const COMPACTION_METRICS_CLASS: &str = "com/lancedb/lance/compaction/CompactionMetrics";
const COMPACTION_METRICS_CONSTRUCTOR_SIG: &str = "(JJJJ)V";
const COMPACTION_PLAN_CLASS: &str = "com/lancedb/lance/compaction/CompactionPlan";
const COMPACTION_PLAN_CONSTRUCTOR_SIG: &str =
    "(Ljava/util/List;JLcom/lancedb/lance/compaction/CompactionOptions;)V";
const REWRITE_RESULT_CLASS: &str = "com/lancedb/lance/compaction/RewriteResult";
const REWRITE_RESULT_CONSTRUCTOR_SIG: &str = "(Lcom/lancedb/lance/compaction/CompactionMetrics;Ljava/util/List;Ljava/util/List;JLjava/util/Map;[B)V";
const COMPACTION_OPTIONS_CLASS: &str = "com/lancedb/lance/compaction/CompactionOptions";
const COMPACTION_OPTIONS_CONSTRUCTOR_SIG: &str =
    "(Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;)V";

impl IntoJava for &TaskData {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let fragments = export_vec(env, &self.fragments)?;
        Ok(env.new_object(
            TASK_DATA_CLASS,
            TASK_DATA_CONSTRUCTOR_SIG,
            &[JValueGen::Object(&fragments)],
        )?)
    }
}

impl IntoJava for &CompactionMetrics {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        Ok(env.new_object(
            COMPACTION_METRICS_CLASS,
            COMPACTION_METRICS_CONSTRUCTOR_SIG,
            &[
                JValueGen::Long(self.fragments_removed as i64),
                JValueGen::Long(self.fragments_added as i64),
                JValueGen::Long(self.files_removed as i64),
                JValueGen::Long(self.files_added as i64),
            ],
        )?)
    }
}

impl IntoJava for &CompactionOptions {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let target_rows_per_fragment =
            to_java_long_obj(env, Some(self.target_rows_per_fragment as i64))?;
        let target_rows_per_fragment_opt = to_java_optional(env, target_rows_per_fragment)?;
        let max_rows_per_group = to_java_long_obj(env, Some(self.max_rows_per_group as i64))?;
        let max_rows_per_group_opt = to_java_optional(env, max_rows_per_group)?;
        let max_bytes_per_file = to_java_long_obj(env, self.max_bytes_per_file.map(|v| v as i64))?;
        let max_bytes_per_file_opt = to_java_optional(env, max_bytes_per_file)?;
        let materialize_deletions = to_java_boolean_obj(env, Some(self.materialize_deletions))?;
        let materialize_deletions_opt = to_java_optional(env, materialize_deletions)?;
        let materialize_deletions_threshold =
            to_java_float_obj(env, Some(self.materialize_deletions_threshold))?;
        let materialize_deletions_threshold_opt =
            to_java_optional(env, materialize_deletions_threshold)?;
        let num_threads = to_java_long_obj(env, self.num_threads.map(|v| v as i64))?;
        let num_threads_opt = to_java_optional(env, num_threads)?;
        let batch_size = to_java_long_obj(env, self.batch_size.map(|v| v as i64))?;
        let batch_size_opt = to_java_optional(env, batch_size)?;
        let defer_index_remap = to_java_boolean_obj(env, Some(self.defer_index_remap))?;
        let defer_index_remap_opt = to_java_optional(env, defer_index_remap)?;

        Ok(env.new_object(
            COMPACTION_OPTIONS_CLASS,
            COMPACTION_OPTIONS_CONSTRUCTOR_SIG,
            &[
                JValueGen::Object(&target_rows_per_fragment_opt),
                JValueGen::Object(&max_rows_per_group_opt),
                JValueGen::Object(&max_bytes_per_file_opt),
                JValueGen::Object(&materialize_deletions_opt),
                JValueGen::Object(&materialize_deletions_threshold_opt),
                JValueGen::Object(&num_threads_opt),
                JValueGen::Object(&batch_size_opt),
                JValueGen::Object(&defer_index_remap_opt),
            ],
        )?)
    }
}

impl IntoJava for &CompactionPlan {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let tasks = export_vec(env, &self.tasks)?;
        let compaction_options = self.options.into_java(env)?;
        Ok(env.new_object(
            COMPACTION_PLAN_CLASS,
            COMPACTION_PLAN_CONSTRUCTOR_SIG,
            &[
                JValueGen::Object(&tasks),
                JValueGen::Long(self.read_version as i64),
                JValueGen::Object(&compaction_options),
            ],
        )?)
    }
}

impl IntoJava for &RewriteResult {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let metrics = self.metrics.into_java(env)?;
        let new_fragments = export_vec(env, &self.new_fragments)?;
        let original_fragments = export_vec(env, &self.original_fragments)?;
        let changed_row_addrs: JObject<'_> =
            if let Some(changed_row_addrs) = &self.changed_row_addrs {
                env.byte_array_from_slice(changed_row_addrs)?.into()
            } else {
                JObject::null()
            };
        let row_id_map = if let Some(row_id_map) = &self.row_id_map {
            let java_map = env.new_object("java/util/HashMap", "()V", &[])?;
            for (k, v) in row_id_map {
                let k_obj = to_java_long_obj(env, Some(*k as i64))?;
                let v_obj = to_java_long_obj(env, v.map(|val| val as i64))?;
                env.call_method(
                    &java_map,
                    "put",
                    "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
                    &[JValue::Object(&k_obj), JValue::Object(&v_obj)],
                )?;
            }
            java_map
        } else {
            JObject::null()
        };
        Ok(env.new_object(
            REWRITE_RESULT_CLASS,
            REWRITE_RESULT_CONSTRUCTOR_SIG,
            &[
                JValueGen::Object(&metrics),
                JValueGen::Object(&new_fragments),
                JValueGen::Object(&original_fragments),
                JValueGen::Long(self.read_version as i64),
                JValueGen::Object(&row_id_map),
                JValueGen::Object(&changed_row_addrs),
            ],
        )?)
    }
}

impl FromJObjectWithEnv<CompactionMetrics> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<CompactionMetrics> {
        let fragments_removed = env
            .call_method(self, "getFragmentsRemoved", "()J", &[])?
            .j()? as usize;
        let fragments_added = env
            .call_method(self, "getFragmentsAdded", "()J", &[])?
            .j()? as usize;
        let files_removed = env.call_method(self, "getFilesRemoved", "()J", &[])?.j()? as usize;
        let files_added = env.call_method(self, "getFilesAdded", "()J", &[])?.j()? as usize;
        Ok(CompactionMetrics {
            fragments_removed,
            fragments_added,
            files_removed,
            files_added,
        })
    }
}

impl FromJObjectWithEnv<TaskData> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<TaskData> {
        let task_data = import_vec_from_method(env, self, "getFragments", |env, fragment| {
            fragment.extract_object(env)
        })?;
        Ok(TaskData {
            fragments: task_data,
        })
    }
}

impl FromJObjectWithEnv<RewriteResult> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<RewriteResult> {
        let metrics_obj = env
            .call_method(
                self,
                "getMetrics",
                "()Lcom/lancedb/lance/compaction/CompactionMetrics;",
                &[],
            )?
            .l()?;
        let metrics = metrics_obj.extract_object(env)?;
        let new_fragments =
            import_vec_from_method(env, self, "getNewFragments", |env, fragment| {
                fragment.extract_object(env)
            })?;
        let read_version = env.call_method(self, "getReadVersion", "()J", &[])?.j()? as u64;
        let original_fragments =
            import_vec_from_method(env, self, "getOriginalFragments", |env, fragment| {
                fragment.extract_object(env)
            })?;
        let changed_row_addrs_obj: JByteArray<'_> = env
            .call_method(self, "getChangedRowAddrs", "()[B", &[])?
            .l()?
            .into();
        let changed_row_addrs = if changed_row_addrs_obj.is_null() {
            None
        } else {
            Some(env.convert_byte_array(changed_row_addrs_obj)?)
        };
        let row_id_map_obj = env
            .call_method(self, "getRowIdMap", "()Ljava/util/Map;", &[])?
            .l()?;
        let row_id_map = if row_id_map_obj.is_null() {
            None
        } else {
            let row_id_jmap = JMap::from_env(env, &row_id_map_obj)?;
            let mut map = HashMap::new();
            let mut iter = row_id_jmap.iter(env)?;
            while let Some((key, value)) = iter.next(env)? {
                let key: Option<i64> = key.extract_object(env)?;
                let value: Option<i64> = value.extract_object(env)?;
                map.insert(key.unwrap() as u64, value.map(|v| v as u64));
            }
            Some(map)
        };
        Ok(RewriteResult {
            metrics,
            new_fragments,
            read_version,
            original_fragments,
            row_id_map,
            changed_row_addrs,
        })
    }
}
