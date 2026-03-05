// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use crate::RT;
use crate::blocking_dataset::{BlockingDataset, NATIVE_DATASET};
use crate::dispatcher::{DISPATCHER, DispatcherMessage};
use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;
use crate::task_tracker::{TASK_TRACKER, TaskInfo};
use crate::traits::import_vec_from_method;
use arrow::array::Float32Array;
use arrow::ffi::FFI_ArrowSchema;
use jni::JNIEnv;
use jni::objects::{JObject, JString};
use jni::sys::{JNI_TRUE, jboolean, jint, jlong};
use lance::dataset::scanner::{AggregateExpr, ColumnOrdering, Scanner};
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::query::{
    BooleanQuery as FtsBooleanQuery, BoostQuery as FtsBoostQuery, FtsQuery,
    MatchQuery as FtsMatchQuery, MultiMatchQuery as FtsMultiMatchQuery, Occur as FtsOccur,
    PhraseQuery as FtsPhraseQuery,
};
use lance_io::ffi::to_ffi_arrow_array_stream;
use lance_linalg::distance::DistanceType;

pub const NATIVE_ASYNC_SCANNER: &str = "nativeAsyncScannerHandle";

/// Async scanner that spawns Tokio tasks for non-blocking I/O
pub struct AsyncScanner {
    pub(crate) inner: Arc<Scanner>,
}

impl AsyncScanner {
    pub fn create(scanner: Scanner) -> Self {
        Self {
            inner: Arc::new(scanner),
        }
    }

    /// Start an async scan task
    pub fn start_scan(&self, task_id: u64, scanner_global_ref: jni::objects::GlobalRef) {
        let scanner = self.inner.clone();

        // Spawn Tokio task for async I/O
        let handle = RT.spawn(async move {
            let result = match scanner.try_into_stream().await {
                Ok(stream) => {
                    // Convert to FFI pointer
                    match to_ffi_arrow_array_stream(stream, RT.handle().clone()) {
                        Ok(ffi_stream) => {
                            let ptr = Box::into_raw(Box::new(ffi_stream)) as i64;
                            (ptr, None)
                        }
                        Err(e) => (-1, Some(e.to_string())),
                    }
                }
                Err(e) => (-1, Some(e.to_string())),
            };

            // Remove from task tracker and send to dispatcher
            if let Some(info) = TASK_TRACKER.complete(task_id).await {
                let dispatcher = DISPATCHER.get().expect("Dispatcher not initialized");
                let _ = dispatcher.send(DispatcherMessage {
                    scanner_global_ref: info.scanner_global_ref,
                    task_id,
                    result_ptr: result.0,
                    error_msg: result.1,
                });
            }
        });

        // Register task
        RT.block_on(async {
            TASK_TRACKER
                .register(
                    task_id,
                    TaskInfo {
                        scanner_global_ref,
                        cancel_handle: handle,
                    },
                )
                .await;
        });
    }
}

// Helper function to build FTS query (copied from blocking_scanner.rs)
fn build_full_text_search_query<'a>(env: &mut JNIEnv<'a>, java_obj: JObject) -> Result<FtsQuery> {
    let type_obj = env
        .call_method(
            &java_obj,
            "getType",
            "()Lorg/lance/ipc/FullTextQuery$Type;",
            &[],
        )?
        .l()?;
    let type_name = env.get_string_from_method(&type_obj, "name")?;

    match type_name.as_str() {
        "MATCH" => {
            let query_text = env.get_string_from_method(&java_obj, "getQueryText")?;
            let column = env.get_string_from_method(&java_obj, "getColumn")?;
            let boost = env.get_f32_from_method(&java_obj, "getBoost")?;
            let fuzziness = env.get_optional_u32_from_method(&java_obj, "getFuzziness")?;
            let max_expansions = env.get_int_as_usize_from_method(&java_obj, "getMaxExpansions")?;
            let operator = env.get_fts_operator_from_method(&java_obj)?;
            let prefix_length = env.get_u32_from_method(&java_obj, "getPrefixLength")?;

            let mut query = FtsMatchQuery::new(query_text);
            query = query.with_column(Some(column));
            query = query
                .with_boost(boost)
                .with_fuzziness(fuzziness)
                .with_max_expansions(max_expansions)
                .with_operator(operator)
                .with_prefix_length(prefix_length);

            Ok(FtsQuery::Match(query))
        }
        "MATCH_PHRASE" => {
            let query_text = env.get_string_from_method(&java_obj, "getQueryText")?;
            let column = env.get_string_from_method(&java_obj, "getColumn")?;
            let slop = env.get_u32_from_method(&java_obj, "getSlop")?;

            let mut query = FtsPhraseQuery::new(query_text);
            query = query.with_column(Some(column));
            query = query.with_slop(slop);

            Ok(FtsQuery::Phrase(query))
        }
        "MULTI_MATCH" => {
            let query_text = env.get_string_from_method(&java_obj, "getQueryText")?;
            let columns: Vec<String> =
                import_vec_from_method(env, &java_obj, "getColumns", |env, elem| {
                    let jstr = JString::from(elem);
                    let value: String = env.get_string(&jstr)?.into();
                    Ok(value)
                })?;

            let boosts: Option<Vec<f32>> =
                env.get_optional_from_method(&java_obj, "getBoosts", |env, list_obj| {
                    crate::traits::import_vec_to_rust(env, &list_obj, |env, elem| {
                        env.get_f32_from_method(&elem, "floatValue")
                    })
                })?;
            let operator = env.get_fts_operator_from_method(&java_obj)?;

            let mut query = FtsMultiMatchQuery::try_new(query_text, columns)?;
            if let Some(boosts) = boosts {
                query = query.try_with_boosts(boosts)?;
            }
            query = query.with_operator(operator);

            Ok(FtsQuery::MultiMatch(query))
        }
        "BOOST" => {
            let positive_obj = env
                .call_method(
                    &java_obj,
                    "getPositive",
                    "()Lorg/lance/ipc/FullTextQuery;",
                    &[],
                )?
                .l()?;
            if positive_obj.is_null() {
                return Err(Error::input_error(
                    "positive query must not be null in BOOST FullTextQuery".to_string(),
                ));
            }
            let negative_obj = env
                .call_method(
                    &java_obj,
                    "getNegative",
                    "()Lorg/lance/ipc/FullTextQuery;",
                    &[],
                )?
                .l()?;
            if negative_obj.is_null() {
                return Err(Error::input_error(
                    "negative query must not be null in BOOST FullTextQuery".to_string(),
                ));
            }

            let positive = build_full_text_search_query(env, positive_obj)?;
            let negative = build_full_text_search_query(env, negative_obj)?;
            let negative_boost = env.get_f32_from_method(&java_obj, "getNegativeBoost")?;

            let query = FtsBoostQuery::new(positive, negative, Some(negative_boost));
            Ok(FtsQuery::Boost(query))
        }
        "BOOLEAN" => {
            let clauses: Vec<(FtsOccur, FtsQuery)> =
                import_vec_from_method(env, &java_obj, "getClauses", |env, clause_obj| {
                    let occur = env.get_occur_from_method(&clause_obj)?;

                    let query_obj = env
                        .call_method(
                            &clause_obj,
                            "getQuery",
                            "()Lorg/lance/ipc/FullTextQuery;",
                            &[],
                        )?
                        .l()?;
                    if query_obj.is_null() {
                        return Err(Error::input_error(
                            "BooleanClause query must not be null".to_string(),
                        ));
                    }
                    let query = build_full_text_search_query(env, query_obj)?;
                    Ok((occur, query))
                })?;

            let boolean_query = FtsBooleanQuery::new(clauses);
            Ok(FtsQuery::Boolean(boolean_query))
        }
        other => Err(Error::input_error(format!(
            "Unsupported FullTextQuery type: {}",
            other
        ))),
    }
}

// JNI Exports

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_ipc_AsyncScanner_createAsyncScanner<'local>(
    mut env: JNIEnv<'local>,
    _class: JObject,
    jdataset: JObject,
    fragment_ids_obj: JObject,
    columns_obj: JObject,
    substrait_filter_obj: JObject,
    filter_obj: JObject,
    batch_size_obj: JObject,
    limit_obj: JObject,
    offset_obj: JObject,
    query_obj: JObject,
    fts_query_obj: JObject,
    with_row_id: jboolean,
    with_row_address: jboolean,
    batch_readahead: jint,
    column_orderings: JObject,
    use_scalar_index: jboolean,
    substrait_aggregate_obj: JObject,
) -> JObject<'local> {
    crate::ok_or_throw!(
        env,
        inner_create_async_scanner(
            &mut env,
            jdataset,
            fragment_ids_obj,
            columns_obj,
            substrait_filter_obj,
            filter_obj,
            batch_size_obj,
            limit_obj,
            offset_obj,
            query_obj,
            fts_query_obj,
            with_row_id,
            with_row_address,
            batch_readahead,
            column_orderings,
            use_scalar_index,
            substrait_aggregate_obj,
        )
    )
}

fn inner_create_async_scanner<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
    fragment_ids_obj: JObject,
    columns_obj: JObject,
    substrait_filter_obj: JObject,
    filter_obj: JObject,
    batch_size_obj: JObject,
    limit_obj: JObject,
    offset_obj: JObject,
    query_obj: JObject,
    fts_query_obj: JObject,
    with_row_id: jboolean,
    with_row_address: jboolean,
    batch_readahead: jint,
    column_orderings: JObject,
    use_scalar_index: jboolean,
    substrait_aggregate_obj: JObject,
) -> Result<JObject<'local>> {
    // Reuse scanner building logic from blocking_scanner.rs
    let fragment_ids_opt = env.get_ints_opt(&fragment_ids_obj)?;
    let dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;

    let mut scanner = dataset_guard.inner.scan();

    // handle fragment_ids
    if let Some(fragment_ids) = fragment_ids_opt {
        let mut fragments = Vec::with_capacity(fragment_ids.len());
        for fragment_id in fragment_ids {
            let Some(fragment) = dataset_guard.inner.get_fragment(fragment_id as usize) else {
                return Err(Error::input_error(format!(
                    "Fragment {fragment_id} not found"
                )));
            };
            fragments.push(fragment.metadata().clone());
        }
        scanner.with_fragments(fragments);
    }
    drop(dataset_guard);

    let columns_opt = env.get_strings_opt(&columns_obj)?;
    if let Some(columns) = columns_opt {
        scanner.project(&columns)?;
    };

    let substrait_opt = env.get_bytes_opt(&substrait_filter_obj)?;
    if let Some(substrait) = substrait_opt {
        RT.block_on(async { scanner.filter_substrait(substrait) })?;
    }

    let filter_opt = env.get_string_opt(&filter_obj)?;
    if let Some(filter) = filter_opt {
        scanner.filter(filter.as_str())?;
    }

    let batch_size_opt = env.get_long_opt(&batch_size_obj)?;
    if let Some(batch_size) = batch_size_opt {
        scanner.batch_size(batch_size as usize);
    }

    let limit_opt = env.get_long_opt(&limit_obj)?;
    let offset_opt = env.get_long_opt(&offset_obj)?;
    scanner
        .limit(limit_opt, offset_opt)
        .map_err(|err| Error::input_error(err.to_string()))?;

    if with_row_id == JNI_TRUE {
        scanner.with_row_id();
    }

    if with_row_address == JNI_TRUE {
        scanner.with_row_address();
    }

    scanner.use_scalar_index(use_scalar_index == JNI_TRUE);

    env.get_optional(&query_obj, |env, java_obj| {
        // Set column and key for nearest search
        let column = env.get_string_from_method(&java_obj, "getColumn")?;
        let key_array = env.get_vec_f32_from_method(&java_obj, "getKey")?;
        let key = Float32Array::from(key_array);
        let k = env.get_int_as_usize_from_method(&java_obj, "getK")?;
        let _ = scanner.nearest(&column, &key, k);

        let minimum_nprobes = env.get_int_as_usize_from_method(&java_obj, "getMinimumNprobes")?;
        scanner.minimum_nprobes(minimum_nprobes);

        let maximum_nprobes = env.get_optional_usize_from_method(&java_obj, "getMaximumNprobes")?;
        if let Some(maximum_nprobes) = maximum_nprobes {
            scanner.maximum_nprobes(maximum_nprobes);
        }

        if let Some(ef) = env.get_optional_usize_from_method(&java_obj, "getEf")? {
            scanner.ef(ef);
        }

        if let Some(refine_factor) =
            env.get_optional_u32_from_method(&java_obj, "getRefineFactor")?
        {
            scanner.refine(refine_factor);
        }

        if let Some(distance_type_str) =
            env.get_optional_string_from_method(&java_obj, "getDistanceTypeString")?
        {
            let distance_type = DistanceType::try_from(distance_type_str.as_str())?;
            scanner.distance_metric(distance_type);
        }

        let use_index = env.get_boolean_from_method(&java_obj, "isUseIndex")?;
        scanner.use_index(use_index);
        Ok(())
    })?;

    env.get_optional(&fts_query_obj, |env, java_obj| {
        let fts_query = build_full_text_search_query(env, java_obj)?;
        let full_text_query = FullTextSearchQuery::new_query(fts_query);
        scanner.full_text_search(full_text_query)?;
        Ok(())
    })?;

    scanner.batch_readahead(batch_readahead as usize);

    env.get_optional(&column_orderings, |env, java_obj| {
        let list = env.get_list(&java_obj)?;
        let mut iter = list.iter(env)?;
        let mut results = Vec::with_capacity(list.size(env)? as usize);
        while let Some(elem) = iter.next(env)? {
            let column_name = env.get_string_from_method(&elem, "getColumnName")?;
            let nulls_first = env.get_boolean_from_method(&elem, "isNullFirst")?;
            let ascending = env.get_boolean_from_method(&elem, "isAscending")?;
            let col_order = ColumnOrdering {
                ascending,
                nulls_first,
                column_name,
            };
            results.push(col_order)
        }
        scanner.order_by(Some(results))?;
        Ok(())
    })?;

    let substrait_aggregate_opt = env.get_bytes_opt(&substrait_aggregate_obj)?;
    if let Some(substrait_aggregate) = substrait_aggregate_opt {
        scanner.aggregate(AggregateExpr::substrait(substrait_aggregate))?;
    }

    let async_scanner = AsyncScanner::create(scanner);

    // Create Java AsyncScanner object
    let j_scanner = env.new_object("org/lance/ipc/AsyncScanner", "()V", &[])?;

    // Attach native handle
    unsafe { env.set_rust_field(&j_scanner, NATIVE_ASYNC_SCANNER, async_scanner)? };

    Ok(j_scanner)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_ipc_AsyncScanner_nativeStartScan(
    mut env: JNIEnv,
    j_scanner: JObject,
    task_id: jlong,
) {
    ok_or_throw_without_return!(env, inner_start_scan(&mut env, j_scanner, task_id as u64));
}

fn inner_start_scan(env: &mut JNIEnv, j_scanner: JObject, task_id: u64) -> Result<()> {
    // Create global reference first, before borrowing scanner
    let scanner_global_ref = env.new_global_ref(&j_scanner)?;

    let scanner_guard =
        unsafe { env.get_rust_field::<_, _, AsyncScanner>(&j_scanner, NATIVE_ASYNC_SCANNER)? };

    scanner_guard.start_scan(task_id, scanner_global_ref);
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_ipc_AsyncScanner_nativeCancelTask(
    _env: JNIEnv,
    _j_scanner: JObject,
    task_id: jlong,
) {
    RT.block_on(async {
        TASK_TRACKER.cancel(task_id as u64).await;
    });
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_ipc_AsyncScanner_releaseNativeScanner(
    mut env: JNIEnv,
    j_scanner: JObject,
) {
    ok_or_throw_without_return!(env, inner_release_async_scanner(&mut env, j_scanner));
}

fn inner_release_async_scanner(env: &mut JNIEnv, j_scanner: JObject) -> Result<()> {
    let _: AsyncScanner = unsafe { env.take_rust_field(j_scanner, NATIVE_ASYNC_SCANNER) }?;
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_ipc_AsyncScanner_importFfiSchema(
    mut env: JNIEnv,
    j_scanner: JObject,
    schema_addr: jlong,
) {
    ok_or_throw_without_return!(
        env,
        inner_import_async_ffi_schema(&mut env, j_scanner, schema_addr)
    );
}

fn inner_import_async_ffi_schema(
    env: &mut JNIEnv,
    j_scanner: JObject,
    schema_addr: jlong,
) -> Result<()> {
    let scanner_guard =
        unsafe { env.get_rust_field::<_, _, AsyncScanner>(j_scanner, NATIVE_ASYNC_SCANNER)? };

    let schema = RT.block_on(scanner_guard.inner.schema())?;
    let ffi_schema = FFI_ArrowSchema::try_from(&*schema)?;
    unsafe { std::ptr::write_unaligned(schema_addr as *mut FFI_ArrowSchema, ffi_schema) }
    Ok(())
}
