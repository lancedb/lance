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

use crate::blocking_dataset::{BlockingDataset, NATIVE_DATASET};
use crate::error::Result;
use crate::traits::FromJString;
use crate::{Error, RT};
use arrow::ffi_stream::FFI_ArrowArrayStream;
use jni::objects::{JClass, JObject, JString};
use jni::sys::{jboolean, jlong, JNI_TRUE};
use jni::JNIEnv;
use lance::dataset::scanner::DatasetRecordBatchStream;
use lance::dataset::sql::SqlQueryBuilder;
use lance_io::ffi::to_ffi_arrow_array_stream;

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_sql_SqlQuery_intoBatchRecords(
    mut env: JNIEnv,
    _class: JClass,
    java_dataset: JObject,
    sql: JString,
    table_name: JString,
    with_row_id: jboolean,
    with_row_addr: jboolean,
    stream_addr: jlong,
) {
    ok_or_throw_without_return!(
        env,
        inner_into_batch_records(
            &mut env,
            java_dataset,
            sql,
            table_name,
            with_row_id,
            with_row_addr,
            stream_addr,
        )
        .map_err(|e| Error::io_error(e.to_string()))
    )
}

fn inner_into_batch_records(
    env: &mut JNIEnv,
    java_dataset: JObject,
    sql: JString,
    table_name: JString,
    with_row_id: jboolean,
    with_row_addr: jboolean,
    stream_addr: jlong,
) -> Result<()> {
    let builder = sql_builder(
        env,
        java_dataset,
        sql,
        table_name,
        with_row_id,
        with_row_addr,
    )?;

    let stream = RT.block_on(async move { builder.build().await.unwrap().into_stream().await });

    let ffi_stream =
        to_ffi_arrow_array_stream(DatasetRecordBatchStream::new(stream), RT.handle().clone())?;

    unsafe { std::ptr::write_unaligned(stream_addr as *mut FFI_ArrowArrayStream, ffi_stream) }
    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_sql_SqlQuery_intoExplainPlan<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass,
    java_dataset: JObject,
    sql: JString,
    table_name: JString,
    with_row_id: jboolean,
    with_row_addr: jboolean,
    verbose: jboolean,
    analyze: jboolean,
) -> JString<'local> {
    ok_or_throw_with_return!(
        env,
        inner_into_explain_plan(
            &mut env,
            java_dataset,
            sql,
            table_name,
            with_row_id,
            with_row_addr,
            verbose,
            analyze
        )
        .map_err(|e| Error::io_error(e.to_string())),
        JString::default()
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_into_explain_plan<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    sql: JString,
    table_name: JString,
    with_row_id: jboolean,
    with_row_addr: jboolean,
    verbose: jboolean,
    analyze: jboolean,
) -> Result<JString<'local>> {
    let builder = sql_builder(
        env,
        java_dataset,
        sql,
        table_name,
        with_row_id,
        with_row_addr,
    )?;

    let explain = RT.block_on(async move {
        builder
            .build()
            .await
            .unwrap()
            .into_explain_plan(verbose == JNI_TRUE, analyze == JNI_TRUE)
            .await
    })?;

    Ok(env.new_string(explain)?)
}

fn sql_builder(
    env: &mut JNIEnv,
    java_dataset: JObject,
    sql: JString,
    table_name: JString,
    with_row_id: jboolean,
    with_row_addr: jboolean,
) -> Result<SqlQueryBuilder> {
    let sql_str = sql.extract(env)?;
    let table_str = table_name.extract(env)?;

    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;

    Ok(dataset_guard
        .inner
        .sql(sql_str.as_str())
        .table_name(table_str.as_str())
        .with_row_id(with_row_id == JNI_TRUE)
        .with_row_addr(with_row_addr == JNI_TRUE))
}
