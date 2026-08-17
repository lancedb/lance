// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::blocking_dataset::{BlockingDataset, NATIVE_DATASET};
use crate::error::Result;
use crate::traits::{FromJObjectWithEnv, IntoJava, import_vec_from_method};
use crate::{JNIEnvExt, block_on};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use jni::JNIEnv;
use jni::objects::{JObject, JString, JValue};
use jni::sys::{jboolean, jlong};
use lance::dataset::{CellFlagChange, InsertBuilder, WriteMode, WriteParams};
use lance_table::format::CellFlagDefinition;
use std::sync::Arc;

impl FromJObjectWithEnv<CellFlagChange> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<CellFlagChange> {
        Ok(CellFlagChange::new(
            env.get_string_from_method(self, "field")?,
            env.get_string_from_method(self, "name")?,
            env.get_boolean_from_method(self, "value")?,
        ))
    }
}

pub(crate) fn extract_cell_flag_changes_from_method(
    env: &mut JNIEnv<'_>,
    object: &JObject<'_>,
    method: &str,
) -> Result<Vec<CellFlagChange>> {
    import_vec_from_method(env, object, method, |env, change| {
        change.extract_object(env)
    })
}

fn definition_to_java<'local>(
    env: &mut JNIEnv<'local>,
    definition: &CellFlagDefinition,
) -> Result<JObject<'local>> {
    let name = env.new_string(&definition.name)?;
    Ok(env.new_object(
        "org/lance/CellFlagDefinition",
        "(JILjava/lang/String;)V",
        &[
            JValue::Long(i64::from(definition.flag_id)),
            JValue::Int(definition.field_id),
            JValue::Object(&name),
        ],
    )?)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCellFlagDefinitions<'local>(
    mut env: JNIEnv<'local>,
    dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_cell_flag_definitions(&mut env, dataset))
}

fn inner_cell_flag_definitions<'local>(
    env: &mut JNIEnv<'local>,
    dataset: JObject,
) -> Result<JObject<'local>> {
    let definitions = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&dataset, NATIVE_DATASET)? };
        dataset.inner.cell_flag_definitions().to_vec()
    };
    let list = env.new_object("java/util/ArrayList", "()V", &[])?;
    for definition in &definitions {
        let java_definition = definition_to_java(env, definition)?;
        env.call_method(
            &list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValue::Object(&java_definition)],
        )?;
    }
    Ok(list)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeRegisterCellFlag<'local>(
    mut env: JNIEnv<'local>,
    dataset: JObject,
    field: JString,
    name: JString,
    initial_value: jboolean,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_register_cell_flag(&mut env, dataset, field, name, initial_value)
    )
}

fn inner_register_cell_flag<'local>(
    env: &mut JNIEnv<'local>,
    dataset: JObject,
    field: JString,
    name: JString,
    initial_value: jboolean,
) -> Result<JObject<'local>> {
    let field: String = env.get_string(&field)?.into();
    let name: String = env.get_string(&name)?.into();
    let definition = {
        let mut dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&dataset, NATIVE_DATASET)? };
        block_on(
            dataset
                .inner
                .register_cell_flag(field, name, initial_value != 0),
        )?
    };
    definition_to_java(env, &definition)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeRenameCellFlag(
    mut env: JNIEnv,
    dataset: JObject,
    field: JString,
    name: JString,
    new_name: JString,
) {
    ok_or_throw_without_return!(
        env,
        inner_rename_cell_flag(&mut env, dataset, field, name, new_name)
    );
}

fn inner_rename_cell_flag(
    env: &mut JNIEnv,
    dataset: JObject,
    field: JString,
    name: JString,
    new_name: JString,
) -> Result<()> {
    let field: String = env.get_string(&field)?.into();
    let name: String = env.get_string(&name)?.into();
    let new_name: String = env.get_string(&new_name)?.into();
    let mut dataset =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(&dataset, NATIVE_DATASET)? };
    Ok(block_on(
        dataset.inner.rename_cell_flag(field, name, new_name),
    )?)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeDropCellFlag(
    mut env: JNIEnv,
    dataset: JObject,
    field: JString,
    name: JString,
) {
    ok_or_throw_without_return!(env, inner_drop_cell_flag(&mut env, dataset, field, name));
}

fn inner_drop_cell_flag(
    env: &mut JNIEnv,
    dataset: JObject,
    field: JString,
    name: JString,
) -> Result<()> {
    let field: String = env.get_string(&field)?.into();
    let name: String = env.get_string(&name)?.into();
    let mut dataset =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(&dataset, NATIVE_DATASET)? };
    Ok(block_on(dataset.inner.drop_cell_flag(field, name))?)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeAppendWithCellFlags<'local>(
    mut env: JNIEnv<'local>,
    dataset: JObject,
    batch_address: jlong,
    changes: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_append_with_cell_flags(&mut env, dataset, batch_address, changes)
    )
}

fn inner_append_with_cell_flags<'local>(
    env: &mut JNIEnv<'local>,
    dataset: JObject,
    batch_address: jlong,
    changes: JObject,
) -> Result<JObject<'local>> {
    let changes =
        crate::traits::import_vec_to_rust(env, &changes, |env, change| change.extract_object(env))?;
    let inner = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&dataset, NATIVE_DATASET)? };
        dataset.inner.clone()
    };
    let stream_ptr = batch_address as *mut FFI_ArrowArrayStream;
    let source_stream = unsafe { ArrowArrayStreamReader::from_raw(stream_ptr)? };
    let params = WriteParams {
        mode: WriteMode::Append,
        ..Default::default()
    };
    let new_dataset = block_on(
        InsertBuilder::new(Arc::new(inner))
            .with_params(&params)
            .with_cell_flags(changes)
            .execute_stream(source_stream),
    )?;
    BlockingDataset { inner: new_dataset }.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeMergeWithCellFlags(
    mut env: JNIEnv,
    dataset: JObject,
    batch_address: jlong,
    left_on: JString,
    right_on: JString,
    changes: JObject,
) {
    ok_or_throw_without_return!(
        env,
        inner_merge_with_cell_flags(&mut env, dataset, batch_address, left_on, right_on, changes,)
    );
}

fn inner_merge_with_cell_flags(
    env: &mut JNIEnv,
    dataset: JObject,
    batch_address: jlong,
    left_on: JString,
    right_on: JString,
    changes: JObject,
) -> Result<()> {
    let left_on: String = env.get_string(&left_on)?.into();
    let right_on: String = env.get_string(&right_on)?.into();
    let changes =
        crate::traits::import_vec_to_rust(env, &changes, |env, change| change.extract_object(env))?;
    let stream_ptr = batch_address as *mut FFI_ArrowArrayStream;
    let source_stream = unsafe { ArrowArrayStreamReader::from_raw(stream_ptr)? };
    let mut dataset =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(&dataset, NATIVE_DATASET)? };
    Ok(block_on(dataset.inner.merge_with_cell_flags(
        source_stream,
        &left_on,
        &right_on,
        &changes,
    ))?)
}
