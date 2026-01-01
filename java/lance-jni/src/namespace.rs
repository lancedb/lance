// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use bytes::Bytes;
use jni::objects::{JByteArray, JMap, JObject, JString};
use jni::sys::{jbyteArray, jlong, jstring};
use jni::JNIEnv;
use lance_namespace::models::*;
use lance_namespace::LanceNamespace as LanceNamespaceTrait;
use lance_namespace_impls::{
    ConnectBuilder, DirectoryNamespace, DirectoryNamespaceBuilder, RestAdapter, RestAdapterConfig,
    RestNamespace, RestNamespaceBuilder,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::utils::to_rust_map;
use crate::RT;

/// Blocking wrapper for DirectoryNamespace
pub struct BlockingDirectoryNamespace {
    pub(crate) inner: DirectoryNamespace,
}

/// Blocking wrapper for RestNamespace
pub struct BlockingRestNamespace {
    pub(crate) inner: RestNamespace,
}

// ============================================================================
// DirectoryNamespace JNI Functions
// ============================================================================

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_createNative(
    mut env: JNIEnv,
    _obj: JObject,
    properties_map: JObject,
) -> jlong {
    ok_or_throw_with_return!(
        env,
        create_directory_namespace_internal(&mut env, properties_map),
        0
    )
}

fn create_directory_namespace_internal(env: &mut JNIEnv, properties_map: JObject) -> Result<jlong> {
    // Convert Java HashMap to Rust HashMap
    let jmap = JMap::from_env(env, &properties_map)?;
    let properties = to_rust_map(env, &jmap)?;

    // Build DirectoryNamespace using builder
    let builder = DirectoryNamespaceBuilder::from_properties(properties, None).map_err(|e| {
        Error::runtime_error(format!("Failed to create DirectoryNamespaceBuilder: {}", e))
    })?;

    let namespace = RT
        .block_on(builder.build())
        .map_err(|e| Error::runtime_error(format!("Failed to build DirectoryNamespace: {}", e)))?;

    let blocking_namespace = BlockingDirectoryNamespace { inner: namespace };
    let handle = Box::into_raw(Box::new(blocking_namespace)) as jlong;
    Ok(handle)
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_releaseNative(
    _env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) {
    if handle != 0 {
        unsafe {
            let _ = Box::from_raw(handle as *mut BlockingDirectoryNamespace);
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_namespaceIdNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) -> jstring {
    let namespace = unsafe { &*(handle as *const BlockingDirectoryNamespace) };
    let namespace_id = namespace.inner.namespace_id();
    ok_or_throw_with_return!(
        env,
        env.new_string(namespace_id).map_err(Error::from),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_listNamespacesNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.list_namespaces(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_describeNamespaceNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.describe_namespace(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_createNamespaceNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.create_namespace(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_dropNamespaceNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.drop_namespace(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_namespaceExistsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) {
    ok_or_throw_without_return!(
        env,
        call_namespace_void_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.namespace_exists(req))
        })
    )
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_listTablesNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.list_tables(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_describeTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.describe_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_registerTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.register_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_tableExistsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) {
    ok_or_throw_without_return!(
        env,
        call_namespace_void_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.table_exists(req))
        })
    )
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_dropTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.drop_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_deregisterTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.deregister_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_countTableRowsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jlong {
    ok_or_throw_with_return!(
        env,
        call_namespace_count_method(&mut env, handle, request_json),
        0
    )
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_createTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
    request_data: JByteArray,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_with_data_method(
            &mut env,
            handle,
            request_json,
            request_data,
            |ns, req, data| { RT.block_on(ns.inner.create_table(req, data)) }
        ),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
#[allow(deprecated)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_createEmptyTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.create_empty_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_declareTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.declare_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_insertIntoTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
    request_data: JByteArray,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_with_data_method(
            &mut env,
            handle,
            request_json,
            request_data,
            |ns, req, data| { RT.block_on(ns.inner.insert_into_table(req, data)) }
        ),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_mergeInsertIntoTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
    request_data: JByteArray,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_with_data_method(
            &mut env,
            handle,
            request_json,
            request_data,
            |ns, req, data| { RT.block_on(ns.inner.merge_insert_into_table(req, data)) }
        ),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_updateTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.update_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_deleteFromTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.delete_from_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_queryTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        call_namespace_query_method(&mut env, handle, request_json),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_createTableIndexNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.create_table_index(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_listTableIndicesNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.list_table_indices(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_describeTableIndexStatsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.describe_table_index_stats(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_describeTransactionNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.describe_transaction(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_alterTransactionNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.alter_transaction(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

// ============================================================================
// RestNamespace JNI Functions
// ============================================================================

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_createNative(
    mut env: JNIEnv,
    _obj: JObject,
    properties_map: JObject,
) -> jlong {
    ok_or_throw_with_return!(
        env,
        create_rest_namespace_internal(&mut env, properties_map),
        0
    )
}

fn create_rest_namespace_internal(env: &mut JNIEnv, properties_map: JObject) -> Result<jlong> {
    // Convert Java HashMap to Rust HashMap
    let jmap = JMap::from_env(env, &properties_map)?;
    let properties = to_rust_map(env, &jmap)?;

    // Build RestNamespace using builder
    let builder = RestNamespaceBuilder::from_properties(properties).map_err(|e| {
        Error::runtime_error(format!("Failed to create RestNamespaceBuilder: {}", e))
    })?;

    let namespace = builder.build();

    let blocking_namespace = BlockingRestNamespace { inner: namespace };
    let handle = Box::into_raw(Box::new(blocking_namespace)) as jlong;
    Ok(handle)
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_releaseNative(
    _env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) {
    if handle != 0 {
        unsafe {
            let _ = Box::from_raw(handle as *mut BlockingRestNamespace);
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_namespaceIdNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) -> jstring {
    let namespace = unsafe { &*(handle as *const BlockingRestNamespace) };
    let namespace_id = namespace.inner.namespace_id();
    ok_or_throw_with_return!(
        env,
        env.new_string(namespace_id).map_err(Error::from),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_listNamespacesNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.list_namespaces(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_describeNamespaceNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.describe_namespace(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_createNamespaceNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.create_namespace(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_dropNamespaceNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.drop_namespace(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_namespaceExistsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) {
    ok_or_throw_without_return!(
        env,
        call_rest_namespace_void_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.namespace_exists(req))
        })
    )
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_listTablesNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.list_tables(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_describeTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.describe_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_registerTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.register_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_tableExistsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) {
    ok_or_throw_without_return!(
        env,
        call_rest_namespace_void_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.table_exists(req))
        })
    )
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_dropTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.drop_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_deregisterTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.deregister_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_countTableRowsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jlong {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_count_method(&mut env, handle, request_json),
        0
    )
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_createTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
    request_data: JByteArray,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_with_data_method(
            &mut env,
            handle,
            request_json,
            request_data,
            |ns, req, data| { RT.block_on(ns.inner.create_table(req, data)) }
        ),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
#[allow(deprecated)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_createEmptyTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.create_empty_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_declareTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.declare_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_insertIntoTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
    request_data: JByteArray,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_with_data_method(
            &mut env,
            handle,
            request_json,
            request_data,
            |ns, req, data| { RT.block_on(ns.inner.insert_into_table(req, data)) }
        ),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_mergeInsertIntoTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
    request_data: JByteArray,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_with_data_method(
            &mut env,
            handle,
            request_json,
            request_data,
            |ns, req, data| { RT.block_on(ns.inner.merge_insert_into_table(req, data)) }
        ),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_updateTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.update_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_deleteFromTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.delete_from_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_queryTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_query_method(&mut env, handle, request_json),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_createTableIndexNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.create_table_index(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_listTableIndicesNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.list_table_indices(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_describeTableIndexStatsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.describe_table_index_stats(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_describeTransactionNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.describe_transaction(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_alterTransactionNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |ns, req| {
            RT.block_on(ns.inner.alter_transaction(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Helper function to call namespace methods that return a response object (DirectoryNamespace)
fn call_namespace_method<'local, Req, Resp, F>(
    env: &mut JNIEnv<'local>,
    handle: jlong,
    request_json: JString,
    f: F,
) -> Result<JString<'local>>
where
    Req: for<'de> Deserialize<'de>,
    Resp: Serialize,
    F: FnOnce(&BlockingDirectoryNamespace, Req) -> lance_core::Result<Resp>,
{
    let namespace = unsafe { &*(handle as *const BlockingDirectoryNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: Req = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    let response = f(namespace, request)
        .map_err(|e| Error::runtime_error(format!("Namespace operation failed: {}", e)))?;

    let response_json = serde_json::to_string(&response)
        .map_err(|e| Error::runtime_error(format!("Failed to serialize response: {}", e)))?;

    env.new_string(response_json).map_err(Into::into)
}

/// Helper function for void methods (DirectoryNamespace)
fn call_namespace_void_method<Req, F>(
    env: &mut JNIEnv,
    handle: jlong,
    request_json: JString,
    f: F,
) -> Result<()>
where
    Req: for<'de> Deserialize<'de>,
    F: FnOnce(&BlockingDirectoryNamespace, Req) -> lance_core::Result<()>,
{
    let namespace = unsafe { &*(handle as *const BlockingDirectoryNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: Req = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    f(namespace, request)
        .map_err(|e| Error::runtime_error(format!("Namespace operation failed: {}", e)))?;

    Ok(())
}

/// Helper function for count methods (DirectoryNamespace)
fn call_namespace_count_method(
    env: &mut JNIEnv,
    handle: jlong,
    request_json: JString,
) -> Result<jlong> {
    let namespace = unsafe { &*(handle as *const BlockingDirectoryNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: CountTableRowsRequest = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    let count = RT
        .block_on(namespace.inner.count_table_rows(request))
        .map_err(|e| Error::runtime_error(format!("Count table rows failed: {}", e)))?;

    Ok(count)
}

/// Helper function for methods with data parameter (DirectoryNamespace)
fn call_namespace_with_data_method<'local, Req, Resp, F>(
    env: &mut JNIEnv<'local>,
    handle: jlong,
    request_json: JString,
    request_data: JByteArray,
    f: F,
) -> Result<JString<'local>>
where
    Req: for<'de> Deserialize<'de>,
    Resp: Serialize,
    F: FnOnce(&BlockingDirectoryNamespace, Req, Bytes) -> lance_core::Result<Resp>,
{
    let namespace = unsafe { &*(handle as *const BlockingDirectoryNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: Req = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    let data_vec = env.convert_byte_array(request_data)?;
    let data = bytes::Bytes::from(data_vec);

    let response = f(namespace, request, data)
        .map_err(|e| Error::runtime_error(format!("Namespace operation failed: {}", e)))?;

    let response_json = serde_json::to_string(&response)
        .map_err(|e| Error::runtime_error(format!("Failed to serialize response: {}", e)))?;

    env.new_string(response_json).map_err(Into::into)
}

/// Helper function for query methods that return byte arrays (DirectoryNamespace)
fn call_namespace_query_method<'local>(
    env: &mut JNIEnv<'local>,
    handle: jlong,
    request_json: JString,
) -> Result<JByteArray<'local>> {
    let namespace = unsafe { &*(handle as *const BlockingDirectoryNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: QueryTableRequest = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    let result_bytes = RT
        .block_on(namespace.inner.query_table(request))
        .map_err(|e| Error::runtime_error(format!("Query table failed: {}", e)))?;

    let byte_array = env.byte_array_from_slice(&result_bytes)?;
    Ok(byte_array)
}

/// Helper function to call namespace methods that return a response object (RestNamespace)
fn call_rest_namespace_method<'local, Req, Resp, F>(
    env: &mut JNIEnv<'local>,
    handle: jlong,
    request_json: JString,
    f: F,
) -> Result<JString<'local>>
where
    Req: for<'de> Deserialize<'de>,
    Resp: Serialize,
    F: FnOnce(&BlockingRestNamespace, Req) -> lance_core::Result<Resp>,
{
    let namespace = unsafe { &*(handle as *const BlockingRestNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: Req = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    let response = f(namespace, request)
        .map_err(|e| Error::runtime_error(format!("Namespace operation failed: {}", e)))?;

    let response_json = serde_json::to_string(&response)
        .map_err(|e| Error::runtime_error(format!("Failed to serialize response: {}", e)))?;

    env.new_string(response_json).map_err(Into::into)
}

/// Helper function for void methods (RestNamespace)
fn call_rest_namespace_void_method<Req, F>(
    env: &mut JNIEnv,
    handle: jlong,
    request_json: JString,
    f: F,
) -> Result<()>
where
    Req: for<'de> Deserialize<'de>,
    F: FnOnce(&BlockingRestNamespace, Req) -> lance_core::Result<()>,
{
    let namespace = unsafe { &*(handle as *const BlockingRestNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: Req = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    f(namespace, request)
        .map_err(|e| Error::runtime_error(format!("Namespace operation failed: {}", e)))?;

    Ok(())
}

/// Helper function for count methods (RestNamespace)
fn call_rest_namespace_count_method(
    env: &mut JNIEnv,
    handle: jlong,
    request_json: JString,
) -> Result<jlong> {
    let namespace = unsafe { &*(handle as *const BlockingRestNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: CountTableRowsRequest = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    let count = RT
        .block_on(namespace.inner.count_table_rows(request))
        .map_err(|e| Error::runtime_error(format!("Count table rows failed: {}", e)))?;

    Ok(count)
}

/// Helper function for methods with data parameter (RestNamespace)
fn call_rest_namespace_with_data_method<'local, Req, Resp, F>(
    env: &mut JNIEnv<'local>,
    handle: jlong,
    request_json: JString,
    request_data: JByteArray,
    f: F,
) -> Result<JString<'local>>
where
    Req: for<'de> Deserialize<'de>,
    Resp: Serialize,
    F: FnOnce(&BlockingRestNamespace, Req, Bytes) -> lance_core::Result<Resp>,
{
    let namespace = unsafe { &*(handle as *const BlockingRestNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: Req = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    let data_vec = env.convert_byte_array(request_data)?;
    let data = bytes::Bytes::from(data_vec);

    let response = f(namespace, request, data)
        .map_err(|e| Error::runtime_error(format!("Namespace operation failed: {}", e)))?;

    let response_json = serde_json::to_string(&response)
        .map_err(|e| Error::runtime_error(format!("Failed to serialize response: {}", e)))?;

    env.new_string(response_json).map_err(Into::into)
}

/// Helper function for query methods that return byte arrays (RestNamespace)
fn call_rest_namespace_query_method<'local>(
    env: &mut JNIEnv<'local>,
    handle: jlong,
    request_json: JString,
) -> Result<JByteArray<'local>> {
    let namespace = unsafe { &*(handle as *const BlockingRestNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: QueryTableRequest = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    let result_bytes = RT
        .block_on(namespace.inner.query_table(request))
        .map_err(|e| Error::runtime_error(format!("Query table failed: {}", e)))?;

    let byte_array = env.byte_array_from_slice(&result_bytes)?;
    Ok(byte_array)
}
// ============================================================================
// RestAdapter - Server for testing
// ============================================================================

/// Wrapper for RestAdapter that manages the server lifecycle
pub struct BlockingRestAdapter {
    backend: Arc<dyn LanceNamespaceTrait>,
    config: RestAdapterConfig,
    server_handle: Option<lance_namespace_impls::RestAdapterHandle>,
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestAdapter_createNative(
    mut env: JNIEnv,
    _obj: JObject,
    namespace_impl: JString,
    properties_map: JObject,
    host: JString,
    port: JObject,
) -> jlong {
    ok_or_throw_with_return!(
        env,
        create_rest_adapter_internal(&mut env, namespace_impl, properties_map, host, port),
        0
    )
}

fn create_rest_adapter_internal(
    env: &mut JNIEnv,
    namespace_impl: JString,
    properties_map: JObject,
    host: JString,
    port: JObject,
) -> Result<jlong> {
    // Get namespace implementation type
    let impl_str: String = env.get_string(&namespace_impl)?.into();

    // Convert Java HashMap to Rust HashMap
    let jmap = JMap::from_env(env, &properties_map)?;
    let properties = to_rust_map(env, &jmap)?;

    // Build backend namespace using ConnectBuilder
    let mut builder = ConnectBuilder::new(impl_str);
    for (k, v) in properties {
        builder = builder.property(k, v);
    }

    let backend = RT
        .block_on(builder.connect())
        .map_err(|e| Error::runtime_error(format!("Failed to build backend namespace: {}", e)))?;

    // Build config with defaults, overriding if values provided
    let mut config = RestAdapterConfig::default();

    // Get host string if not null
    if !host.is_null() {
        config.host = env.get_string(&host)?.into();
    }

    // Get port if not null (Integer object)
    if !port.is_null() {
        let port_value = env
            .call_method(&port, "intValue", "()I", &[])?
            .i()
            .map_err(|e| Error::runtime_error(format!("Failed to get port value: {}", e)))?;
        config.port = port_value as u16;
    }

    let adapter = BlockingRestAdapter {
        backend,
        config,
        server_handle: None,
    };

    let handle = Box::into_raw(Box::new(adapter)) as jlong;
    Ok(handle)
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestAdapter_start(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) {
    ok_or_throw_without_return!(env, start_internal(handle))
}

fn start_internal(handle: jlong) -> Result<()> {
    let adapter = unsafe { &mut *(handle as *mut BlockingRestAdapter) };
    let rest_adapter = RestAdapter::new(adapter.backend.clone(), adapter.config.clone());
    let server_handle = RT.block_on(rest_adapter.start())?;
    adapter.server_handle = Some(server_handle);
    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestAdapter_getPort(
    _env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) -> jni::sys::jint {
    let adapter = unsafe { &*(handle as *const BlockingRestAdapter) };
    adapter
        .server_handle
        .as_ref()
        .map(|h| h.port() as jni::sys::jint)
        .unwrap_or(0)
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestAdapter_stop(
    _env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) {
    let adapter = unsafe { &mut *(handle as *mut BlockingRestAdapter) };

    if let Some(server_handle) = adapter.server_handle.take() {
        server_handle.shutdown();
    }
}

#[no_mangle]
pub extern "system" fn Java_org_lance_namespace_RestAdapter_releaseNative(
    _env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) {
    if handle != 0 {
        unsafe {
            let mut adapter = Box::from_raw(handle as *mut BlockingRestAdapter);
            if let Some(server_handle) = adapter.server_handle.take() {
                server_handle.shutdown();
            }
        }
    }
}
