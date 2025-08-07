use crate::blocking_dataset::{BlockingDataset, NATIVE_DATASET};
use crate::error::Result;
use crate::traits::{import_vec, FromJObjectWithEnv, FromJString, IntoJava};
use crate::utils::{to_java_map, to_rust_map};
use crate::Error;
use arrow::datatypes::Schema;
use arrow_schema::ffi::FFI_ArrowSchema;
use jni::objects::{JMap, JObject, JString, JValue};
use jni::JNIEnv;
use lance::dataset::transaction::{Operation, Transaction, TransactionBuilder};
use lance_core::datatypes::Schema as LanceSchema;
use std::collections::HashMap;
use std::sync::Arc;

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeReadTransaction<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_read_transaction(&mut env, java_dataset))
}

fn inner_read_transaction<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let transaction = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&java_dataset, NATIVE_DATASET) }?;
        dataset_guard.read_transaction()?
    };

    let transaction = match transaction {
        Some(transaction) => convert_to_java_transaction(env, transaction, &java_dataset)?,
        None => JObject::null(),
    };
    Ok(transaction)
}

fn convert_to_java_transaction<'local>(
    env: &mut JNIEnv<'local>,
    transaction: Transaction,
    java_dataset: &JObject,
) -> Result<JObject<'local>> {
    let uuid = env.new_string(transaction.uuid)?;
    let transaction_properties = match transaction.transaction_properties {
        Some(properties) => to_java_map(env, &properties)?,
        _ => JObject::null(),
    };
    let operation = convert_to_java_operation_inner(env, transaction.operation)?;
    let blobs_op = convert_to_java_operation(env, transaction.blobs_op)?;

    let java_transaction = env.new_object(
        "com/lancedb/lance/Transaction",
        "(Lcom/lancedb/lance/Dataset;JLjava/lang/String;Lcom/lancedb/lance/operation/Operation;Lcom/lancedb/lance/operation/Operation;Ljava/util/Map;Ljava/util/Map;)V",
        &[
            JValue::Object(java_dataset),
            JValue::Long(transaction.read_version as i64),
            JValue::Object(&uuid),
            JValue::Object(&operation),
            JValue::Object(&blobs_op),
            JValue::Object(&JObject::null()),
            JValue::Object(&transaction_properties),
        ],
    )?;
    Ok(java_transaction)
}

fn convert_to_java_operation<'local>(
    env: &mut JNIEnv<'local>,
    operation: Option<Operation>,
) -> Result<JObject<'local>> {
    let operation = match operation {
        Some(operation) => convert_to_java_operation_inner(env, operation)?,
        None => JObject::null(),
    };
    Ok(operation)
}

fn convert_to_java_operation_inner<'local>(
    env: &mut JNIEnv<'local>,
    operation: Operation,
) -> Result<JObject<'local>> {
    match operation {
        Operation::Append {
            fragments: rust_fragments,
        } => {
            let java_fragments = env.new_object("java/util/ArrayList", "()V", &[])?;
            for fragment in rust_fragments {
                let java_fragment = fragment.into_java(env)?;
                env.call_method(
                    &java_fragments,
                    "add",
                    "(Ljava/lang/Object;)Z",
                    &[JValue::Object(&java_fragment)],
                )?;
            }

            Ok(env.new_object(
                "com/lancedb/lance/operation/Append",
                "(Ljava/util/List;)V",
                &[JValue::Object(&java_fragments)],
            )?)
        }
        Operation::Overwrite {
            fragments: rust_fragments,
            schema,
            config_upsert_values,
        } => {
            let java_fragments = env.new_object("java/util/ArrayList", "()V", &[])?;
            for fragment in rust_fragments {
                let java_fragment = fragment.into_java(env)?;
                env.call_method(
                    &java_fragments,
                    "add",
                    "(Ljava/lang/Object;)Z",
                    &[JValue::Object(&java_fragment)],
                )?;
            }

            let java_schema = convert_to_java_schema(env, schema)?;
            let java_config = match config_upsert_values {
                Some(config_upsert_values) => to_java_map(env, &config_upsert_values)?,
                _ => JObject::null(),
            };

            Ok(env.new_object(
                "com/lancedb/lance/operation/Overwrite",
                "(Ljava/util/List;Lorg/apache/arrow/vector/types/pojo/Schema;Ljava/util/Map;)V",
                &[
                    JValue::Object(&java_fragments),
                    JValue::Object(&java_schema),
                    JValue::Object(&java_config),
                ],
            )?)
        }
        Operation::Project { schema } => {
            let java_schema = convert_to_java_schema(env, schema)?;

            Ok(env.new_object(
                "com/lancedb/lance/operation/Project",
                "(Lorg/apache/arrow/vector/types/pojo/Schema;)V",
                &[JValue::Object(&java_schema)],
            )?)
        }
        Operation::UpdateConfig {
            upsert_values,
            delete_keys,
            schema_metadata,
            field_metadata,
        } => {
            let upsert_values = match upsert_values {
                Some(config_values) => to_java_map(env, &config_values)?,
                _ => JObject::null(),
            };
            let delete_keys = match delete_keys {
                Some(keys) => {
                    let java_keys = env.new_object("java/util/ArrayList", "(I)V", &[])?;
                    for key in keys {
                        let java_key = env.new_string(key)?;
                        env.call_method(
                            &java_keys,
                            "add",
                            "(Ljava/lang/Object;)Z",
                            &[JValue::Object(&java_key)],
                        )?;
                    }
                    java_keys
                }
                _ => JObject::null(),
            };
            let schema_metadata = match schema_metadata {
                Some(schema_metadata) => to_java_map(env, &schema_metadata)?,
                _ => JObject::null(),
            };
            let field_metadata = match field_metadata {
                Some(field_metadata) => {
                    let java_map = env.new_object("java/util/HashMap", "()V", &[])?;
                    let map = JMap::from_env(env, &java_map)?;

                    for (field_id, field_meta) in field_metadata {
                        let java_field_id = env.new_object(
                            "java/lang/Integer",
                            "(I)V",
                            &[JValue::Int(field_id as i32)],
                        )?;

                        let java_field_metadata = to_java_map(env, &field_meta)?;
                        map.put(env, &java_field_id, &java_field_metadata)?;
                    }
                    java_map
                }
                _ => JObject::null(),
            };
            let java_operation = env.new_object(
                "com/lancedb/lance/operation/UpdateConfig",
                "(Ljava/util/Map;Ljava/util/List;Ljava/util/Map;Ljava/util/Map;)V",
                &[
                    JValue::Object(&upsert_values),
                    JValue::Object(&delete_keys),
                    JValue::Object(&schema_metadata),
                    JValue::Object(&field_metadata),
                ],
            )?;
            Ok(java_operation)
        }
        Operation::Merge {
            fragments: rust_fragments,
            schema,
        } => {
            let java_fragments = env.new_object("java/util/ArrayList", "()V", &[])?;
            for fragment in rust_fragments {
                let java_fragment = fragment.into_java(env)?;
                env.call_method(
                    &java_fragments,
                    "add",
                    "(Ljava/lang/Object;)Z",
                    &[JValue::Object(&java_fragment)],
                )?;
            }

            let java_schema = convert_to_java_schema(env, schema)?;

            Ok(env.new_object(
                "com/lancedb/lance/operation/Merge",
                "(Ljava/util/List;Lorg/apache/arrow/vector/types/pojo/Schema;)V",
                &[
                    JValue::Object(&java_fragments),
                    JValue::Object(&java_schema),
                ],
            )?)
        }
        _ => unimplemented!(),
    }
}

fn convert_to_java_schema<'local>(
    env: &mut JNIEnv<'local>,
    schema: LanceSchema,
) -> Result<JObject<'local>> {
    let java_schema = schema.into_java(env)?;
    Ok(env
        .call_method(
            &java_schema,
            "asArrowSchema",
            "()Lorg/apache/arrow/vector/types/pojo/Schema;",
            &[],
        )?
        .l()?)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Dataset_nativeCommitTransaction<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    java_transaction: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_commit_transaction(&mut env, java_dataset, java_transaction)
    )
}

fn inner_commit_transaction<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    java_transaction: JObject,
) -> Result<JObject<'local>> {
    let write_param_jobj = env
        .call_method(&java_transaction, "writeParams", "()Ljava/util/Map;", &[])?
        .l()?;
    let write_param_jmap = JMap::from_env(env, &write_param_jobj)?;
    let write_param = to_rust_map(env, &write_param_jmap)?;
    let transaction = convert_to_rust_transaction(env, java_transaction, Some(&java_dataset))?;
    let new_blocking_ds = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.commit_transaction(transaction, write_param)?
    };
    new_blocking_ds.into_java(env)
}

fn convert_to_rust_transaction(
    env: &mut JNIEnv,
    java_transaction: JObject,
    java_dataset: Option<&JObject>,
) -> Result<Transaction> {
    let read_ver = env
        .call_method(&java_transaction, "readVersion", "()J", &[])?
        .j()?;
    let uuid = env
        .call_method(&java_transaction, "uuid", "()Ljava/lang/String;", &[])?
        .l()?;
    let uuid = JString::from(uuid);
    let uuid = env.get_string(&uuid)?.into();
    let op = env
        .call_method(
            &java_transaction,
            "operation",
            "()Lcom/lancedb/lance/operation/Operation;",
            &[],
        )?
        .l()?;
    let op = convert_to_rust_operation(env, op, java_dataset)?;

    let blobs_op = env
        .call_method(
            &java_transaction,
            "blobsOperation",
            "()Lcom/lancedb/lance/operation/Operation;",
            &[],
        )?
        .l()?;
    let blobs_op = if blobs_op.is_null() {
        None
    } else {
        Some(convert_to_rust_operation(env, blobs_op, java_dataset)?)
    };

    let transaction_properties = env
        .call_method(
            &java_transaction,
            "transactionProperties",
            "()Ljava/util/Map;",
            &[],
        )?
        .l()?;
    let transaction_properties = JMap::from_env(env, &transaction_properties)?;
    let transaction_properties = to_rust_map(env, &transaction_properties)?;
    Ok(TransactionBuilder::new(read_ver as u64, op)
        .uuid(uuid)
        .blobs_op(blobs_op)
        .transaction_properties(Some(Arc::new(transaction_properties)))
        .build())
}

fn convert_to_rust_operation(
    env: &mut JNIEnv,
    java_operation: JObject,
    java_dataset: Option<&JObject>,
) -> Result<Operation> {
    let name = env
        .call_method(&java_operation, "name", "()Ljava/lang/String;", &[])?
        .l()?;
    let name = JString::from(name);
    let name: String = env.get_string(&name)?.into();
    let op = match name.as_str() {
        "Project" => Operation::Project {
            schema: convert_schema_from_operation(env, &java_operation, java_dataset.unwrap())?,
        },
        "UpdateConfig" => {
            let upsert_values = env
                .call_method(&java_operation, "upsertValues", "()Ljava/util/Map;", &[])?
                .l()?;
            let upsert_values = if !upsert_values.is_null() {
                let upsert_values = JMap::from_env(env, &upsert_values)?;
                Some(to_rust_map(env, &upsert_values)?)
            } else {
                None
            };

            let delete_keys = env
                .call_method(&java_operation, "deleteKeys", "()Ljava/util/List;", &[])?
                .l()?;
            let delete_keys = if !delete_keys.is_null() {
                let keys = import_vec(env, &delete_keys)?;
                let keys = keys
                    .into_iter()
                    .map(|key| JString::from(key))
                    .map(|key| key.extract(env))
                    .collect::<Result<Vec<_>>>()?;
                Some(keys)
            } else {
                None
            };

            let schema_metadata = env
                .call_method(&java_operation, "schemaMetadata", "()Ljava/util/Map;", &[])?
                .l()?;
            let schema_metadata = if !schema_metadata.is_null() {
                let schema_metadata = JMap::from_env(env, &schema_metadata)?;
                Some(to_rust_map(env, &schema_metadata)?)
            } else {
                None
            };

            let field_metadata = env
                .call_method(&java_operation, "fieldMetadata", "()Ljava/util/Map;", &[])?
                .l()?;
            let field_metadata = if !field_metadata.is_null() {
                let field_metadata = JMap::from_env(env, &field_metadata)?;
                let mut field_metadata_map = HashMap::new();
                let mut iter = field_metadata.iter(env)?;
                env.with_local_frame(16, |env| {
                    while let Some((key, value)) = iter.next(env)? {
                        let field_id = env.call_method(&key, "intValue", "()I", &[])?.i()? as u32;
                        let inner_map = JMap::from_env(env, &value)?;
                        let value_map = to_rust_map(env, &inner_map)?;
                        field_metadata_map.insert(field_id, value_map);
                    }
                    Ok::<(), Error>(())
                })?;
                Some(field_metadata_map)
            } else {
                None
            };

            Operation::UpdateConfig {
                upsert_values,
                delete_keys,
                schema_metadata,
                field_metadata,
            }
        }
        "Append" => {
            let fragment_objs = env
                .call_method(&java_operation, "fragments", "()Ljava/util/List;", &[])?
                .l()?;
            let fragment_objs = import_vec(env, &fragment_objs)?;
            let mut fragments = Vec::with_capacity(fragment_objs.len());
            for f in fragment_objs {
                fragments.push(f.extract_object(env)?);
            }
            Operation::Append { fragments }
        }
        "Overwrite" => {
            let fragment_objs = env
                .call_method(&java_operation, "fragments", "()Ljava/util/List;", &[])?
                .l()?;
            let fragment_objs = import_vec(env, &fragment_objs)?;
            let mut fragments = Vec::with_capacity(fragment_objs.len());
            for f in fragment_objs {
                fragments.push(f.extract_object(env)?);
            }
            let config_upsert_values = env
                .call_method(
                    &java_operation,
                    "configUpsertValues",
                    "()Ljava/util/Map;",
                    &[],
                )?
                .l()?;
            let config_upsert_values = if config_upsert_values.is_null() {
                None
            } else {
                let config_upsert_values = JMap::from_env(env, &config_upsert_values)?;
                Some(to_rust_map(env, &config_upsert_values)?)
            };
            Operation::Overwrite {
                fragments,
                schema: convert_schema_from_operation(env, &java_operation, java_dataset.unwrap())?,
                config_upsert_values,
            }
        }
        "Merge" => {
            let fragment_objs = env
                .call_method(&java_operation, "fragments", "()Ljava/util/List;", &[])?
                .l()?;
            let fragment_objs = import_vec(env, &fragment_objs)?;
            let mut fragments = Vec::with_capacity(fragment_objs.len());
            for f in fragment_objs {
                fragments.push(f.extract_object(env)?);
            }
            Operation::Merge {
                fragments,
                schema: convert_schema_from_operation(env, &java_operation, java_dataset.unwrap())?,
            }
        }
        _ => unimplemented!(),
    };
    Ok(op)
}

fn convert_schema_from_operation(
    env: &mut JNIEnv,
    java_operation: &JObject,
    java_dataset: &JObject,
) -> Result<LanceSchema> {
    let java_buffer_allocator = env
        .call_method(
            java_dataset,
            "allocator",
            "()Lorg/apache/arrow/memory/BufferAllocator;",
            &[],
        )?
        .l()?;
    let schema_ptr = env
        .call_method(
            java_operation,
            "exportSchema",
            "(Lorg/apache/arrow/memory/BufferAllocator;)J",
            &[JValue::Object(&java_buffer_allocator)],
        )?
        .j()?;
    let c_schema_ptr = schema_ptr as *mut FFI_ArrowSchema;
    let c_schema = unsafe { FFI_ArrowSchema::from_raw(c_schema_ptr) };
    let schema = Schema::try_from(&c_schema)?;
    Ok(
        LanceSchema::try_from(&schema)
            .expect("Failed to convert from arrow schema to lance schema"),
    )
}
