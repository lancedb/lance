use crate::blocking_dataset::{BlockingDataset, NATIVE_DATASET};
use crate::error::Result;
use crate::traits::{import_vec, FromJObjectWithEnv, IntoJava};
use crate::utils::{to_java_map, to_rust_map};
use arrow::datatypes::Schema;
use arrow_schema::ffi::FFI_ArrowSchema;
use jni::objects::{JMap, JObject, JString, JValue};
use jni::JNIEnv;
use lance::dataset::transaction::{Operation, Transaction, TransactionBuilder};
use lance_core::datatypes::Schema as LanceSchema;
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
