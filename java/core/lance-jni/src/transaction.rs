use crate::blocking_dataset::{BlockingDataset, NATIVE_DATASET};
use crate::error::Result;
use crate::traits::IntoJava;
use crate::utils::to_rust_map;
use arrow::datatypes::Schema;
use arrow_schema::ffi::FFI_ArrowSchema;
use jni::objects::{JMap, JObject, JString};
use jni::JNIEnv;
use lance::dataset::transaction::{Operation, Transaction};
use lance_core::datatypes::Schema as LanceSchema;

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_Transaction_commitNative<'local>(
    mut env: JNIEnv<'local>,
    jtransaction: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_commit_transaction(&mut env, jtransaction))
}

fn inner_commit_transaction<'local>(
    env: &mut JNIEnv<'local>,
    java_tx: JObject,
) -> Result<JObject<'local>> {
    let java_dataset: JObject = env
        .call_method(&java_tx, "dataset", "()Lcom/lancedb/lance/Dataset;", &[])?
        .l()?;
    let write_param_jobj = env
        .call_method(&java_tx, "writeParams", "()Ljava/util/Map;", &[])?
        .l()?;
    let write_param_jmap = JMap::from_env(env, &write_param_jobj)?;
    let write_param = to_rust_map(env, &write_param_jmap)?;
    let transaction = convert_to_rust_transaction(env, java_tx)?;
    let new_blocking_ds = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.commit_transaction(transaction, write_param)?
    };
    new_blocking_ds.into_java(env)
}

fn convert_to_rust_transaction(env: &mut JNIEnv, java_tx: JObject) -> Result<Transaction> {
    let read_ver = env.call_method(&java_tx, "readVersion", "()J", &[])?.j()?;
    let uuid = env
        .call_method(&java_tx, "uuid", "()Ljava/lang/String;", &[])?
        .l()?;
    let uuid = JString::from(uuid);
    let uuid = env.get_string(&uuid)?.into();
    let op = env
        .call_method(
            &java_tx,
            "operation",
            "()Lcom/lancedb/lance/operation/Operation;",
            &[],
        )?
        .l()?;
    let op = convert_to_rust_operation(env, op)?;

    let blobs_op = env
        .call_method(
            &java_tx,
            "blobsOperation",
            "()Lcom/lancedb/lance/operation/Operation;",
            &[],
        )?
        .l()?;
    let blobs_op = if blobs_op.is_null() {
        None
    } else {
        Some(convert_to_rust_operation(env, blobs_op)?)
    };

    Ok(Transaction {
        read_version: read_ver as u64,
        uuid,
        operation: op,
        blobs_op,
        tag: None,
    })
}

fn convert_to_rust_operation(env: &mut JNIEnv, java_operation: JObject) -> Result<Operation> {
    let name = env
        .call_method(&java_operation, "name", "()Ljava/lang/String;", &[])?
        .l()?;
    let name = JString::from(name);
    let name: String = env.get_string(&name)?.into();
    let op = match name.as_str() {
        "Project" => {
            let schema_ptr = env
                .call_method(&java_operation, "exportSchema", "()J", &[])?
                .j()?;
            log::info!("Schema pointer: {:?}", schema_ptr);
            let c_schema_ptr = schema_ptr as *mut FFI_ArrowSchema;
            let c_schema = unsafe { FFI_ArrowSchema::from_raw(c_schema_ptr) };
            let schema = Schema::try_from(&c_schema)?;

            Operation::Project {
                schema: LanceSchema::try_from(&schema)
                    .expect("Failed to convert from arrow schema to lance schema"),
            }
        }
        _ => unimplemented!(),
    };
    Ok(op)
}
