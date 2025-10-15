// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::blocking_dataset::{BlockingDataset, NATIVE_DATASET};
use crate::error::Result;
use crate::traits::{export_vec, import_vec_from_method, FromJObjectWithEnv, IntoJava, JLance};
use crate::utils::{to_java_map, to_rust_map};
use crate::Error;
use crate::JNIEnvExt;
use arrow::datatypes::Schema;
use arrow_schema::ffi::FFI_ArrowSchema;
use chrono::DateTime;
use jni::objects::{JByteArray, JLongArray, JMap, JObject, JString, JValue, JValueGen};
use jni::sys::jbyte;
use jni::JNIEnv;
use lance::dataset::transaction::{
    DataReplacementGroup, Operation, RewriteGroup, RewrittenIndex, Transaction, TransactionBuilder,
    UpdateMap, UpdateMapEntry, UpdateMode,
};
use lance::table::format::{Fragment, IndexMetadata};
use lance_core::datatypes::Schema as LanceSchema;
use prost::Message;
use prost_types::Any;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

impl IntoJava for &RewriteGroup {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let old_fragments = export_vec(env, &self.old_fragments)?;
        let new_fragments = export_vec(env, &self.new_fragments)?;

        Ok(env.new_object(
            "com/lancedb/lance/operation/RewriteGroup",
            "(Ljava/util/List;Ljava/util/List;)V",
            &[
                JValue::Object(&old_fragments),
                JValue::Object(&new_fragments),
            ],
        )?)
    }
}

impl IntoJava for &RewrittenIndex {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let old_id = self.old_id.into_java(env)?;
        let new_id = self.new_id.into_java(env)?;

        let new_index_details_type_url = env.new_string(self.new_index_details.type_url.clone())?;
        let new_index_details_value = env.byte_array_from_slice(&self.new_index_details.value)?;

        Ok(env.new_object(
            "com/lancedb/lance/operation/RewrittenIndex",
            "(Ljava/util/UUID;Ljava/util/UUID;Ljava/lang/String;[BII)V",
            &[
                JValue::Object(&old_id),
                JValue::Object(&new_id),
                JValue::Object(&new_index_details_type_url),
                JValue::Object(&new_index_details_value),
                JValue::Int(self.new_index_version as i32),
            ],
        )?)
    }
}

impl IntoJava for &DataReplacementGroup {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let fragment_id = self.0;
        let new_file = self.1.into_java(env)?;

        Ok(env.new_object(
            "com/lancedb/lance/operation/DataReplacement$DataReplacementGroup",
            "(JLcom/lancedb/lance/fragment/DataFile;)V",
            &[JValue::Long(fragment_id as i64), JValue::Object(&new_file)],
        )?)
    }
}

impl IntoJava for &IndexMetadata {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let uuid = self.uuid.into_java(env)?;

        let fields = {
            let array_list = env.new_object("java/util/ArrayList", "()V", &[])?;
            for field in &self.fields {
                let field_obj =
                    env.new_object("java/lang/Integer", "(I)V", &[JValue::Int(*field)])?;
                env.call_method(
                    &array_list,
                    "add",
                    "(Ljava/lang/Object;)Z",
                    &[JValue::Object(&field_obj)],
                )?;
            }
            array_list
        };
        let name = env.new_string(&self.name)?;

        let fragments = if let Some(bitmap) = &self.fragment_bitmap {
            let array_list = env.new_object("java/util/ArrayList", "()V", &[])?;
            for frag_id in bitmap.iter() {
                let id_obj =
                    env.new_object("java/lang/Integer", "(I)V", &[JValue::Int(frag_id as i32)])?;
                env.call_method(
                    &array_list,
                    "add",
                    "(Ljava/lang/Object;)Z",
                    &[JValue::Object(&id_obj)],
                )?;
            }
            array_list
        } else {
            JObject::null()
        };

        // Convert index_details to byte array
        let index_details = if let Some(details) = &self.index_details {
            let bytes = details.encode_to_vec();
            let jbytes: &[jbyte] =
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const jbyte, bytes.len()) };

            let byte_array = env.new_byte_array(bytes.len() as i32)?;
            env.set_byte_array_region(&byte_array, 0, jbytes)?;
            byte_array.into()
        } else {
            JObject::null()
        };

        // Convert created_at to Instant
        let created_at = if let Some(dt) = &self.created_at {
            let seconds = dt.timestamp();
            let nanos = dt.timestamp_subsec_nanos() as i64;
            env.call_static_method(
                "java/time/Instant",
                "ofEpochSecond",
                "(JJ)Ljava/time/Instant;",
                &[JValue::Long(seconds), JValue::Long(nanos)],
            )?
            .l()?
        } else {
            JObject::null()
        };

        // Convert base_id from Option<u32> to Integer for Java
        let base_id = if let Some(id) = self.base_id {
            env.new_object("java/lang/Integer", "(I)V", &[JValue::Int(id as i32)])?
        } else {
            JObject::null()
        };

        // Create IndexMetadata object
        Ok(env.new_object(
            "com/lancedb/lance/index/Index",
            "(Ljava/util/UUID;Ljava/util/List;Ljava/lang/String;JLjava/util/List;[BILjava/time/Instant;Ljava/lang/Integer;)V",
            &[
                JValue::Object(&uuid),
                JValue::Object(&fields),
                JValue::Object(&name),
                JValue::Long(self.dataset_version as i64),
                JValue::Object(&fragments),
                JValue::Object(&index_details),
                JValue::Int(self.index_version),
                JValue::Object(&created_at),
                JValue::Object(&base_id),
            ],
        )?)
    }
}

impl IntoJava for &UpdateMode {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let name = match self {
            UpdateMode::RewriteRows => "RewriteRows",
            UpdateMode::RewriteColumns => "RewriteColumns",
        };
        let update_mode_type_class = "com/lancedb/lance/operation/Update$UpdateMode";
        env.get_static_field(
            update_mode_type_class,
            name,
            format!("L{};", update_mode_type_class),
        )?
        .l()
        .map_err(|e| {
            Error::runtime_error(format!("failed to get {}: {}", update_mode_type_class, e))
        })
    }
}

impl FromJObjectWithEnv<RewriteGroup> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<RewriteGroup> {
        let old_fragments: Vec<Fragment> =
            import_vec_from_method(env, self, "oldFragments", |env, fragment| {
                fragment.extract_object(env)
            })?;
        let new_fragments: Vec<Fragment> =
            import_vec_from_method(env, self, "newFragments", |env, fragment| {
                fragment.extract_object(env)
            })?;
        Ok(RewriteGroup {
            old_fragments,
            new_fragments,
        })
    }
}

impl FromJObjectWithEnv<RewrittenIndex> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<RewrittenIndex> {
        let java_old_id = env.get_field(self, "oldId", "Ljava/util/UUID;")?.l()?;
        let java_new_id = env.get_field(self, "newId", "Ljava/util/UUID;")?.l()?;
        let java_old_id = java_old_id.extract_object(env)?;
        let java_new_id = java_new_id.extract_object(env)?;

        let new_index_details_type_url = env
            .get_field(self, "newIndexDetailsTypeUrl", "Ljava/lang/String;")?
            .l()?;
        let new_index_details_type_url: String = env
            .get_string(&JString::from(new_index_details_type_url))?
            .to_str()?
            .to_string();

        let new_index_details_value = env.get_field(self, "newIndexDetailsValue", "[B")?.l()?;
        let new_index_details_value =
            env.convert_byte_array(JByteArray::from(new_index_details_value))?;

        let new_index_version = env.get_field(self, "newIndexVersion", "I")?.i()?;
        Ok(RewrittenIndex {
            old_id: java_old_id,
            new_id: java_new_id,
            new_index_details: prost_types::Any {
                type_url: new_index_details_type_url,
                value: new_index_details_value,
            },
            new_index_version: new_index_version as u32,
        })
    }
}

impl FromJObjectWithEnv<IndexMetadata> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<IndexMetadata> {
        let uuid = env
            .get_field(self, "uuid", "Ljava/util/UUID;")?
            .l()?
            .extract_object(env)?;

        let fields: Vec<i32> = import_vec_from_method(env, self, "fields", |env, field_id| {
            field_id.extract_object(env)
        })?;

        let name = env.get_string_from_method(self, "name")?;
        let dataset_version = env.get_field(self, "datasetVersion", "J")?.j()? as u64;

        let fragment_bitmap: Option<RoaringBitmap> =
            env.get_optional_from_method(self, "fragments", |env, fragments_obj| {
                let frag_ids = env.get_integers(&fragments_obj)?;
                let bitmap = frag_ids
                    .iter()
                    .map(|val| *val as u32)
                    .collect::<RoaringBitmap>();
                Ok(bitmap)
            })?;

        let index_details: Option<Arc<Any>> =
            env.get_optional_from_method(self, "indexDetails", |env, details_obj| {
                let byte_array: JByteArray = details_obj.into();
                let bytes = env.convert_byte_array(&byte_array)?;
                let any = Any::decode(&bytes[..]).map_err(|e| {
                    Error::input_error(format!("Invalid index_details data: {}", e))
                })?;
                Ok(Arc::new(any))
            })?;

        let index_version = env.get_field(self, "indexVersion", "I")?.i()?;
        let created_at =
            env.get_optional_from_method(self, "createdAt", |env, created_at_obj| {
                let seconds = env
                    .call_method(&created_at_obj, "getEpochSecond", "()J", &[])?
                    .j()?;
                let nanos = env
                    .call_method(&created_at_obj, "getNano", "()I", &[])?
                    .i()? as u32;
                Ok(DateTime::from_timestamp(seconds, nanos).unwrap())
            })?;
        let base_id = env.get_optional_u32_from_method(self, "baseId")?;

        Ok(IndexMetadata {
            uuid,
            fields,
            name,
            dataset_version,
            fragment_bitmap,
            index_details,
            index_version,
            created_at,
            base_id,
        })
    }
}

impl FromJObjectWithEnv<DataReplacementGroup> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<DataReplacementGroup> {
        let fragment_id = env.call_method(self, "fragmentId", "()J", &[])?.j()? as u64;
        let new_file = env
            .call_method(
                self,
                "replacedFile",
                "()Lcom/lancedb/lance/fragment/DataFile;",
                &[],
            )?
            .l()?
            .extract_object(env)?;

        Ok(DataReplacementGroup(fragment_id, new_file))
    }
}

impl FromJObjectWithEnv<UpdateMode> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<UpdateMode> {
        let s = env
            .call_method(self, "toString", "()Ljava/lang/String;", &[])?
            .l()?;
        let s: String = env.get_string(&JString::from(s))?.into();
        let t = if s == "RewriteRows" {
            UpdateMode::RewriteRows
        } else {
            UpdateMode::RewriteColumns
        };
        Ok(t)
    }
}

impl IntoJava for Uuid {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let uuid_string = env.new_string(self.to_string())?;
        let uuid_class = env.find_class("java/util/UUID")?;

        env.call_static_method(
            uuid_class,
            "fromString",
            "(Ljava/lang/String;)Ljava/util/UUID;",
            &[JValue::Object(&uuid_string)],
        )?
        .l()
        .map_err(Into::into)
    }
}

impl FromJObjectWithEnv<Uuid> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<Uuid> {
        let uuid_string = env
            .call_method(self, "toString", "()Ljava/lang/String;", &[])?
            .l()?;
        let uuid_string = JString::from(uuid_string);
        let uuid_string: String = env.get_string(&uuid_string)?.into();
        let uuid = Uuid::parse_str(uuid_string.to_string().as_str()).map_err(|e| {
            Error::input_error(format!(
                "Invalid UUID string: {}, error: {}",
                uuid_string, e
            ))
        })?;
        Ok(uuid)
    }
}

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
    let operation = convert_to_java_operation(env, Some(transaction.operation))?;

    let java_transaction = env.new_object(
        "com/lancedb/lance/Transaction",
        "(Lcom/lancedb/lance/Dataset;JLjava/lang/String;Lcom/lancedb/lance/operation/Operation;Ljava/util/Map;Ljava/util/Map;)V",
        &[
            JValue::Object(java_dataset),
            JValue::Long(transaction.read_version as i64),
            JValue::Object(&uuid),
            JValue::Object(&operation),
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
            let java_fragments = export_vec(env, &rust_fragments)?;

            Ok(env.new_object(
                "com/lancedb/lance/operation/Append",
                "(Ljava/util/List;)V",
                &[JValue::Object(&java_fragments)],
            )?)
        }
        Operation::Delete {
            updated_fragments,
            deleted_fragment_ids,
            predicate,
        } => {
            let updated_fragments_obj = export_vec(env, &updated_fragments)?;

            let deleted_ids: Vec<JLance<i64>> = deleted_fragment_ids
                .iter()
                .map(|x| JLance(*x as i64))
                .collect();
            let removed_fragment_ids_obj = export_vec(env, &deleted_ids)?;

            let predicate_obj = env.new_string(&predicate)?;

            Ok(env.new_object(
                "com/lancedb/lance/operation/Delete",
                "(Ljava/util/List;Ljava/util/List;Ljava/lang/String;)V",
                &[
                    JValue::Object(&updated_fragments_obj),
                    JValue::Object(&removed_fragment_ids_obj),
                    JValue::Object(&predicate_obj),
                ],
            )?)
        }
        Operation::Overwrite {
            fragments: rust_fragments,
            schema,
            config_upsert_values,
            initial_bases: _,
        } => {
            let java_fragments = export_vec(env, &rust_fragments)?;
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
        Operation::Update {
            removed_fragment_ids,
            updated_fragments,
            new_fragments,
            fields_modified,
            mem_wal_to_merge: _,
            fields_for_preserving_frag_bitmap,
            update_mode,
        } => {
            let removed_ids: Vec<JLance<i64>> = removed_fragment_ids
                .iter()
                .map(|x| JLance(*x as i64))
                .collect();
            let removed_fragment_ids_obj = export_vec(env, &removed_ids)?;
            let updated_fragments_obj = export_vec(env, &updated_fragments)?;
            let new_fragments_obj = export_vec(env, &new_fragments)?;
            let fields_modified = JLance(fields_modified.clone()).into_java(env)?;
            let fields_for_preserving_frag_bitmap =
                JLance(fields_for_preserving_frag_bitmap.clone()).into_java(env)?;
            let update_mode = match update_mode {
                Some(update_mode) => update_mode.into_java(env),
                None => Ok(JObject::null()),
            }?;
            let update_mode_optional = env
                .call_static_method(
                    "java/util/Optional",
                    "ofNullable",
                    "(Ljava/lang/Object;)Ljava/util/Optional;",
                    &[JValue::Object(&update_mode)],
                )?
                .l()?;
            Ok(env.new_object(
                "com/lancedb/lance/operation/Update",
                "(Ljava/util/List;Ljava/util/List;Ljava/util/List;[J[JLjava/util/Optional;)V",
                &[
                    JValue::Object(&removed_fragment_ids_obj),
                    JValue::Object(&updated_fragments_obj),
                    JValue::Object(&new_fragments_obj),
                    JValueGen::Object(&fields_modified),
                    JValueGen::Object(&fields_for_preserving_frag_bitmap),
                    JValue::Object(&update_mode_optional),
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
        Operation::Rewrite {
            groups,
            rewritten_indices,
            frag_reuse_index,
        } => {
            let java_groups = export_vec(env, &groups)?;
            let java_indices = export_vec(env, &rewritten_indices)?;
            let java_frag_reuse_index = match frag_reuse_index {
                Some(index) => index.into_java(env)?,
                None => JObject::null(),
            };

            Ok(env.new_object(
                "com/lancedb/lance/operation/Rewrite",
                "(Ljava/util/List;Ljava/util/List;Lcom/lancedb/lance/index/Index;)V",
                &[
                    JValue::Object(&java_groups),
                    JValue::Object(&java_indices),
                    JValue::Object(&java_frag_reuse_index),
                ],
            )?)
        }
        Operation::UpdateConfig {
            config_updates,
            table_metadata_updates,
            schema_metadata_updates,
            field_metadata_updates,
        } => {
            let config_updates_obj = export_update_map(env, &config_updates)?;
            let table_metadata_updates_obj = export_update_map(env, &table_metadata_updates)?;
            let schema_metadata_updates_obj = export_update_map(env, &schema_metadata_updates)?;

            // Handle field_metadata_updates
            let field_metadata_updates_obj = if field_metadata_updates.is_empty() {
                JObject::null()
            } else {
                let java_map = env.new_object("java/util/HashMap", "()V", &[])?;
                let map = JMap::from_env(env, &java_map)?;

                for (field_id, update_map) in field_metadata_updates {
                    let java_field_id =
                        env.new_object("java/lang/Integer", "(I)V", &[JValue::Int(field_id)])?;

                    let update_map_obj = export_update_map(env, &Some(update_map.clone()))?;
                    map.put(env, &java_field_id, &update_map_obj)?;
                }
                java_map
            };

            let java_operation = env.new_object(
                "com/lancedb/lance/operation/UpdateConfig",
                "(Lcom/lancedb/lance/operation/UpdateMap;Lcom/lancedb/lance/operation/UpdateMap;Lcom/lancedb/lance/operation/UpdateMap;Ljava/util/Map;)V",
                &[
                    JValue::Object(&config_updates_obj),
                    JValue::Object(&table_metadata_updates_obj),
                    JValue::Object(&schema_metadata_updates_obj),
                    JValue::Object(&field_metadata_updates_obj),
                ],
            )?;
            Ok(java_operation)
        }
        Operation::DataReplacement { replacements } => {
            let java_replacements = export_vec(env, &replacements)?;

            Ok(env.new_object(
                "com/lancedb/lance/operation/DataReplacement",
                "(Ljava/util/List;)V",
                &[JValue::Object(&java_replacements)],
            )?)
        }
        Operation::Merge {
            fragments: rust_fragments,
            schema,
        } => {
            let java_fragments = export_vec(env, &rust_fragments)?;
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
        Operation::Restore { version } => Ok(env.new_object(
            "com/lancedb/lance/operation/Restore",
            "(J)V",
            &[JValue::Long(version as i64)],
        )?),
        Operation::ReserveFragments { num_fragments } => Ok(env.new_object(
            "com/lancedb/lance/operation/ReserveFragments",
            "(I)V",
            &[JValue::Int(num_fragments as i32)],
        )?),
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
    let read_ver = env.get_u64_from_method(&java_transaction, "readVersion")?;
    let uuid = env.get_string_from_method(&java_transaction, "uuid")?;
    let op = env
        .call_method(
            &java_transaction,
            "operation",
            "()Lcom/lancedb/lance/operation/Operation;",
            &[],
        )?
        .l()?;
    let op = convert_to_rust_operation(env, &op, java_dataset)?;

    let transaction_properties = env.get_optional_from_method(
        &java_transaction,
        "transactionProperties",
        |env, transaction_properties| {
            let transaction_properties = JMap::from_env(env, &transaction_properties)?;
            to_rust_map(env, &transaction_properties)
        },
    )?;
    Ok(TransactionBuilder::new(read_ver, op)
        .uuid(uuid)
        .transaction_properties(transaction_properties.map(Arc::new))
        .build())
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

fn convert_to_rust_operation(
    env: &mut JNIEnv<'_>,
    java_operation: &JObject<'_>,
    java_dataset: Option<&JObject<'_>>,
) -> Result<Operation> {
    let op_name = env.get_string_from_method(java_operation, "name")?;
    let op = match op_name.as_str() {
        "Project" => Operation::Project {
            schema: convert_schema_from_operation(env, java_operation, java_dataset.unwrap())?,
        },
        "UpdateConfig" => {
            let config_updates_obj = env
                .call_method(
                    java_operation,
                    "configUpdates",
                    "()Lcom/lancedb/lance/operation/UpdateMap;",
                    &[],
                )?
                .l()?;
            let config_updates = if config_updates_obj.is_null() {
                None
            } else {
                extract_update_map(env, &config_updates_obj)?
            };

            let table_metadata_updates_obj = env
                .call_method(
                    java_operation,
                    "tableMetadataUpdates",
                    "()Lcom/lancedb/lance/operation/UpdateMap;",
                    &[],
                )?
                .l()?;
            let table_metadata_updates = if table_metadata_updates_obj.is_null() {
                None
            } else {
                extract_update_map(env, &table_metadata_updates_obj)?
            };

            let schema_metadata_updates_obj = env
                .call_method(
                    java_operation,
                    "schemaMetadataUpdates",
                    "()Lcom/lancedb/lance/operation/UpdateMap;",
                    &[],
                )?
                .l()?;
            let schema_metadata_updates = if schema_metadata_updates_obj.is_null() {
                None
            } else {
                extract_update_map(env, &schema_metadata_updates_obj)?
            };

            let field_metadata_updates_obj = env
                .call_method(
                    java_operation,
                    "fieldMetadataUpdates",
                    "()Ljava/util/Map;",
                    &[],
                )?
                .l()?;
            let mut field_metadata_updates = HashMap::new();
            if !field_metadata_updates_obj.is_null() {
                let field_metadata_map = JMap::from_env(env, &field_metadata_updates_obj)?;
                let mut iter = field_metadata_map.iter(env)?;
                env.with_local_frame(16, |env| {
                    while let Some((key, value)) = iter.next(env)? {
                        let field_id = env.call_method(&key, "intValue", "()I", &[])?.i()?;
                        if let Some(update_map) = extract_update_map(env, &value)? {
                            field_metadata_updates.insert(field_id, update_map);
                        }
                    }
                    Ok::<(), Error>(())
                })?;
            }

            Operation::UpdateConfig {
                config_updates,
                table_metadata_updates,
                schema_metadata_updates,
                field_metadata_updates,
            }
        }
        "Append" => {
            let fragments =
                import_vec_from_method(env, java_operation, "fragments", |env, fragment| {
                    fragment.extract_object(env)
                })?;
            Operation::Append { fragments }
        }
        "Delete" => {
            let updated_fragments: Vec<Fragment> = import_vec_from_method(
                env,
                java_operation,
                "updatedFragments",
                |env, fragment| fragment.extract_object(env),
            )?;

            let deleted_fragment_ids: Vec<u64> = import_vec_from_method(
                env,
                java_operation,
                "deletedFragmentIds",
                |env, fragment_id| {
                    Ok(env.call_method(fragment_id, "longValue", "()J", &[])?.j()? as u64)
                },
            )?;

            let predicate = env.get_string_from_method(java_operation, "predicate")?;

            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids,
                predicate,
            }
        }
        "Overwrite" => {
            let fragments: Vec<Fragment> =
                import_vec_from_method(env, java_operation, "fragments", |env, fragment| {
                    fragment.extract_object(env)
                })?;

            let config_upsert_values = env.get_optional_from_method(
                java_operation,
                "configUpsertValues",
                |env, config_upsert_values| {
                    let config_upsert_values = JMap::from_env(env, &config_upsert_values)?;
                    to_rust_map(env, &config_upsert_values)
                },
            )?;
            let schema = convert_schema_from_operation(env, java_operation, java_dataset.unwrap())?;
            Operation::Overwrite {
                fragments,
                schema,
                config_upsert_values,
                initial_bases: None,
            }
        }
        "Rewrite" => {
            let groups: Vec<RewriteGroup> =
                import_vec_from_method(env, java_operation, "groups", |env, group| {
                    group.extract_object(env)
                })?;

            let rewritten_indices: Vec<RewrittenIndex> =
                import_vec_from_method(env, java_operation, "rewrittenIndices", |env, index| {
                    index.extract_object(env)
                })?;

            let frag_reuse_index: Option<IndexMetadata> = env.get_optional_from_method(
                java_operation,
                "fragReuseIndex",
                |env, frag_reuse_index| frag_reuse_index.extract_object(env),
            )?;

            Operation::Rewrite {
                groups,
                rewritten_indices,
                frag_reuse_index,
            }
        }
        "Update" => {
            let removed_fragment_ids = import_vec_from_method(
                env,
                java_operation,
                "removedFragmentIds",
                |env, fragment_id| {
                    Ok(env.call_method(fragment_id, "longValue", "()J", &[])?.j()? as u64)
                },
            )?;

            let updated_fragments: Vec<Fragment> = import_vec_from_method(
                env,
                java_operation,
                "updatedFragments",
                |env, fragment| fragment.extract_object(env),
            )?;

            let new_fragments: Vec<Fragment> =
                import_vec_from_method(env, java_operation, "newFragments", |env, fragment| {
                    fragment.extract_object(env)
                })?;

            let fields_modified = env
                .call_method(java_operation, "fieldsModified", "()[J", &[])?
                .l()?;
            let fields_modified = JLongArray::from(fields_modified).extract_object(env)?;

            let fields_for_preserving_frag_bitmap = env
                .call_method(java_operation, "fieldsForPreservingFragBitmap", "()[J", &[])?
                .l()?;
            let fields_for_preserving_frag_bitmap =
                JLongArray::from(fields_for_preserving_frag_bitmap).extract_object(env)?;

            let update_mode: Option<UpdateMode> =
                env.get_optional_from_method(java_operation, "updateMode", |env, update_mode| {
                    update_mode.extract_object(env)
                })?;

            Operation::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
                fields_modified,
                mem_wal_to_merge: None,
                fields_for_preserving_frag_bitmap,
                update_mode,
            }
        }
        "DataReplacement" => {
            let replacements: Vec<DataReplacementGroup> =
                import_vec_from_method(env, java_operation, "replacements", |env, replacement| {
                    replacement.extract_object(env)
                })?;
            Operation::DataReplacement { replacements }
        }
        "Merge" => {
            let fragments: Vec<Fragment> =
                import_vec_from_method(env, java_operation, "fragments", |env, fragment| {
                    fragment.extract_object(env)
                })?;
            Operation::Merge {
                fragments,
                schema: convert_schema_from_operation(env, java_operation, java_dataset.unwrap())?,
            }
        }
        "Restore" => {
            let version: u64 = env
                .call_method(java_operation, "version", "()J", &[])?
                .j()? as u64;
            return Ok(Operation::Restore { version });
        }
        "ReserveFragments" => {
            let num_fragments = env
                .call_method(java_operation, "numFragments", "()I", &[])?
                .i()? as u32;
            return Ok(Operation::ReserveFragments { num_fragments });
        }
        "CreateIndex" => {
            let new_indices =
                import_vec_from_method(env, java_operation, "getNewIndices", |env, index| {
                    index.extract_object(env)
                })?;
            let removed_indices =
                import_vec_from_method(env, java_operation, "getRemovedIndices", |env, index| {
                    index.extract_object(env)
                })?;
            return Ok(Operation::CreateIndex {
                new_indices,
                removed_indices,
            });
        }
        _ => unimplemented!(),
    };
    Ok(op)
}

fn extract_update_map(env: &mut JNIEnv, update_map_obj: &JObject) -> Result<Option<UpdateMap>> {
    if update_map_obj.is_null() {
        return Ok(None);
    }

    let updates_obj = env
        .call_method(update_map_obj, "updates", "()Ljava/util/Map;", &[])?
        .l()?;
    let replace = env
        .call_method(update_map_obj, "replace", "()Z", &[])?
        .z()?;

    if updates_obj.is_null() {
        return Ok(None);
    }

    let updates_map = JMap::from_env(env, &updates_obj)?;
    let mut entries = Vec::new();
    let mut iter = updates_map.iter(env)?;

    env.with_local_frame(16, |env| {
        while let Some((key, value)) = iter.next(env)? {
            let key_jstring = JString::from(key);
            let key_string: String = env.get_string(&key_jstring)?.into();

            let value_string = if value.is_null() {
                None
            } else {
                let value_jstring = JString::from(value);
                let value_str = env.get_string(&value_jstring)?.into();
                Some(value_str)
            };

            entries.push(UpdateMapEntry {
                key: key_string,
                value: value_string,
            });
        }
        Ok::<(), Error>(())
    })?;

    Ok(Some(UpdateMap {
        update_entries: entries,
        replace,
    }))
}

fn export_update_map<'a>(
    env: &mut JNIEnv<'a>,
    update_map: &Option<UpdateMap>,
) -> Result<JObject<'a>> {
    match update_map {
        None => Ok(JObject::null()),
        Some(map) => {
            // Create a Java HashMap for the updates
            let updates_map = env.new_object("java/util/HashMap", "()V", &[])?;
            let jmap = JMap::from_env(env, &updates_map)?;

            for entry in &map.update_entries {
                let key = env.new_string(&entry.key)?;
                let value = match &entry.value {
                    Some(val) => JObject::from(env.new_string(val)?),
                    None => JObject::null(),
                };
                jmap.put(env, &key, &value)?;
            }

            // Create UpdateMap object
            let update_map_obj = env.new_object(
                "com/lancedb/lance/operation/UpdateMap",
                "(Ljava/util/Map;Z)V",
                &[
                    JValue::Object(&updates_map),
                    JValue::Bool(map.replace as u8),
                ],
            )?;
            Ok(update_map_obj)
        }
    }
}
