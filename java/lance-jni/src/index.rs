// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::error::Result;
use crate::traits::{export_vec, IntoJava};
use jni::objects::{JObject, JValue};
use jni::sys::jbyte;
use jni::JNIEnv;
use lance::table::format::IndexMetadata;
use lance_index::IndexDescription;
use prost::Message;
use prost_types::Any;
use std::sync::Arc;

impl IntoJava for &Arc<dyn IndexDescription> {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let field_ids_list = {
            let array_list = env.new_object("java/util/ArrayList", "()V", &[])?;
            for id in self.field_ids() {
                let int_obj =
                    env.new_object("java/lang/Integer", "(I)V", &[JValue::Int(*id as i32)])?;
                env.call_method(
                    &array_list,
                    "add",
                    "(Ljava/lang/Object;)Z",
                    &[JValue::Object(&int_obj)],
                )?;
            }
            array_list
        };
        let name = env.new_string(self.name())?;
        let type_url = env.new_string(self.type_url())?;
        let index_type = env.new_string(self.index_type())?;
        let rows_indexed = self.rows_indexed() as i64;
        let metadata_list = export_vec(env, self.metadata())?;
        let details_json = self.details()?;
        let details = env.new_string(details_json)?;

        let j_index_desc = env.new_object(
            "org/lance/index/IndexDescription",
            "(Ljava/lang/String;Ljava/util/List;Ljava/lang/String;Ljava/lang/String;JLjava/util/List;Ljava/lang/String;)V",
            &[
                JValue::Object(&name),
                JValue::Object(&field_ids_list),
                JValue::Object(&type_url),
                JValue::Object(&index_type),
                JValue::Long(rows_indexed),
                JValue::Object(&metadata_list),
                JValue::Object(&details),
            ],
        )?;
        Ok(j_index_desc)
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

        // Determine index type from index_details type_url
        let index_type = determine_index_type(env, &self.index_details)?;

        // Create Index object
        Ok(env.new_object(
            "org/lance/index/Index",
            "(Ljava/util/UUID;Ljava/util/List;Ljava/lang/String;JLjava/util/List;[BILjava/time/Instant;Ljava/lang/Integer;Lorg/lance/index/IndexType;)V",
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
                JValue::Object(&index_type),
            ],
        )?)
    }
}

/// Determine the IndexType enum value from index_details protobuf
fn determine_index_type<'local>(
    env: &mut JNIEnv<'local>,
    index_details: &Option<Arc<Any>>,
) -> Result<JObject<'local>> {
    let type_name = if let Some(details) = index_details {
        // Extract type name from type_url (e.g., ".lance.index.BTreeIndexDetails" -> "BTREE")
        let type_url = &details.type_url;
        let type_part = type_url.split('.').next_back().unwrap_or("");
        let lower = type_part.to_lowercase();

        if lower.contains("btree") {
            Some("BTREE")
        } else if lower.contains("bitmap") {
            Some("BITMAP")
        } else if lower.contains("labellist") {
            Some("LABEL_LIST")
        } else if lower.contains("inverted") {
            Some("INVERTED")
        } else if lower.contains("ngram") {
            Some("NGRAM")
        } else if lower.contains("zonemap") {
            Some("ZONEMAP")
        } else if lower.contains("bloomfilter") {
            Some("BLOOM_FILTER")
        } else if lower.contains("ivfhnsw") {
            if lower.contains("sq") {
                Some("IVF_HNSW_SQ")
            } else if lower.contains("pq") {
                Some("IVF_HNSW_PQ")
            } else {
                Some("IVF_HNSW_FLAT")
            }
        } else if lower.contains("ivf") {
            if lower.contains("sq") {
                Some("IVF_SQ")
            } else if lower.contains("pq") {
                Some("IVF_PQ")
            } else {
                Some("IVF_FLAT")
            }
        } else if lower.contains("vector") {
            Some("VECTOR")
        } else {
            None
        }
    } else {
        None
    };

    match type_name {
        Some(name) => {
            let index_type = env
                .get_static_field(
                    "org/lance/index/IndexType",
                    name,
                    "Lorg/lance/index/IndexType;",
                )?
                .l()?;
            Ok(index_type)
        }
        None => Ok(JObject::null()),
    }
}
