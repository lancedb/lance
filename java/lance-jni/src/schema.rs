// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::error::{Error, Result};
use crate::traits::IntoJava;
use crate::utils::to_java_map;
use arrow::datatypes::DataType;
use arrow_schema::{TimeUnit, UnionFields};
use jni::objects::{JObject, JValue};
use jni::sys::{jboolean, jint};
use jni::JNIEnv;
use lance_core::datatypes::{Field, Schema};

impl IntoJava for Schema {
    fn into_java<'local>(self, env: &mut JNIEnv<'local>) -> Result<JObject<'local>> {
        let jfield_list = env.new_object("java/util/ArrayList", "()V", &[])?;
        for lance_field in self.fields.iter() {
            let java_field = convert_to_java_field(env, lance_field)?;
            env.call_method(
                &jfield_list,
                "add",
                "(Ljava/lang/Object;)Z",
                &[JValue::Object(&java_field)],
            )?;
        }
        let metadata = to_java_map(env, &self.metadata)?;
        Ok(env.new_object(
            "com/lancedb/lance/schema/LanceSchema",
            "(Ljava/util/List;Ljava/util/Map;)V",
            &[JValue::Object(&jfield_list), JValue::Object(&metadata)],
        )?)
    }
}

pub fn convert_to_java_field<'local>(
    env: &mut JNIEnv<'local>,
    lance_field: &Field,
) -> Result<JObject<'local>> {
    let name = env.new_string(&lance_field.name)?;
    let children = convert_children_fields(env, lance_field)?;
    let metadata = to_java_map(env, &lance_field.metadata)?;
    let arrow_type = convert_arrow_type(env, &lance_field.data_type())?;
    let ctor_sig = "(IILjava/lang/String;".to_owned()
        + "ZLorg/apache/arrow/vector/types/pojo/ArrowType;"
        + "Lorg/apache/arrow/vector/types/pojo/DictionaryEncoding;"
        + "Ljava/util/Map;"
        + "Ljava/util/List;Z)V";
    let field_obj = env.new_object(
        "com/lancedb/lance/schema/LanceField",
        ctor_sig.as_str(),
        &[
            JValue::Int(lance_field.id as jint),
            JValue::Int(lance_field.parent_id as jint),
            JValue::Object(&JObject::from(name)),
            JValue::Bool(lance_field.nullable as jboolean),
            JValue::Object(&arrow_type),
            JValue::Object(&JObject::null()),
            JValue::Object(&metadata),
            JValue::Object(&children),
            JValue::Bool(lance_field.unenforced_primary_key as jboolean),
        ],
    )?;

    Ok(field_obj)
}

fn convert_children_fields<'local>(
    env: &mut JNIEnv<'local>,
    lance_field: &Field,
) -> Result<JObject<'local>> {
    let children_list = env.new_object("java/util/ArrayList", "()V", &[])?;
    for lance_field in lance_field.children.iter() {
        let field = convert_to_java_field(env, lance_field)?;
        env.call_method(
            &children_list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValue::Object(&field)],
        )?;
    }
    Ok(children_list)
}

pub fn convert_arrow_type<'local>(
    env: &mut JNIEnv<'local>,
    arrow_type: &DataType,
) -> Result<JObject<'local>> {
    match arrow_type {
        DataType::Null => convert_null_type(env),
        DataType::Boolean => convert_boolean_type(env),
        DataType::Int8 => convert_int_type(env, 8, true),
        DataType::Int16 => convert_int_type(env, 16, true),
        DataType::Int32 => convert_int_type(env, 32, true),
        DataType::Int64 => convert_int_type(env, 64, true),
        DataType::UInt8 => convert_int_type(env, 8, false),
        DataType::UInt16 => convert_int_type(env, 16, false),
        DataType::UInt32 => convert_int_type(env, 32, false),
        DataType::UInt64 => convert_int_type(env, 64, false),
        DataType::Float16 => convert_floating_point_type(env, "HALF"),
        DataType::Float32 => convert_floating_point_type(env, "SINGLE"),
        DataType::Float64 => convert_floating_point_type(env, "DOUBLE"),
        DataType::Utf8 => convert_utf8_type(env, false),
        DataType::LargeUtf8 => convert_utf8_type(env, true),
        DataType::Binary => convert_binary_type(env, false),
        DataType::LargeBinary => convert_binary_type(env, true),
        DataType::FixedSizeBinary(len) => convert_fixed_size_binary_type(env, *len),
        DataType::Date32 => convert_date_type(env, "DAY"),
        DataType::Date64 => convert_date_type(env, "MILLISECOND"),
        DataType::Time32(unit) => convert_time_type(env, *unit, 32),
        DataType::Time64(unit) => convert_time_type(env, *unit, 64),
        DataType::Timestamp(unit, tz) => convert_timestamp_type(env, *unit, tz.as_deref()),
        DataType::Duration(unit) => convert_duration_type(env, *unit),
        DataType::Decimal128(precision, scale) => {
            convert_decimal_type(env, *precision, *scale, 128)
        }
        DataType::Decimal256(precision, scale) => {
            convert_decimal_type(env, *precision, *scale, 256)
        }
        DataType::List(..) => convert_list_type(env, false),
        DataType::LargeList(..) => convert_list_type(env, true),
        DataType::FixedSizeList(.., len) => convert_fixed_size_list_type(env, *len),
        DataType::Struct(..) => convert_struct_type(env),
        DataType::Union(fields, mode) => convert_union_type(env, fields, *mode),
        DataType::Map(.., keys_sorted) => convert_map_type(env, *keys_sorted),
        _ => Err(Error::input_error(
            "ArrowSchema conversion error".to_string(),
        )),
    }
}

fn convert_null_type<'local>(env: &mut JNIEnv<'local>) -> Result<JObject<'local>> {
    Ok(env
        .get_static_field(
            "org/apache/arrow/vector/types/pojo/ArrowType$Null",
            "INSTANCE",
            "Lorg/apache/arrow/vector/types/pojo/ArrowType$Null;",
        )?
        .l()?)
}

fn convert_boolean_type<'local>(env: &mut JNIEnv<'local>) -> Result<JObject<'local>> {
    Ok(env
        .get_static_field(
            "org/apache/arrow/vector/types/pojo/ArrowType$Bool",
            "INSTANCE",
            "Lorg/apache/arrow/vector/types/pojo/ArrowType$Bool;",
        )?
        .l()?)
}

fn convert_int_type<'local>(
    env: &mut JNIEnv<'local>,
    bit_width: i32,
    is_signed: bool,
) -> Result<JObject<'local>> {
    Ok(env.new_object(
        "org/apache/arrow/vector/types/pojo/ArrowType$Int",
        "(IZ)V",
        &[
            JValue::Int(bit_width as jint),
            JValue::Bool(is_signed as jboolean),
        ],
    )?)
}

fn convert_floating_point_type<'local>(
    env: &mut JNIEnv<'local>,
    precision: &str,
) -> Result<JObject<'local>> {
    let precision_enum = env
        .get_static_field(
            "org/apache/arrow/vector/types/FloatingPointPrecision",
            precision,
            "Lorg/apache/arrow/vector/types/FloatingPointPrecision;",
        )?
        .l()?;

    Ok(env.new_object(
        "org/apache/arrow/vector/types/pojo/ArrowType$FloatingPoint",
        "(Lorg/apache/arrow/vector/types/FloatingPointPrecision;)V",
        &[JValue::Object(&precision_enum)],
    )?)
}

fn convert_utf8_type<'local>(env: &mut JNIEnv<'local>, is_large: bool) -> Result<JObject<'local>> {
    let class_name = if is_large {
        "org/apache/arrow/vector/types/pojo/ArrowType$LargeUtf8"
    } else {
        "org/apache/arrow/vector/types/pojo/ArrowType$Utf8"
    };

    convert_arrow_type_by_class_name(env, class_name)
}

fn convert_binary_type<'local>(
    env: &mut JNIEnv<'local>,
    is_large: bool,
) -> Result<JObject<'local>> {
    let class_name = if is_large {
        "org/apache/arrow/vector/types/pojo/ArrowType$LargeBinary"
    } else {
        "org/apache/arrow/vector/types/pojo/ArrowType$Binary"
    };

    convert_arrow_type_by_class_name(env, class_name)
}

fn convert_arrow_type_by_class_name<'local>(
    env: &mut JNIEnv<'local>,
    class_name: &str,
) -> Result<JObject<'local>> {
    let class = env.find_class(class_name)?;
    let field_sig = format!("L{};", class_name);
    let instance = env.get_static_field(class, "INSTANCE", &field_sig)?.l()?;
    Ok(instance)
}

fn convert_fixed_size_binary_type<'local>(
    env: &mut JNIEnv<'local>,
    byte_width: i32,
) -> Result<JObject<'local>> {
    let class = env.find_class("org/apache/arrow/vector/types/pojo/ArrowType$FixedSizeBinary")?;
    Ok(env.new_object(class, "(I)V", &[JValue::Int(byte_width)])?)
}

fn convert_date_type<'local>(env: &mut JNIEnv<'local>, unit: &str) -> Result<JObject<'local>> {
    let class = env.find_class("org/apache/arrow/vector/types/pojo/ArrowType$Date")?;
    let unit_enum = env
        .get_static_field(
            "org/apache/arrow/vector/types/DateUnit",
            unit,
            "Lorg/apache/arrow/vector/types/DateUnit;",
        )?
        .l()?;

    Ok(env.new_object(
        class,
        "(Lorg/apache/arrow/vector/types/DateUnit;)V",
        &[JValue::Object(&unit_enum)],
    )?)
}

fn convert_time_type<'local>(
    env: &mut JNIEnv<'local>,
    unit: TimeUnit,
    bit_width: i32,
) -> Result<JObject<'local>> {
    let class = env.find_class("org/apache/arrow/vector/types/pojo/ArrowType$Time")?;
    let unit_str = match unit {
        TimeUnit::Second => "SECOND",
        TimeUnit::Millisecond => "MILLISECOND",
        TimeUnit::Microsecond => "MICROSECOND",
        TimeUnit::Nanosecond => "NANOSECOND",
    };

    let unit_enum = env
        .get_static_field(
            "org/apache/arrow/vector/types/TimeUnit",
            unit_str,
            "Lorg/apache/arrow/vector/types/TimeUnit;",
        )?
        .l()?;

    Ok(env.new_object(
        class,
        "(Lorg/apache/arrow/vector/types/TimeUnit;I)V",
        &[JValue::Object(&unit_enum), JValue::Int(bit_width)],
    )?)
}

fn convert_timestamp_type<'local>(
    env: &mut JNIEnv<'local>,
    unit: TimeUnit,
    timezone: Option<&str>,
) -> Result<JObject<'local>> {
    let class = env.find_class("org/apache/arrow/vector/types/pojo/ArrowType$Timestamp")?;
    let unit_str = match unit {
        TimeUnit::Second => "SECOND",
        TimeUnit::Millisecond => "MILLISECOND",
        TimeUnit::Microsecond => "MICROSECOND",
        TimeUnit::Nanosecond => "NANOSECOND",
    };

    let unit_enum = env
        .get_static_field(
            "org/apache/arrow/vector/types/TimeUnit",
            unit_str,
            "Lorg/apache/arrow/vector/types/TimeUnit;",
        )?
        .l()?;

    let timezone_str = timezone.unwrap_or("-");
    let j_timezone = env.new_string(timezone_str)?;

    Ok(env.new_object(
        class,
        "(Lorg/apache/arrow/vector/types/TimeUnit;Ljava/lang/String;)V",
        &[JValue::Object(&unit_enum), JValue::Object(&j_timezone)],
    )?)
}

fn convert_duration_type<'local>(
    env: &mut JNIEnv<'local>,
    unit: TimeUnit,
) -> Result<JObject<'local>> {
    let class = env.find_class("org/apache/arrow/vector/types/pojo/ArrowType$Duration")?;
    let unit_str = match unit {
        TimeUnit::Second => "SECOND",
        TimeUnit::Millisecond => "MILLISECOND",
        TimeUnit::Microsecond => "MICROSECOND",
        TimeUnit::Nanosecond => "NANOSECOND",
    };

    let unit_enum = env
        .get_static_field(
            "org/apache/arrow/vector/types/TimeUnit",
            unit_str,
            "Lorg/apache/arrow/vector/types/TimeUnit;",
        )?
        .l()?;

    Ok(env.new_object(
        class,
        "(Lorg/apache/arrow/vector/types/TimeUnit;)V",
        &[JValue::Object(&unit_enum)],
    )?)
}

fn convert_decimal_type<'local>(
    env: &mut JNIEnv<'local>,
    precision: u8,
    scale: i8,
    bit_width: i32,
) -> Result<JObject<'local>> {
    let class = env.find_class("org/apache/arrow/vector/types/pojo/ArrowType$Decimal")?;
    Ok(env.new_object(
        class,
        "(III)V",
        &[
            JValue::Int(precision as jint),
            JValue::Int(scale as jint),
            JValue::Int(bit_width),
        ],
    )?)
}

fn convert_list_type<'local>(env: &mut JNIEnv<'local>, is_large: bool) -> Result<JObject<'local>> {
    let class_name = if is_large {
        "org/apache/arrow/vector/types/pojo/ArrowType$LargeList"
    } else {
        "org/apache/arrow/vector/types/pojo/ArrowType$List"
    };

    convert_arrow_type_by_class_name(env, class_name)
}

fn convert_fixed_size_list_type<'local>(
    env: &mut JNIEnv<'local>,
    list_size: i32,
) -> Result<JObject<'local>> {
    Ok(env.new_object(
        "org/apache/arrow/vector/types/pojo/ArrowType$FixedSizeList",
        "(I)V",
        &[JValue::Int(list_size)],
    )?)
}

fn convert_struct_type<'local>(env: &mut JNIEnv<'local>) -> Result<JObject<'local>> {
    Ok(env
        .get_static_field(
            "org/apache/arrow/vector/types/pojo/ArrowType$Struct",
            "INSTANCE",
            "Lorg/apache/arrow/vector/types/pojo/ArrowType$Struct;",
        )?
        .l()?)
}

fn convert_union_type<'local>(
    env: &mut JNIEnv<'local>,
    fields: &UnionFields,
    mode: arrow_schema::UnionMode,
) -> Result<JObject<'local>> {
    let class = env.find_class("org/apache/arrow/vector/types/pojo/ArrowType$Union")?;

    let mode_str = match mode {
        arrow_schema::UnionMode::Sparse => "SPARSE",
        arrow_schema::UnionMode::Dense => "DENSE",
    };
    let mode_enum = env
        .get_static_field(
            "org/apache/arrow/vector/types/UnionMode",
            mode_str,
            "Lorg/apache/arrow/vector/types/UnionMode;",
        )?
        .l()?;

    let jarray = env.new_int_array(fields.size() as jint)?;

    let mut rust_array = vec![0; fields.size()];
    for (i, (type_id, _)) in fields.iter().enumerate() {
        rust_array[i] = type_id as i32;
    }
    env.set_int_array_region(&jarray, 0, &rust_array)?;

    Ok(env.new_object(
        class,
        "(Lorg/apache/arrow/vector/types/UnionMode;[I)V",
        &[JValue::Object(&mode_enum), JValue::Object(&jarray)],
    )?)
}

fn convert_map_type<'local>(
    env: &mut JNIEnv<'local>,
    keys_sorted: bool,
) -> Result<JObject<'local>> {
    Ok(env.new_object(
        "org/apache/arrow/vector/types/pojo/ArrowType$Map",
        "(Z)V",
        &[JValue::Bool(keys_sorted as jboolean)],
    )?)
}
