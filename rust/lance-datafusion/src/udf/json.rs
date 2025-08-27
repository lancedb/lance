// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::builder::{BooleanBuilder, Float64Builder, Int64Builder, StringBuilder};
use arrow_array::{Array, ArrayRef, LargeBinaryArray, StringArray};
use arrow_schema::DataType;
use datafusion::error::Result;
use datafusion::logical_expr::{ScalarUDF, Volatility};
use datafusion::physical_plan::ColumnarValue;
use datafusion::prelude::create_udf;
use std::sync::Arc;

/// Create the json_extract UDF for extracting JSONPath from JSONB data
pub fn json_extract_udf() -> ScalarUDF {
    create_udf(
        "json_extract",
        vec![DataType::LargeBinary, DataType::Utf8],
        DataType::Utf8,
        Volatility::Immutable,
        Arc::new(json_extract_columnar_impl),
    )
}

/// Implementation of json_extract function with ColumnarValue
fn json_extract_columnar_impl(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let arrays: Vec<ArrayRef> = args
        .iter()
        .map(|arg| match arg {
            ColumnarValue::Array(arr) => arr.clone(),
            ColumnarValue::Scalar(scalar) => scalar.to_array().unwrap(),
        })
        .collect();

    let result = json_extract_impl(&arrays)?;
    Ok(ColumnarValue::Array(result))
}

/// Implementation of json_extract function
fn json_extract_impl(args: &[ArrayRef]) -> Result<ArrayRef> {
    if args.len() != 2 {
        return Err(datafusion::error::DataFusionError::Execution(
            "json_extract requires exactly 2 arguments".to_string(),
        ));
    }

    let jsonb_array = args[0]
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "First argument must be LargeBinary".to_string(),
            )
        })?;

    let path_array = args[1]
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "Second argument must be String".to_string(),
            )
        })?;

    let mut builder = StringBuilder::new();

    for i in 0..jsonb_array.len() {
        if jsonb_array.is_null(i) || path_array.is_null(i) {
            builder.append_null();
        } else {
            let jsonb_bytes = jsonb_array.value(i);
            let path = path_array.value(i);

            match extract_json_path(jsonb_bytes, path) {
                Ok(Some(value)) => builder.append_value(&value),
                Ok(None) => builder.append_null(),
                Err(e) => {
                    return Err(datafusion::error::DataFusionError::Execution(format!(
                        "Failed to extract JSONPath: {}",
                        e
                    )));
                }
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Extract value from JSONB using JSONPath
fn extract_json_path(jsonb_bytes: &[u8], path: &str) -> Result<Option<String>> {
    let json_path = jsonb::jsonpath::parse_json_path(path.as_bytes()).map_err(|e| {
        datafusion::error::DataFusionError::Execution(format!("Invalid JSONPath: {}", e))
    })?;

    let raw_jsonb = jsonb::RawJsonb::new(jsonb_bytes);
    let mut selector = jsonb::jsonpath::Selector::new(raw_jsonb);
    match selector.select_values(&json_path) {
        Ok(values) => {
            if values.is_empty() {
                Ok(None)
            } else {
                // Return the first matched value
                Ok(Some(values[0].to_string()))
            }
        }
        Err(_) => Ok(None), // Path not found or error
    }
}

/// Create the json_exists UDF for checking if a JSONPath exists
pub fn json_exists_udf() -> ScalarUDF {
    create_udf(
        "json_exists",
        vec![DataType::LargeBinary, DataType::Utf8],
        DataType::Boolean,
        Volatility::Immutable,
        Arc::new(json_exists_columnar_impl),
    )
}

/// Implementation of json_exists function with ColumnarValue
fn json_exists_columnar_impl(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let arrays: Vec<ArrayRef> = args
        .iter()
        .map(|arg| match arg {
            ColumnarValue::Array(arr) => arr.clone(),
            ColumnarValue::Scalar(scalar) => scalar.to_array().unwrap(),
        })
        .collect();

    let result = json_exists_impl(&arrays)?;
    Ok(ColumnarValue::Array(result))
}

/// Implementation of json_exists function
fn json_exists_impl(args: &[ArrayRef]) -> Result<ArrayRef> {
    if args.len() != 2 {
        return Err(datafusion::error::DataFusionError::Execution(
            "json_exists requires exactly 2 arguments".to_string(),
        ));
    }

    let jsonb_array = args[0]
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "First argument must be LargeBinary".to_string(),
            )
        })?;

    let path_array = args[1]
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "Second argument must be String".to_string(),
            )
        })?;

    let mut builder = BooleanBuilder::new();

    for i in 0..jsonb_array.len() {
        if jsonb_array.is_null(i) || path_array.is_null(i) {
            builder.append_null();
        } else {
            let jsonb_bytes = jsonb_array.value(i);
            let path = path_array.value(i);

            match check_json_path_exists(jsonb_bytes, path) {
                Ok(exists) => builder.append_value(exists),
                Err(e) => {
                    return Err(datafusion::error::DataFusionError::Execution(format!(
                        "Failed to check JSONPath existence: {}",
                        e
                    )));
                }
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Check if a JSONPath exists in JSONB
fn check_json_path_exists(jsonb_bytes: &[u8], path: &str) -> Result<bool> {
    let json_path = jsonb::jsonpath::parse_json_path(path.as_bytes()).map_err(|e| {
        datafusion::error::DataFusionError::Execution(format!("Invalid JSONPath: {}", e))
    })?;

    let raw_jsonb = jsonb::RawJsonb::new(jsonb_bytes);
    let mut selector = jsonb::jsonpath::Selector::new(raw_jsonb);
    match selector.exists(&json_path) {
        Ok(exists) => Ok(exists),
        Err(_) => Ok(false),
    }
}

/// Create the json_get UDF for getting a field value as JSON string
pub fn json_get_udf() -> ScalarUDF {
    create_udf(
        "json_get",
        vec![DataType::LargeBinary, DataType::Utf8],
        DataType::LargeBinary,
        Volatility::Immutable,
        Arc::new(json_get_columnar_impl),
    )
}

/// Implementation of json_get function with ColumnarValue
fn json_get_columnar_impl(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let arrays: Vec<ArrayRef> = args
        .iter()
        .map(|arg| match arg {
            ColumnarValue::Array(arr) => arr.clone(),
            ColumnarValue::Scalar(scalar) => scalar.to_array().unwrap(),
        })
        .collect();

    let result = json_get_impl(&arrays)?;
    Ok(ColumnarValue::Array(result))
}

/// Implementation of json_get function
fn json_get_impl(args: &[ArrayRef]) -> Result<ArrayRef> {
    if args.len() != 2 {
        return Err(datafusion::error::DataFusionError::Execution(
            "json_get requires exactly 2 arguments".to_string(),
        ));
    }

    let jsonb_array = args[0]
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "First argument must be LargeBinary".to_string(),
            )
        })?;

    let key_array = args[1]
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "Second argument must be String".to_string(),
            )
        })?;

    let mut builder = arrow_array::builder::LargeBinaryBuilder::new();

    for i in 0..jsonb_array.len() {
        if jsonb_array.is_null(i) || key_array.is_null(i) {
            builder.append_null();
        } else {
            let jsonb_bytes = jsonb_array.value(i);
            let key = key_array.value(i);

            match get_json_field(jsonb_bytes, key) {
                Ok(Some(value)) => builder.append_value(value),
                Ok(None) => builder.append_null(),
                Err(e) => {
                    return Err(datafusion::error::DataFusionError::Execution(format!(
                        "Failed to get JSON field: {}",
                        e
                    )));
                }
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Get a field value from JSONB (returns JSONB bytes)
fn get_json_field(jsonb_bytes: &[u8], key: &str) -> Result<Option<Vec<u8>>> {
    let raw_jsonb = jsonb::RawJsonb::new(jsonb_bytes);

    // Try as object field first
    match raw_jsonb.get_by_name(key, false) {
        Ok(Some(value)) => return Ok(Some(value.as_raw().as_ref().to_vec())),
        Ok(None) => {}
        Err(e) => {
            return Err(datafusion::error::DataFusionError::Execution(format!(
                "Failed to get field: {}",
                e
            )))
        }
    }

    // Try as array index
    if let Ok(index) = key.parse::<usize>() {
        match raw_jsonb.get_by_index(index) {
            Ok(Some(value)) => return Ok(Some(value.as_raw().as_ref().to_vec())),
            Ok(None) => {}
            Err(e) => {
                return Err(datafusion::error::DataFusionError::Execution(format!(
                    "Failed to get array element: {}",
                    e
                )))
            }
        }
    }

    Ok(None)
}

/// Create the json_get_string UDF for getting a string value
pub fn json_get_string_udf() -> ScalarUDF {
    create_udf(
        "json_get_string",
        vec![DataType::LargeBinary, DataType::Utf8],
        DataType::Utf8,
        Volatility::Immutable,
        Arc::new(json_get_string_columnar_impl),
    )
}

/// Implementation of json_get_string function with ColumnarValue
fn json_get_string_columnar_impl(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let arrays: Vec<ArrayRef> = args
        .iter()
        .map(|arg| match arg {
            ColumnarValue::Array(arr) => arr.clone(),
            ColumnarValue::Scalar(scalar) => scalar.to_array().unwrap(),
        })
        .collect();

    let result = json_get_string_impl(&arrays)?;
    Ok(ColumnarValue::Array(result))
}

/// Implementation of json_get_string function
fn json_get_string_impl(args: &[ArrayRef]) -> Result<ArrayRef> {
    if args.len() != 2 {
        return Err(datafusion::error::DataFusionError::Execution(
            "json_get_string requires exactly 2 arguments".to_string(),
        ));
    }

    let jsonb_array = args[0]
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "First argument must be LargeBinary".to_string(),
            )
        })?;

    let key_array = args[1]
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "Second argument must be String".to_string(),
            )
        })?;

    let mut builder = StringBuilder::new();

    for i in 0..jsonb_array.len() {
        if jsonb_array.is_null(i) || key_array.is_null(i) {
            builder.append_null();
        } else {
            let jsonb_bytes = jsonb_array.value(i);
            let key = key_array.value(i);

            match get_json_field_as_string(jsonb_bytes, key) {
                Ok(Some(value)) => builder.append_value(&value),
                Ok(None) => builder.append_null(),
                Err(e) => {
                    return Err(datafusion::error::DataFusionError::Execution(format!(
                        "Failed to get JSON string: {}",
                        e
                    )));
                }
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Get a field value as string with type coercion
fn get_json_field_as_string(jsonb_bytes: &[u8], key: &str) -> Result<Option<String>> {
    let raw_jsonb = jsonb::RawJsonb::new(jsonb_bytes);

    // Try as object field first
    let value = match raw_jsonb.get_by_name(key, false) {
        Ok(Some(value)) => value,
        Ok(None) => {
            // Try as array index
            if let Ok(index) = key.parse::<usize>() {
                match raw_jsonb.get_by_index(index) {
                    Ok(Some(value)) => value,
                    Ok(None) => return Ok(None),
                    Err(e) => {
                        return Err(datafusion::error::DataFusionError::Execution(format!(
                            "Failed to get array element: {}",
                            e
                        )))
                    }
                }
            } else {
                return Ok(None);
            }
        }
        Err(e) => {
            return Err(datafusion::error::DataFusionError::Execution(format!(
                "Failed to get field: {}",
                e
            )))
        }
    };

    // Convert to string and inspect
    let json_str = value.to_string();

    // Check for null
    if json_str == "null" {
        return Ok(None);
    }

    // Check if it's a string (starts and ends with quotes)
    if json_str.starts_with('"') && json_str.ends_with('"') {
        // Remove quotes
        Ok(Some(json_str[1..json_str.len() - 1].to_string()))
    } else if json_str == "true" || json_str == "false" {
        // Boolean
        Ok(Some(json_str))
    } else if json_str.starts_with('[') || json_str.starts_with('{') {
        // Array or object - cannot convert to string
        Err(datafusion::error::DataFusionError::Execution(
            "Cannot convert JSON object or array to string".to_string(),
        ))
    } else {
        // Number or other value
        Ok(Some(json_str))
    }
}

/// Create the json_get_int UDF for getting an integer value
pub fn json_get_int_udf() -> ScalarUDF {
    create_udf(
        "json_get_int",
        vec![DataType::LargeBinary, DataType::Utf8],
        DataType::Int64,
        Volatility::Immutable,
        Arc::new(json_get_int_columnar_impl),
    )
}

/// Implementation of json_get_int function with ColumnarValue
fn json_get_int_columnar_impl(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let arrays: Vec<ArrayRef> = args
        .iter()
        .map(|arg| match arg {
            ColumnarValue::Array(arr) => arr.clone(),
            ColumnarValue::Scalar(scalar) => scalar.to_array().unwrap(),
        })
        .collect();

    let result = json_get_int_impl(&arrays)?;
    Ok(ColumnarValue::Array(result))
}

/// Implementation of json_get_int function
fn json_get_int_impl(args: &[ArrayRef]) -> Result<ArrayRef> {
    if args.len() != 2 {
        return Err(datafusion::error::DataFusionError::Execution(
            "json_get_int requires exactly 2 arguments".to_string(),
        ));
    }

    let jsonb_array = args[0]
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "First argument must be LargeBinary".to_string(),
            )
        })?;

    let key_array = args[1]
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "Second argument must be String".to_string(),
            )
        })?;

    let mut builder = Int64Builder::new();

    for i in 0..jsonb_array.len() {
        if jsonb_array.is_null(i) || key_array.is_null(i) {
            builder.append_null();
        } else {
            let jsonb_bytes = jsonb_array.value(i);
            let key = key_array.value(i);

            match get_json_field_as_int(jsonb_bytes, key) {
                Ok(Some(value)) => builder.append_value(value),
                Ok(None) => builder.append_null(),
                Err(e) => {
                    return Err(datafusion::error::DataFusionError::Execution(format!(
                        "Failed to get JSON int: {}",
                        e
                    )));
                }
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Get a field value as integer with type coercion
fn get_json_field_as_int(jsonb_bytes: &[u8], key: &str) -> Result<Option<i64>> {
    let raw_jsonb = jsonb::RawJsonb::new(jsonb_bytes);

    // Try as object field first
    let value = match raw_jsonb.get_by_name(key, false) {
        Ok(Some(value)) => value,
        Ok(None) => {
            // Try as array index
            if let Ok(index) = key.parse::<usize>() {
                match raw_jsonb.get_by_index(index) {
                    Ok(Some(value)) => value,
                    Ok(None) => return Ok(None),
                    Err(e) => {
                        return Err(datafusion::error::DataFusionError::Execution(format!(
                            "Failed to get array element: {}",
                            e
                        )))
                    }
                }
            } else {
                return Ok(None);
            }
        }
        Err(e) => {
            return Err(datafusion::error::DataFusionError::Execution(format!(
                "Failed to get field: {}",
                e
            )))
        }
    };

    // Convert to string and parse
    let json_str = value.to_string();

    // Check for null
    if json_str == "null" {
        return Ok(None);
    }

    // Boolean conversion
    if json_str == "true" {
        return Ok(Some(1));
    } else if json_str == "false" {
        return Ok(Some(0));
    }

    // String value (remove quotes)
    let s = if json_str.starts_with('"') && json_str.ends_with('"') {
        &json_str[1..json_str.len() - 1]
    } else {
        &json_str
    };

    // Try to parse as integer
    if let Ok(n) = s.parse::<i64>() {
        Ok(Some(n))
    } else if let Ok(f) = s.parse::<f64>() {
        // Truncate float to int
        Ok(Some(f as i64))
    } else if s.starts_with('[') || s.starts_with('{') {
        Err(datafusion::error::DataFusionError::Execution(
            "Cannot convert JSON object or array to integer".to_string(),
        ))
    } else {
        Err(datafusion::error::DataFusionError::Execution(format!(
            "Cannot convert string '{}' to integer",
            s
        )))
    }
}

/// Create the json_get_float UDF for getting a float value
pub fn json_get_float_udf() -> ScalarUDF {
    create_udf(
        "json_get_float",
        vec![DataType::LargeBinary, DataType::Utf8],
        DataType::Float64,
        Volatility::Immutable,
        Arc::new(json_get_float_columnar_impl),
    )
}

/// Implementation of json_get_float function with ColumnarValue
fn json_get_float_columnar_impl(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let arrays: Vec<ArrayRef> = args
        .iter()
        .map(|arg| match arg {
            ColumnarValue::Array(arr) => arr.clone(),
            ColumnarValue::Scalar(scalar) => scalar.to_array().unwrap(),
        })
        .collect();

    let result = json_get_float_impl(&arrays)?;
    Ok(ColumnarValue::Array(result))
}

/// Implementation of json_get_float function
fn json_get_float_impl(args: &[ArrayRef]) -> Result<ArrayRef> {
    if args.len() != 2 {
        return Err(datafusion::error::DataFusionError::Execution(
            "json_get_float requires exactly 2 arguments".to_string(),
        ));
    }

    let jsonb_array = args[0]
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "First argument must be LargeBinary".to_string(),
            )
        })?;

    let key_array = args[1]
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "Second argument must be String".to_string(),
            )
        })?;

    let mut builder = Float64Builder::new();

    for i in 0..jsonb_array.len() {
        if jsonb_array.is_null(i) || key_array.is_null(i) {
            builder.append_null();
        } else {
            let jsonb_bytes = jsonb_array.value(i);
            let key = key_array.value(i);

            match get_json_field_as_float(jsonb_bytes, key) {
                Ok(Some(value)) => builder.append_value(value),
                Ok(None) => builder.append_null(),
                Err(e) => {
                    return Err(datafusion::error::DataFusionError::Execution(format!(
                        "Failed to get JSON float: {}",
                        e
                    )));
                }
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Get a field value as float with type coercion
fn get_json_field_as_float(jsonb_bytes: &[u8], key: &str) -> Result<Option<f64>> {
    let raw_jsonb = jsonb::RawJsonb::new(jsonb_bytes);

    // Try as object field first
    let value = match raw_jsonb.get_by_name(key, false) {
        Ok(Some(value)) => value,
        Ok(None) => {
            // Try as array index
            if let Ok(index) = key.parse::<usize>() {
                match raw_jsonb.get_by_index(index) {
                    Ok(Some(value)) => value,
                    Ok(None) => return Ok(None),
                    Err(e) => {
                        return Err(datafusion::error::DataFusionError::Execution(format!(
                            "Failed to get array element: {}",
                            e
                        )))
                    }
                }
            } else {
                return Ok(None);
            }
        }
        Err(e) => {
            return Err(datafusion::error::DataFusionError::Execution(format!(
                "Failed to get field: {}",
                e
            )))
        }
    };

    // Convert to string and parse
    let json_str = value.to_string();

    // Check for null
    if json_str == "null" {
        return Ok(None);
    }

    // Boolean conversion
    if json_str == "true" {
        return Ok(Some(1.0));
    } else if json_str == "false" {
        return Ok(Some(0.0));
    }

    // String value (remove quotes)
    let s = if json_str.starts_with('"') && json_str.ends_with('"') {
        &json_str[1..json_str.len() - 1]
    } else {
        &json_str
    };

    // Try to parse as float
    s.parse::<f64>().map(Some).map_err(|_| {
        if s.starts_with('[') || s.starts_with('{') {
            datafusion::error::DataFusionError::Execution(
                "Cannot convert JSON object or array to float".to_string(),
            )
        } else {
            datafusion::error::DataFusionError::Execution(format!(
                "Cannot convert string '{}' to float",
                s
            ))
        }
    })
}

/// Create the json_get_bool UDF for getting a boolean value
pub fn json_get_bool_udf() -> ScalarUDF {
    create_udf(
        "json_get_bool",
        vec![DataType::LargeBinary, DataType::Utf8],
        DataType::Boolean,
        Volatility::Immutable,
        Arc::new(json_get_bool_columnar_impl),
    )
}

/// Implementation of json_get_bool function with ColumnarValue
fn json_get_bool_columnar_impl(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let arrays: Vec<ArrayRef> = args
        .iter()
        .map(|arg| match arg {
            ColumnarValue::Array(arr) => arr.clone(),
            ColumnarValue::Scalar(scalar) => scalar.to_array().unwrap(),
        })
        .collect();

    let result = json_get_bool_impl(&arrays)?;
    Ok(ColumnarValue::Array(result))
}

/// Implementation of json_get_bool function
fn json_get_bool_impl(args: &[ArrayRef]) -> Result<ArrayRef> {
    if args.len() != 2 {
        return Err(datafusion::error::DataFusionError::Execution(
            "json_get_bool requires exactly 2 arguments".to_string(),
        ));
    }

    let jsonb_array = args[0]
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "First argument must be LargeBinary".to_string(),
            )
        })?;

    let key_array = args[1]
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "Second argument must be String".to_string(),
            )
        })?;

    let mut builder = BooleanBuilder::new();

    for i in 0..jsonb_array.len() {
        if jsonb_array.is_null(i) || key_array.is_null(i) {
            builder.append_null();
        } else {
            let jsonb_bytes = jsonb_array.value(i);
            let key = key_array.value(i);

            match get_json_field_as_bool(jsonb_bytes, key) {
                Ok(Some(value)) => builder.append_value(value),
                Ok(None) => builder.append_null(),
                Err(e) => {
                    return Err(datafusion::error::DataFusionError::Execution(format!(
                        "Failed to get JSON bool: {}",
                        e
                    )));
                }
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Get a field value as boolean with type coercion
fn get_json_field_as_bool(jsonb_bytes: &[u8], key: &str) -> Result<Option<bool>> {
    let raw_jsonb = jsonb::RawJsonb::new(jsonb_bytes);

    // Try as object field first
    let value = match raw_jsonb.get_by_name(key, false) {
        Ok(Some(value)) => value,
        Ok(None) => {
            // Try as array index
            if let Ok(index) = key.parse::<usize>() {
                match raw_jsonb.get_by_index(index) {
                    Ok(Some(value)) => value,
                    Ok(None) => return Ok(None),
                    Err(e) => {
                        return Err(datafusion::error::DataFusionError::Execution(format!(
                            "Failed to get array element: {}",
                            e
                        )))
                    }
                }
            } else {
                return Ok(None);
            }
        }
        Err(e) => {
            return Err(datafusion::error::DataFusionError::Execution(format!(
                "Failed to get field: {}",
                e
            )))
        }
    };

    // Convert to string and parse
    let json_str = value.to_string();

    // Check for null
    if json_str == "null" {
        return Ok(None);
    }

    // Direct boolean
    if json_str == "true" {
        return Ok(Some(true));
    } else if json_str == "false" {
        return Ok(Some(false));
    }

    // String value (remove quotes and check)
    let s = if json_str.starts_with('"') && json_str.ends_with('"') {
        json_str[1..json_str.len() - 1].to_lowercase()
    } else {
        json_str.to_lowercase()
    };

    // String to bool conversion
    match s.as_str() {
        "true" | "1" | "yes" | "y" | "on" => Ok(Some(true)),
        "false" | "0" | "no" | "n" | "off" => Ok(Some(false)),
        _ => {
            // Try as number
            if let Ok(n) = s.parse::<i64>() {
                Ok(Some(n != 0))
            } else if let Ok(f) = s.parse::<f64>() {
                Ok(Some(f != 0.0))
            } else if s.starts_with('[') || s.starts_with('{') {
                Err(datafusion::error::DataFusionError::Execution(
                    "Cannot convert JSON object or array to boolean".to_string(),
                ))
            } else {
                Err(datafusion::error::DataFusionError::Execution(format!(
                    "Cannot convert string '{}' to boolean",
                    s
                )))
            }
        }
    }
}

/// Create the json_array_contains UDF for checking if array contains a value
pub fn json_array_contains_udf() -> ScalarUDF {
    create_udf(
        "json_array_contains",
        vec![DataType::LargeBinary, DataType::Utf8, DataType::Utf8],
        DataType::Boolean,
        Volatility::Immutable,
        Arc::new(json_array_contains_columnar_impl),
    )
}

/// Implementation of json_array_contains function with ColumnarValue
fn json_array_contains_columnar_impl(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let arrays: Vec<ArrayRef> = args
        .iter()
        .map(|arg| match arg {
            ColumnarValue::Array(arr) => arr.clone(),
            ColumnarValue::Scalar(scalar) => scalar.to_array().unwrap(),
        })
        .collect();

    let result = json_array_contains_impl(&arrays)?;
    Ok(ColumnarValue::Array(result))
}

/// Implementation of json_array_contains function
fn json_array_contains_impl(args: &[ArrayRef]) -> Result<ArrayRef> {
    if args.len() != 3 {
        return Err(datafusion::error::DataFusionError::Execution(
            "json_array_contains requires exactly 3 arguments".to_string(),
        ));
    }

    let jsonb_array = args[0]
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "First argument must be LargeBinary".to_string(),
            )
        })?;

    let path_array = args[1]
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "Second argument must be String".to_string(),
            )
        })?;

    let value_array = args[2]
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "Third argument must be String".to_string(),
            )
        })?;

    let mut builder = BooleanBuilder::new();

    for i in 0..jsonb_array.len() {
        if jsonb_array.is_null(i) || path_array.is_null(i) || value_array.is_null(i) {
            builder.append_null();
        } else {
            let jsonb_bytes = jsonb_array.value(i);
            let path = path_array.value(i);
            let value = value_array.value(i);

            match check_array_contains(jsonb_bytes, path, value) {
                Ok(contains) => builder.append_value(contains),
                Err(e) => {
                    return Err(datafusion::error::DataFusionError::Execution(format!(
                        "Failed to check array contains: {}",
                        e
                    )));
                }
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Check if a JSON array at path contains a value
fn check_array_contains(jsonb_bytes: &[u8], path: &str, value: &str) -> Result<bool> {
    let json_path = jsonb::jsonpath::parse_json_path(path.as_bytes()).map_err(|e| {
        datafusion::error::DataFusionError::Execution(format!("Invalid JSONPath: {}", e))
    })?;

    let raw_jsonb = jsonb::RawJsonb::new(jsonb_bytes);
    let mut selector = jsonb::jsonpath::Selector::new(raw_jsonb);
    match selector.select_values(&json_path) {
        Ok(values) => {
            for v in values {
                // Convert to raw JSONB for direct access
                let raw = v.as_raw();
                // Check if it's an array by trying to iterate
                let mut index = 0;
                loop {
                    match raw.get_by_index(index) {
                        Ok(Some(elem)) => {
                            let elem_str = elem.to_string();
                            // Compare as JSON strings (with quotes for strings)
                            if elem_str == value || elem_str == format!("\"{}\"", value) {
                                return Ok(true);
                            }
                            index += 1;
                        }
                        Ok(None) => break, // End of array
                        Err(_) => break,   // Not an array or error
                    }
                }
            }
            Ok(false)
        }
        Err(_) => Ok(false),
    }
}

/// Create the json_array_length UDF for getting array length
pub fn json_array_length_udf() -> ScalarUDF {
    create_udf(
        "json_array_length",
        vec![DataType::LargeBinary, DataType::Utf8],
        DataType::Int64,
        Volatility::Immutable,
        Arc::new(json_array_length_columnar_impl),
    )
}

/// Implementation of json_array_length function with ColumnarValue
fn json_array_length_columnar_impl(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let arrays: Vec<ArrayRef> = args
        .iter()
        .map(|arg| match arg {
            ColumnarValue::Array(arr) => arr.clone(),
            ColumnarValue::Scalar(scalar) => scalar.to_array().unwrap(),
        })
        .collect();

    let result = json_array_length_impl(&arrays)?;
    Ok(ColumnarValue::Array(result))
}

/// Implementation of json_array_length function
fn json_array_length_impl(args: &[ArrayRef]) -> Result<ArrayRef> {
    if args.len() != 2 {
        return Err(datafusion::error::DataFusionError::Execution(
            "json_array_length requires exactly 2 arguments".to_string(),
        ));
    }

    let jsonb_array = args[0]
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "First argument must be LargeBinary".to_string(),
            )
        })?;

    let path_array = args[1]
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "Second argument must be String".to_string(),
            )
        })?;

    let mut builder = Int64Builder::new();

    for i in 0..jsonb_array.len() {
        if jsonb_array.is_null(i) || path_array.is_null(i) {
            builder.append_null();
        } else {
            let jsonb_bytes = jsonb_array.value(i);
            let path = path_array.value(i);

            match get_array_length(jsonb_bytes, path) {
                Ok(Some(len)) => builder.append_value(len),
                Ok(None) => builder.append_null(),
                Err(e) => {
                    return Err(datafusion::error::DataFusionError::Execution(format!(
                        "Failed to get array length: {}",
                        e
                    )));
                }
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Get the length of a JSON array at path
fn get_array_length(jsonb_bytes: &[u8], path: &str) -> Result<Option<i64>> {
    let json_path = jsonb::jsonpath::parse_json_path(path.as_bytes()).map_err(|e| {
        datafusion::error::DataFusionError::Execution(format!("Invalid JSONPath: {}", e))
    })?;

    let raw_jsonb = jsonb::RawJsonb::new(jsonb_bytes);
    let mut selector = jsonb::jsonpath::Selector::new(raw_jsonb);
    match selector.select_values(&json_path) {
        Ok(values) => {
            if values.is_empty() {
                return Ok(None);
            }
            let first = &values[0];
            let raw = first.as_raw();

            // Count array elements by iterating
            let mut count = 0;
            loop {
                match raw.get_by_index(count) {
                    Ok(Some(_)) => count += 1,
                    Ok(None) => break, // End of array
                    Err(_) => {
                        // Not an array
                        if count == 0 {
                            return Err(datafusion::error::DataFusionError::Execution(format!(
                                "Path does not point to an array: {}",
                                path
                            )));
                        }
                        break;
                    }
                }
            }
            Ok(Some(count as i64))
        }
        Err(_) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::builder::LargeBinaryBuilder;
    use arrow_array::{BooleanArray, Int64Array};

    fn create_test_jsonb(json_str: &str) -> Vec<u8> {
        jsonb::parse_value(json_str.as_bytes()).unwrap().to_vec()
    }

    #[tokio::test]
    async fn test_json_extract_udf() -> Result<()> {
        let json = r#"{"user": {"name": "Alice", "age": 30}}"#;
        let jsonb_bytes = create_test_jsonb(json);

        let mut binary_builder = LargeBinaryBuilder::new();
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_null();

        let jsonb_array = Arc::new(binary_builder.finish());
        let path_array = Arc::new(StringArray::from(vec![
            Some("$.user.name"),
            Some("$.user.age"),
            Some("$.user.name"),
        ]));

        let result = json_extract_impl(&[jsonb_array, path_array])?;
        let string_array = result.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(string_array.len(), 3);
        assert_eq!(string_array.value(0), "\"Alice\"");
        assert_eq!(string_array.value(1), "30");
        assert!(string_array.is_null(2));

        Ok(())
    }

    #[tokio::test]
    async fn test_json_exists_udf() -> Result<()> {
        let json = r#"{"user": {"name": "Alice", "age": 30}, "tags": ["rust", "json"]}"#;
        let jsonb_bytes = create_test_jsonb(json);

        let mut binary_builder = LargeBinaryBuilder::new();
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_null();

        let jsonb_array = Arc::new(binary_builder.finish());
        let path_array = Arc::new(StringArray::from(vec![
            Some("$.user.name"),
            Some("$.user.email"),
            Some("$.tags"),
            Some("$.any"),
        ]));

        let result = json_exists_impl(&[jsonb_array, path_array])?;
        let bool_array = result.as_any().downcast_ref::<BooleanArray>().unwrap();

        assert_eq!(bool_array.len(), 4);
        assert!(bool_array.value(0));
        assert!(!bool_array.value(1));
        assert!(bool_array.value(2));
        assert!(bool_array.is_null(3));

        Ok(())
    }

    #[tokio::test]
    async fn test_json_get_udf() -> Result<()> {
        let json = r#"{"name": "Alice", "nested": {"value": 42}, "arr": [1, 2, 3]}"#;
        let jsonb_bytes = create_test_jsonb(json);

        let mut binary_builder = LargeBinaryBuilder::new();
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_null();

        let jsonb_array = Arc::new(binary_builder.finish());
        let key_array = Arc::new(StringArray::from(vec![
            Some("name"),
            Some("nested"),
            Some("not_exists"),
            Some("any"),
        ]));

        let result = json_get_impl(&[jsonb_array, key_array])?;
        let binary_array = result.as_any().downcast_ref::<LargeBinaryArray>().unwrap();

        assert_eq!(binary_array.len(), 4);
        assert!(!binary_array.is_null(0));
        assert!(!binary_array.is_null(1));
        assert!(binary_array.is_null(2));
        assert!(binary_array.is_null(3));

        // Verify returned values are valid JSONB
        let value0 = jsonb::RawJsonb::new(binary_array.value(0));
        assert_eq!(value0.to_string(), "\"Alice\"");

        let value1 = jsonb::RawJsonb::new(binary_array.value(1));
        assert!(value1.to_string().contains("\"value\":42"));

        Ok(())
    }

    #[tokio::test]
    async fn test_json_get_string_udf() -> Result<()> {
        // Test valid string conversions
        let json = r#"{"str": "hello", "num": 123, "bool": true, "null": null}"#;
        let jsonb_bytes = create_test_jsonb(json);

        let mut binary_builder = LargeBinaryBuilder::new();
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);

        let jsonb_array = Arc::new(binary_builder.finish());
        let key_array = Arc::new(StringArray::from(vec![
            Some("str"),
            Some("num"),
            Some("bool"),
            Some("null"),
        ]));

        let result = json_get_string_impl(&[jsonb_array, key_array])?;
        let string_array = result.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(string_array.len(), 4);
        assert_eq!(string_array.value(0), "hello");
        assert_eq!(string_array.value(1), "123");
        assert_eq!(string_array.value(2), "true");
        assert!(string_array.is_null(3));

        Ok(())
    }

    #[tokio::test]
    async fn test_json_get_string_error_on_object() -> Result<()> {
        // Test that objects cannot be converted to string
        let json = r#"{"obj": {}}"#;
        let jsonb_bytes = create_test_jsonb(json);

        let mut binary_builder = LargeBinaryBuilder::new();
        binary_builder.append_value(&jsonb_bytes);

        let jsonb_array = Arc::new(binary_builder.finish());
        let key_array = Arc::new(StringArray::from(vec![Some("obj")]));

        let result = json_get_string_impl(&[jsonb_array, key_array]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Cannot convert"));

        Ok(())
    }

    #[tokio::test]
    async fn test_json_get_int_udf() -> Result<()> {
        let json = r#"{"int": 42, "float": 3.14, "str_num": "99", "bool": true, "str": "abc"}"#;
        let jsonb_bytes = create_test_jsonb(json);

        let mut binary_builder = LargeBinaryBuilder::new();
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);

        let jsonb_array = Arc::new(binary_builder.finish());
        let key_array = Arc::new(StringArray::from(vec![
            Some("int"),
            Some("float"),
            Some("str_num"),
            Some("bool"),
        ]));

        let result = json_get_int_impl(&[jsonb_array, key_array])?;
        let int_array = result.as_any().downcast_ref::<Int64Array>().unwrap();

        assert_eq!(int_array.len(), 4);
        assert_eq!(int_array.value(0), 42);
        assert_eq!(int_array.value(1), 3); // Truncated
        assert_eq!(int_array.value(2), 99);
        assert_eq!(int_array.value(3), 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_json_get_int_error() -> Result<()> {
        let json = r#"{"str": "not_a_number"}"#;
        let jsonb_bytes = create_test_jsonb(json);

        let mut binary_builder = LargeBinaryBuilder::new();
        binary_builder.append_value(&jsonb_bytes);

        let jsonb_array = Arc::new(binary_builder.finish());
        let key_array = Arc::new(StringArray::from(vec![Some("str")]));

        let result = json_get_int_impl(&[jsonb_array, key_array]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Cannot convert string"));

        Ok(())
    }

    #[tokio::test]
    async fn test_json_get_bool_udf() -> Result<()> {
        let json =
            r#"{"bool_true": true, "bool_false": false, "num": 1, "zero": 0, "str": "true"}"#;
        let jsonb_bytes = create_test_jsonb(json);

        let mut binary_builder = LargeBinaryBuilder::new();
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);

        let jsonb_array = Arc::new(binary_builder.finish());
        let key_array = Arc::new(StringArray::from(vec![
            Some("bool_true"),
            Some("bool_false"),
            Some("num"),
            Some("zero"),
            Some("str"),
        ]));

        let result = json_get_bool_impl(&[jsonb_array, key_array])?;
        let bool_array = result.as_any().downcast_ref::<BooleanArray>().unwrap();

        assert_eq!(bool_array.len(), 5);
        assert!(bool_array.value(0));
        assert!(!bool_array.value(1));
        assert!(bool_array.value(2));
        assert!(!bool_array.value(3));
        assert!(bool_array.value(4));

        Ok(())
    }

    #[tokio::test]
    async fn test_json_array_contains_udf() -> Result<()> {
        let json = r#"{"tags": ["rust", "json", "database"], "nums": [1, 2, 3]}"#;
        let jsonb_bytes = create_test_jsonb(json);

        let mut binary_builder = LargeBinaryBuilder::new();
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_null();

        let jsonb_array = Arc::new(binary_builder.finish());
        let path_array = Arc::new(StringArray::from(vec![
            Some("$.tags"),
            Some("$.tags"),
            Some("$.nums"),
            Some("$.tags"),
        ]));
        let value_array = Arc::new(StringArray::from(vec![
            Some("rust"),
            Some("python"),
            Some("2"),
            Some("any"),
        ]));

        let result = json_array_contains_impl(&[jsonb_array, path_array, value_array])?;
        let bool_array = result.as_any().downcast_ref::<BooleanArray>().unwrap();

        assert_eq!(bool_array.len(), 4);
        assert!(bool_array.value(0));
        assert!(!bool_array.value(1));
        assert!(bool_array.value(2));
        assert!(bool_array.is_null(3));

        Ok(())
    }

    #[tokio::test]
    async fn test_json_array_length_udf() -> Result<()> {
        let json = r#"{"empty": [], "tags": ["a", "b", "c"], "nested": {"arr": [1, 2]}}"#;
        let jsonb_bytes = create_test_jsonb(json);

        let mut binary_builder = LargeBinaryBuilder::new();
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_null();

        let jsonb_array = Arc::new(binary_builder.finish());
        let path_array = Arc::new(StringArray::from(vec![
            Some("$.empty"),
            Some("$.tags"),
            Some("$.nested.arr"),
            Some("$.any"),
        ]));

        let result = json_array_length_impl(&[jsonb_array, path_array])?;
        let int_array = result.as_any().downcast_ref::<Int64Array>().unwrap();

        assert_eq!(int_array.len(), 4);
        assert_eq!(int_array.value(0), 0);
        assert_eq!(int_array.value(1), 3);
        assert_eq!(int_array.value(2), 2);
        assert!(int_array.is_null(3));

        Ok(())
    }

    #[tokio::test]
    async fn test_json_array_access() -> Result<()> {
        let json = r#"["first", "second", "third"]"#;
        let jsonb_bytes = create_test_jsonb(json);

        let mut binary_builder = LargeBinaryBuilder::new();
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);

        let jsonb_array = Arc::new(binary_builder.finish());
        let key_array = Arc::new(StringArray::from(vec![
            Some("0"),
            Some("1"),
            Some("10"), // Out of bounds
        ]));

        let result = json_get_string_impl(&[jsonb_array, key_array])?;
        let string_array = result.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(string_array.len(), 3);
        assert_eq!(string_array.value(0), "first");
        assert_eq!(string_array.value(1), "second");
        assert!(string_array.is_null(2));

        Ok(())
    }
}
