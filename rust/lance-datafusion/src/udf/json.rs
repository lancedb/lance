// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::builder::StringBuilder;
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
    match raw_jsonb.select_by_path(&json_path) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::builder::LargeBinaryBuilder;

    #[tokio::test]
    async fn test_variant_extract_udf() -> Result<()> {
        // Create test JSONB data
        let json = r#"{"user": {"name": "Alice", "age": 30}}"#;
        let jsonb_bytes = jsonb::parse_value(json.as_bytes()).unwrap().to_vec();

        // Create arrays
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

        // Call UDF
        let result = json_extract_impl(&[jsonb_array, path_array])?;
        let string_array = result.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(string_array.len(), 3);
        assert_eq!(string_array.value(0), "\"Alice\"");
        assert_eq!(string_array.value(1), "30");
        assert!(string_array.is_null(2));

        Ok(())
    }
}
