// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Datafusion user defined functions

use arrow_array::{ArrayRef, BooleanArray, StringArray};
use arrow_schema::DataType;
use datafusion::logical_expr::{create_udf, ScalarUDF, Volatility};
use datafusion::prelude::SessionContext;
use datafusion_functions::utils::make_scalar_function;
use std::sync::{Arc, LazyLock};

/// Register UDF functions to datafusion context.
pub fn register_functions(ctx: &SessionContext) {
    ctx.register_udf(CONTAINS_TOKENS_UDF.clone());
}

/// This method checks whether a string contains another string. It utilizes FTS (Full-Text Search)
/// indexes, but due to the false negative characteristic of FTS, the results may have omissions.
/// For example, "bakin" will not match documents containing "baking."
/// If the query string is a whole word, or if you prioritize better performance, `contains_tokens`
/// is the better choice. Otherwise, you can use the `contains` method to obtain accurate results.
///
///
/// Usage
/// * Use `contains_tokens` in sql.
/// ```rust,ignore
/// let sql = "SELECT * FROM table WHERE contains_tokens(text_col, 'bakin')"
/// let mut ds = Dataset::open(&ds_path).await?;
/// let mut builder = ds.sql(&sql);
/// let records = builder.clone().build().await?.into_batch_records().await?;
/// ```
fn contains_tokens() -> ScalarUDF {
    let function = Arc::new(make_scalar_function(
        |args: &[ArrayRef]| {
            let column = args[0].as_any().downcast_ref::<StringArray>().ok_or(
                datafusion::error::DataFusionError::Execution(
                    "First argument of contains_tokens can't be cast to string".to_string(),
                ),
            )?;
            let scalar_str = args[1].as_any().downcast_ref::<StringArray>().ok_or(
                datafusion::error::DataFusionError::Execution(
                    "Second argument of contains_tokens can't be cast to string".to_string(),
                ),
            )?;

            let result = column
                .iter()
                .enumerate()
                .map(|(i, column)| column.map(|value| value.contains(scalar_str.value(i))));

            Ok(Arc::new(BooleanArray::from_iter(result)) as ArrayRef)
        },
        vec![],
    ));

    create_udf(
        "contains_tokens",
        vec![DataType::Utf8, DataType::Utf8],
        DataType::Boolean,
        Volatility::Immutable,
        function,
    )
}

static CONTAINS_TOKENS_UDF: LazyLock<ScalarUDF> = LazyLock::new(contains_tokens);

#[cfg(test)]
mod tests {
    use crate::udf::CONTAINS_TOKENS_UDF;
    use arrow_array::{Array, BooleanArray, StringArray};
    use arrow_schema::{DataType, Field};
    use datafusion::logical_expr::ScalarFunctionArgs;
    use datafusion::physical_plan::ColumnarValue;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_contains_tokens() {
        // Prepare arguments
        let contains_tokens = CONTAINS_TOKENS_UDF.clone();
        let text_col = Arc::new(StringArray::from(vec![
            "a cat",
            "lovely cat",
            "white cat",
            "catch up",
            "fish",
        ]));
        let token = Arc::new(StringArray::from(vec!["cat", "cat", "cat", "cat", "cat"]));

        let args = vec![ColumnarValue::Array(text_col), ColumnarValue::Array(token)];
        let arg_fields = vec![
            Arc::new(Field::new("text_col".to_string(), DataType::Utf8, false)),
            Arc::new(Field::new("token".to_string(), DataType::Utf8, false)),
        ];

        let args = ScalarFunctionArgs {
            args,
            arg_fields,
            number_rows: 5,
            return_field: Arc::new(Field::new("res".to_string(), DataType::Boolean, false)),
        };

        // Invoke contains_tokens manually
        let values = contains_tokens.invoke_with_args(args).unwrap();

        if let ColumnarValue::Array(array) = values {
            let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            assert_eq!(
                array.clone(),
                BooleanArray::from(vec![true, true, true, true, false])
            );
        } else {
            panic!("Expected an Array but got {:?}", values);
        }
    }
}
