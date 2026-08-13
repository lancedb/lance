// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Datafusion user defined functions

use arrow_array::{Array, ArrayRef, BooleanArray, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion::common::Result as DFResult;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    Volatility, create_udf,
};
use datafusion::prelude::SessionContext;
use datafusion::scalar::ScalarValue;
use datafusion_functions::utils::make_scalar_function;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};

pub mod json;

/// Register UDF functions to datafusion context.
pub fn register_functions(ctx: &SessionContext) {
    ctx.register_udf(CONTAINS_TOKENS_UDF.clone());
    ctx.register_udf(IS_ASSIGNED_UDF.clone());
    // JSON functions
    ctx.register_udf(json::json_extract_udf());
    ctx.register_udf(json::json_extract_with_type_udf());
    ctx.register_udf(json::json_exists_udf());
    ctx.register_udf(json::json_get_udf());
    ctx.register_udf(json::json_get_string_udf());
    ctx.register_udf(json::json_get_int_udf());
    ctx.register_udf(json::json_get_float_udf());
    ctx.register_udf(json::json_get_bool_udf());
    ctx.register_udf(json::json_array_contains_udf());
    ctx.register_udf(json::json_array_length_udf());
    // GEO functions
    #[cfg(feature = "geo")]
    lance_geo::register_functions(ctx);
    #[cfg(not(feature = "geo"))]
    register_geo_stub_functions(ctx);
}

/// Stable logical name of Lance's snapshot-bound assignment expression.
///
/// ```
/// use lance_datafusion::udf::IS_ASSIGNED_NAME;
/// assert_eq!(IS_ASSIGNED_NAME, "is_assigned");
/// ```
pub const IS_ASSIGNED_NAME: &str = "is_assigned";

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct IsAssignedUdf {
    signature: Signature,
}

impl Default for IsAssignedUdf {
    fn default() -> Self {
        Self {
            // Snapshot state is stable for one query but is not a pure function
            // of the Arrow value supplied as the logical field argument.
            signature: Signature::any(1, Volatility::Stable),
        }
    }
}

impl ScalarUDFImpl for IsAssignedUdf {
    fn name(&self) -> &str {
        IS_ASSIGNED_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::Boolean)
    }

    fn return_field_from_args(&self, _args: ReturnFieldArgs) -> DFResult<FieldRef> {
        Ok(Arc::new(Field::new(
            IS_ASSIGNED_NAME,
            DataType::Boolean,
            false,
        )))
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        Err(datafusion::error::DataFusionError::Plan(
            "is_assigned(field) must be bound to a Lance dataset snapshot before execution"
                .to_string(),
        ))
    }
}

/// Assignment membership for one fragment in a bound dataset snapshot.
///
/// ```
/// use lance_datafusion::udf::AssignmentFragment;
/// let state = AssignmentFragment::All;
/// assert!(matches!(state, AssignmentFragment::All));
/// ```
#[derive(Debug, Clone)]
pub enum AssignmentFragment {
    /// Every physical row in the fragment is assigned.
    All,
    /// Only the listed physical row offsets are assigned.
    Partial(Arc<RoaringBitmap>),
}

#[derive(Debug, Clone)]
struct BoundIsAssignedUdf {
    binding_id: u64,
    field_id: i32,
    fragments: Arc<HashMap<u32, AssignmentFragment>>,
    signature: Signature,
}

impl PartialEq for BoundIsAssignedUdf {
    fn eq(&self, other: &Self) -> bool {
        self.binding_id == other.binding_id && self.field_id == other.field_id
    }
}

impl Eq for BoundIsAssignedUdf {}

impl Hash for BoundIsAssignedUdf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.binding_id.hash(state);
        self.field_id.hash(state);
    }
}

impl BoundIsAssignedUdf {
    fn value(&self, row_address: u64) -> bool {
        let fragment_id = (row_address >> 32) as u32;
        let row_offset = row_address as u32;
        match self.fragments.get(&fragment_id) {
            Some(AssignmentFragment::All) => true,
            Some(AssignmentFragment::Partial(bitmap)) => bitmap.contains(row_offset),
            None => false,
        }
    }
}

impl ScalarUDFImpl for BoundIsAssignedUdf {
    fn name(&self) -> &str {
        IS_ASSIGNED_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::Boolean)
    }

    fn return_field_from_args(&self, _args: ReturnFieldArgs) -> DFResult<FieldRef> {
        Ok(Arc::new(Field::new(
            IS_ASSIGNED_NAME,
            DataType::Boolean,
            false,
        )))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        match &args.args[0] {
            ColumnarValue::Array(array) => {
                let row_addresses =
                    array
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .ok_or_else(|| {
                            datafusion::error::DataFusionError::Execution(format!(
                                "bound is_assigned expected UInt64 row addresses, got {}",
                                array.data_type()
                            ))
                        })?;
                let values = (0..row_addresses.len()).map(|index| {
                    if row_addresses.is_valid(index) {
                        self.value(row_addresses.value(index))
                    } else {
                        false
                    }
                });
                Ok(ColumnarValue::Array(Arc::new(BooleanArray::from_iter(
                    values.map(Some),
                ))))
            }
            ColumnarValue::Scalar(ScalarValue::UInt64(row_address)) => Ok(ColumnarValue::Scalar(
                ScalarValue::Boolean(Some(row_address.is_some_and(|value| self.value(value)))),
            )),
            value => Err(datafusion::error::DataFusionError::Execution(format!(
                "bound is_assigned expected UInt64 row addresses, got {}",
                value.data_type()
            ))),
        }
    }
}

static NEXT_ASSIGNMENT_BINDING_ID: AtomicU64 = AtomicU64::new(1);

/// Create a snapshot-bound `is_assigned` UDF whose argument is a physical row
/// address. The scanner replaces the user's field argument with `_rowaddr` only
/// after resolving the field to a stable ID and loading the referenced state.
///
/// ```
/// use std::collections::HashMap;
/// use lance_datafusion::udf::{AssignmentFragment, bound_is_assigned_udf};
///
/// let udf = bound_is_assigned_udf(7, HashMap::from([(3, AssignmentFragment::All)]));
/// assert_eq!(udf.name(), "is_assigned");
/// ```
pub fn bound_is_assigned_udf(
    field_id: i32,
    fragments: HashMap<u32, AssignmentFragment>,
) -> ScalarUDF {
    ScalarUDF::new_from_impl(BoundIsAssignedUdf {
        binding_id: NEXT_ASSIGNMENT_BINDING_ID.fetch_add(1, Ordering::Relaxed),
        field_id,
        fragments: Arc::new(fragments),
        signature: Signature::exact(vec![DataType::UInt64], Volatility::Immutable),
    })
}

/// Build the native DataFusion logical expression for
/// `is_assigned(field)`. A Lance scanner resolves the field reference to a
/// stable field ID and binds snapshot state before physical planning.
///
/// ```
/// use datafusion::logical_expr::col;
/// use lance_datafusion::udf::is_assigned;
///
/// let expression = is_assigned(col("embedding"));
/// assert!(expression.to_string().starts_with("is_assigned("));
/// ```
pub fn is_assigned(field: datafusion::logical_expr::Expr) -> datafusion::logical_expr::Expr {
    datafusion::logical_expr::Expr::ScalarFunction(
        datafusion::logical_expr::expr::ScalarFunction::new_udf(
            Arc::new(IS_ASSIGNED_UDF.clone()),
            vec![field],
        ),
    )
}

/// When the `geo` feature is disabled, register stub UDFs for spatial SQL functions
/// so that users get a clear error mentioning the feature flag instead of
/// DataFusion's generic "Unknown function" error.
#[cfg(not(feature = "geo"))]
fn register_geo_stub_functions(ctx: &SessionContext) {
    let geo_funcs = [
        "st_intersects",
        "st_contains",
        "st_within",
        "st_touches",
        "st_crosses",
        "st_overlaps",
        "st_covers",
        "st_coveredby",
        "st_distance",
        "st_area",
        "st_length",
    ];

    for name in geo_funcs {
        let func_name = name.to_string();
        let stub = Arc::new(make_scalar_function(
            move |_args: &[ArrayRef]| {
                Err(datafusion::error::DataFusionError::Plan(format!(
                    "Function '{}' requires the `geo` feature. \
                     Rebuild with `--features geo` to enable geospatial functions.",
                    func_name
                )))
            },
            vec![],
        ));

        ctx.register_udf(create_udf(
            name,
            vec![DataType::Binary, DataType::Binary],
            DataType::Boolean,
            Volatility::Immutable,
            stub,
        ));
    }
}

/// This method checks whether a string contains all specified tokens. The tokens are separated by
/// punctuations and white spaces.
///
/// The functionality is equivalent to FTS MatchQuery (with fuzziness disabled, Operator::And,
/// and using the simple tokenizer). If FTS index exists and suites the query, it will be used to
/// optimize the query.
///
/// Usage
/// * Use `contains_tokens` in sql.
/// ```rust,ignore
/// let sql = "SELECT * FROM table WHERE contains_tokens(text_col, 'fox jumps dog')";
/// let mut ds = Dataset::open(&ds_path).await?;
/// let ctx = SessionContext::new();
/// ctx.register_table(
///     "table",
///     Arc::new(LanceTableProvider::new(dataset, false, false)),
/// )?;
/// register_functions(&ctx);
/// let df = ctx.sql(sql).await?;
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

            let tokens: Option<Vec<&str>> = match scalar_str.len() {
                0 => None,
                _ => Some(collect_tokens(scalar_str.value(0))),
            };

            let result = column.iter().map(|text| {
                text.map(|text| {
                    let text_tokens = collect_tokens(text);
                    if let Some(tokens) = &tokens {
                        tokens.len()
                            == tokens
                                .iter()
                                .filter(|token| text_tokens.contains(*token))
                                .count()
                    } else {
                        true
                    }
                })
            });

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

/// Split tokens separated by punctuations and white spaces.
fn collect_tokens(text: &str) -> Vec<&str> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect()
}

pub static CONTAINS_TOKENS_UDF: LazyLock<ScalarUDF> = LazyLock::new(contains_tokens);
/// Unbound logical `is_assigned(field)` expression registered with DataFusion
/// and Substrait. Lance scanners replace it with a snapshot-bound instance.
///
/// ```
/// use lance_datafusion::udf::IS_ASSIGNED_UDF;
/// assert_eq!(IS_ASSIGNED_UDF.name(), "is_assigned");
/// ```
pub static IS_ASSIGNED_UDF: LazyLock<ScalarUDF> =
    LazyLock::new(|| ScalarUDF::new_from_impl(IsAssignedUdf::default()));

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
            "a cat catch a fish",
            "a fish catch a cat",
            "a white cat catch a big fish",
            "cat catchup fish",
            "cat fish catch",
        ]));
        let token = Arc::new(StringArray::from(vec![
            " cat catch fish.",
            " cat catch fish.",
            " cat catch fish.",
            " cat catch fish.",
            " cat catch fish.",
        ]));

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
            config_options: Arc::new(Default::default()),
        };

        // Invoke contains_tokens manually
        let values = contains_tokens.invoke_with_args(args).unwrap();

        if let ColumnarValue::Array(array) = values {
            let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            assert_eq!(
                array.clone(),
                BooleanArray::from(vec![true, true, true, false, true])
            );
        } else {
            panic!("Expected an Array but got {:?}", values);
        }
    }
}
