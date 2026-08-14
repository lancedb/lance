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
use std::sync::{Arc, LazyLock, OnceLock};

pub mod json;

/// Register UDF functions to datafusion context.
pub fn register_functions(ctx: &SessionContext) {
    ctx.register_udf(CONTAINS_TOKENS_UDF.clone());
    ctx.register_udf(CELL_FLAG_UDF.clone());
    ctx.register_udf(CELL_FLAG_ID_UDF.clone());
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

/// Stable logical name of Lance's snapshot-bound cell flag expression.
///
/// ```
/// use lance_datafusion::udf::CELL_FLAG_NAME;
/// assert_eq!(CELL_FLAG_NAME, "cell_flag");
/// ```
pub const CELL_FLAG_NAME: &str = "cell_flag";

/// Internal Substrait transport name for a snapshot-resolved stable flag ID.
#[doc(hidden)]
pub const CELL_FLAG_ID_NAME: &str = "__lance_cell_flag_id";

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct CellFlagUdf {
    signature: Signature,
}

impl Default for CellFlagUdf {
    fn default() -> Self {
        Self {
            // Snapshot state is stable for one query but is not a pure function
            // of the Arrow value and flag name supplied as logical arguments.
            signature: Signature::any(2, Volatility::Stable),
        }
    }
}

impl ScalarUDFImpl for CellFlagUdf {
    fn name(&self) -> &str {
        CELL_FLAG_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::Boolean)
    }

    fn return_field_from_args(&self, _args: ReturnFieldArgs) -> DFResult<FieldRef> {
        Ok(Arc::new(Field::new(
            CELL_FLAG_NAME,
            DataType::Boolean,
            false,
        )))
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        Err(datafusion::error::DataFusionError::Plan(
            "cell_flag(field, name) must be bound to a Lance dataset snapshot before execution"
                .to_string(),
        ))
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct CellFlagIdUdf {
    signature: Signature,
}

impl Default for CellFlagIdUdf {
    fn default() -> Self {
        Self {
            signature: Signature::exact(vec![DataType::UInt32], Volatility::Stable),
        }
    }
}

impl ScalarUDFImpl for CellFlagIdUdf {
    fn name(&self) -> &str {
        CELL_FLAG_ID_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::Boolean)
    }

    fn return_field_from_args(&self, _args: ReturnFieldArgs) -> DFResult<FieldRef> {
        Ok(Arc::new(Field::new(
            CELL_FLAG_ID_NAME,
            DataType::Boolean,
            false,
        )))
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        Err(datafusion::error::DataFusionError::Plan(
            "Internal cell flag ID transport must be bound to a Lance dataset snapshot before execution"
                .to_string(),
        ))
    }
}

/// True membership for one flag in one fragment of a bound dataset snapshot.
///
/// ```
/// use lance_datafusion::udf::FlagFragment;
/// let state = FlagFragment::All;
/// assert!(matches!(state, FlagFragment::All));
/// ```
#[derive(Debug, Clone)]
pub enum FlagFragment {
    /// Every physical row in the fragment is true.
    All,
    /// Only the listed physical row offsets are true.
    Partial(Arc<RoaringBitmap>),
}

/// Snapshot state and complete fragment domain carried by a bound cell-flag UDF.
#[doc(hidden)]
pub type BoundCellFlagSnapshot = (u32, Arc<HashMap<u32, FlagFragment>>, RoaringBitmap);

#[derive(Debug, Clone)]
struct BoundCellFlagUdf {
    binding_id: u64,
    flag_id: u32,
    snapshot: Arc<CellFlagSnapshot>,
    signature: Signature,
}

#[derive(Debug)]
struct CellFlagSnapshot {
    fragments: Arc<HashMap<u32, FlagFragment>>,
    covered_fragments: RoaringBitmap,
}

impl PartialEq for BoundCellFlagUdf {
    fn eq(&self, other: &Self) -> bool {
        self.binding_id == other.binding_id && self.flag_id == other.flag_id
    }
}

impl Eq for BoundCellFlagUdf {}

impl Hash for BoundCellFlagUdf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.binding_id.hash(state);
        self.flag_id.hash(state);
    }
}

impl BoundCellFlagUdf {
    fn value(&self, row_address: u64) -> bool {
        let row_address = lance_core::utils::address::RowAddress::from(row_address);
        let fragment_id = row_address.fragment_id();
        let row_offset = row_address.row_offset();
        match self.snapshot.fragments.get(&fragment_id) {
            Some(FlagFragment::All) => true,
            Some(FlagFragment::Partial(bitmap)) => bitmap.contains(row_offset),
            None => false,
        }
    }
}

#[derive(Debug)]
struct DeferredCellFlagState {
    binding_id: u64,
    snapshot: OnceLock<Arc<CellFlagSnapshot>>,
}

/// A shared snapshot binding initialized by a Lance table provider while its
/// physical scan is planned.
///
/// This allows a logical DataFusion expression above the table scan to retain
/// ordinary synchronous scalar-expression semantics. The provider performs
/// cell-flag I/O asynchronously before DataFusion constructs or executes the
/// parent projection, filter, aggregate, or sort.
#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct DeferredCellFlagBinding {
    flag_id: u32,
    state: Arc<DeferredCellFlagState>,
}

impl DeferredCellFlagBinding {
    /// Stable flag ID resolved from the field and name arguments.
    pub fn flag_id(&self) -> u32 {
        self.flag_id
    }

    /// Initialize this binding with the referenced snapshot state.
    pub fn initialize(&self, fragments: HashMap<u32, FlagFragment>) -> DFResult<()> {
        let covered_fragments = fragments.keys().copied().collect();
        self.initialize_with_coverage(fragments, covered_fragments)
    }

    /// Initialize this binding and record the complete fragment domain for
    /// in which absent entries mean false.
    #[doc(hidden)]
    pub fn initialize_with_coverage(
        &self,
        fragments: HashMap<u32, FlagFragment>,
        covered_fragments: RoaringBitmap,
    ) -> DFResult<()> {
        if self.state.snapshot.get().is_some() {
            return Ok(());
        }
        self.state
            .snapshot
            .set(Arc::new(CellFlagSnapshot {
                fragments: Arc::new(fragments),
                covered_fragments,
            }))
            .map_err(|_| {
                datafusion::error::DataFusionError::Internal(format!(
                    "cell_flag binding {} for flag ID {} was initialized concurrently",
                    self.state.binding_id, self.flag_id
                ))
            })
    }
}

#[derive(Debug, Clone)]
struct DeferredBoundCellFlagUdf {
    binding: DeferredCellFlagBinding,
    signature: Signature,
}

impl PartialEq for DeferredBoundCellFlagUdf {
    fn eq(&self, other: &Self) -> bool {
        self.binding.state.binding_id == other.binding.state.binding_id
            && self.binding.flag_id == other.binding.flag_id
    }
}

impl Eq for DeferredBoundCellFlagUdf {}

impl Hash for DeferredBoundCellFlagUdf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.binding.state.binding_id.hash(state);
        self.binding.flag_id.hash(state);
    }
}

impl DeferredBoundCellFlagUdf {
    fn value(&self, row_address: u64) -> DFResult<bool> {
        let snapshot = self.binding.state.snapshot.get().ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(format!(
                "cell_flag binding {} for flag ID {} was not initialized by its Lance table provider",
                self.binding.state.binding_id, self.binding.flag_id
            ))
        })?;
        let row_address = lance_core::utils::address::RowAddress::from(row_address);
        let fragment_id = row_address.fragment_id();
        let row_offset = row_address.row_offset();
        Ok(match snapshot.fragments.get(&fragment_id) {
            Some(FlagFragment::All) => true,
            Some(FlagFragment::Partial(bitmap)) => bitmap.contains(row_offset),
            None => false,
        })
    }
}

impl ScalarUDFImpl for DeferredBoundCellFlagUdf {
    fn name(&self) -> &str {
        CELL_FLAG_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::Boolean)
    }

    fn return_field_from_args(&self, _args: ReturnFieldArgs) -> DFResult<FieldRef> {
        Ok(Arc::new(Field::new(
            CELL_FLAG_NAME,
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
                                "bound cell_flag expected UInt64 row addresses, got {}",
                                array.data_type()
                            ))
                        })?;
                let values = (0..row_addresses.len())
                    .map(|index| {
                        if row_addresses.is_valid(index) {
                            self.value(row_addresses.value(index))
                        } else {
                            Ok(false)
                        }
                    })
                    .collect::<DFResult<Vec<_>>>()?;
                Ok(ColumnarValue::Array(Arc::new(BooleanArray::from_iter(
                    values.into_iter().map(Some),
                ))))
            }
            ColumnarValue::Scalar(ScalarValue::UInt64(row_address)) => Ok(ColumnarValue::Scalar(
                ScalarValue::Boolean(Some(match row_address {
                    Some(value) => self.value(*value)?,
                    None => false,
                })),
            )),
            value => Err(datafusion::error::DataFusionError::Execution(format!(
                "bound cell_flag expected UInt64 row addresses, got {}",
                value.data_type()
            ))),
        }
    }
}

impl ScalarUDFImpl for BoundCellFlagUdf {
    fn name(&self) -> &str {
        CELL_FLAG_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::Boolean)
    }

    fn return_field_from_args(&self, _args: ReturnFieldArgs) -> DFResult<FieldRef> {
        Ok(Arc::new(Field::new(
            CELL_FLAG_NAME,
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
                                "bound cell_flag expected UInt64 row addresses, got {}",
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
                "bound cell_flag expected UInt64 row addresses, got {}",
                value.data_type()
            ))),
        }
    }
}

static NEXT_CELL_FLAG_BINDING_ID: AtomicU64 = AtomicU64::new(1);

/// Create a snapshot-bound `cell_flag` UDF whose argument is a physical row
/// address. The scanner replaces the user's `(field, name)` arguments with
/// `_rowaddr` only after resolving the registered stable flag ID.
///
/// ```
/// use std::collections::HashMap;
/// use lance_datafusion::udf::{FlagFragment, bound_cell_flag_udf};
///
/// let udf = bound_cell_flag_udf(7, HashMap::from([(3, FlagFragment::All)]));
/// assert_eq!(udf.name(), "cell_flag");
/// ```
pub fn bound_cell_flag_udf(flag_id: u32, fragments: HashMap<u32, FlagFragment>) -> ScalarUDF {
    let covered_fragments = fragments.keys().copied().collect();
    bound_cell_flag_udf_with_coverage(flag_id, fragments, covered_fragments)
}

/// Create a snapshot-bound `cell_flag` UDF with an explicit fragment domain.
///
/// The fragment domain is required by exact row-selection pushdown because a
/// fragment missing from `fragments` is known to be entirely false, while
/// a fragment outside `covered_fragments` is unknown to this binding.
#[doc(hidden)]
pub fn bound_cell_flag_udf_with_coverage(
    flag_id: u32,
    fragments: HashMap<u32, FlagFragment>,
    covered_fragments: RoaringBitmap,
) -> ScalarUDF {
    ScalarUDF::new_from_impl(BoundCellFlagUdf {
        binding_id: NEXT_CELL_FLAG_BINDING_ID.fetch_add(1, Ordering::Relaxed),
        flag_id,
        snapshot: Arc::new(CellFlagSnapshot {
            fragments: Arc::new(fragments),
            covered_fragments,
        }),
        signature: Signature::exact(vec![DataType::UInt64], Volatility::Immutable),
    })
}

/// Create an `cell_flag` UDF and a shared binding that a Lance table
/// provider initializes before execution.
#[doc(hidden)]
pub fn deferred_bound_cell_flag_udf(flag_id: u32) -> (ScalarUDF, DeferredCellFlagBinding) {
    let binding = DeferredCellFlagBinding {
        flag_id,
        state: Arc::new(DeferredCellFlagState {
            binding_id: NEXT_CELL_FLAG_BINDING_ID.fetch_add(1, Ordering::Relaxed),
            snapshot: OnceLock::new(),
        }),
    };
    let udf = ScalarUDF::new_from_impl(DeferredBoundCellFlagUdf {
        binding: binding.clone(),
        signature: Signature::exact(vec![DataType::UInt64], Volatility::Immutable),
    });
    (udf, binding)
}

/// Return true only for the public, unbound logical `cell_flag` UDF.
#[doc(hidden)]
pub fn is_unbound_cell_flag_udf(udf: &ScalarUDF) -> bool {
    udf.inner().downcast_ref::<CellFlagUdf>().is_some()
}

/// Return true only for Lance's internal stable-ID transport UDF.
#[doc(hidden)]
pub fn is_cell_flag_id_udf(udf: &ScalarUDF) -> bool {
    udf.inner().downcast_ref::<CellFlagIdUdf>().is_some()
}

/// Return the stable flag ID carried by an eager or deferred bound UDF.
#[doc(hidden)]
pub fn bound_cell_flag_flag_id(udf: &ScalarUDF) -> Option<u32> {
    udf.inner()
        .downcast_ref::<BoundCellFlagUdf>()
        .map(|bound| bound.flag_id)
        .or_else(|| {
            udf.inner()
                .downcast_ref::<DeferredBoundCellFlagUdf>()
                .map(|bound| bound.binding.flag_id)
        })
}

/// Return the immutable state carried by an initialized bound cell-flag UDF.
#[doc(hidden)]
pub fn bound_cell_flag_snapshot(udf: &ScalarUDF) -> Option<BoundCellFlagSnapshot> {
    if let Some(bound) = udf.inner().downcast_ref::<BoundCellFlagUdf>() {
        return Some((
            bound.flag_id,
            bound.snapshot.fragments.clone(),
            bound.snapshot.covered_fragments.clone(),
        ));
    }
    let bound = udf.inner().downcast_ref::<DeferredBoundCellFlagUdf>()?;
    let snapshot = bound.binding.state.snapshot.get()?;
    Some((
        bound.binding.flag_id,
        snapshot.fragments.clone(),
        snapshot.covered_fragments.clone(),
    ))
}

/// Build the native DataFusion logical expression for
/// `cell_flag(field, name)`. A Lance scanner resolves the pair to a stable flag
/// ID and binds snapshot state before physical planning.
///
/// ```
/// use datafusion::logical_expr::col;
/// use lance_datafusion::udf::cell_flag;
///
/// let expression = cell_flag(col("embedding"), "computed");
/// assert!(expression.to_string().starts_with("cell_flag("));
/// ```
pub fn cell_flag(
    field: datafusion::logical_expr::Expr,
    name: impl Into<String>,
) -> datafusion::logical_expr::Expr {
    datafusion::logical_expr::Expr::ScalarFunction(
        datafusion::logical_expr::expr::ScalarFunction::new_udf(
            Arc::new(CELL_FLAG_UDF.clone()),
            vec![field, datafusion::logical_expr::lit(name.into())],
        ),
    )
}

/// Build the internal Substrait transport expression for a resolved flag ID.
#[doc(hidden)]
pub fn cell_flag_id(flag_id: u32) -> datafusion::logical_expr::Expr {
    datafusion::logical_expr::Expr::ScalarFunction(
        datafusion::logical_expr::expr::ScalarFunction::new_udf(
            Arc::new(CELL_FLAG_ID_UDF.clone()),
            vec![datafusion::logical_expr::lit(flag_id)],
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
/// Unbound logical `cell_flag(field, name)` expression registered with DataFusion
/// and Substrait. Lance scanners replace it with a snapshot-bound instance.
///
/// ```
/// use lance_datafusion::udf::CELL_FLAG_UDF;
/// assert_eq!(CELL_FLAG_UDF.name(), "cell_flag");
/// ```
pub static CELL_FLAG_UDF: LazyLock<ScalarUDF> =
    LazyLock::new(|| ScalarUDF::new_from_impl(CellFlagUdf::default()));
/// Internal stable-ID transport for distributed Substrait plans.
#[doc(hidden)]
pub static CELL_FLAG_ID_UDF: LazyLock<ScalarUDF> =
    LazyLock::new(|| ScalarUDF::new_from_impl(CellFlagIdUdf::default()));

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
