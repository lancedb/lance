// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Exec plan planner

use std::collections::{BTreeSet, VecDeque};
use std::sync::Arc;

use arrow::compute::CastOptions;
use arrow_array::ListArray;
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType as ArrowDataType, Field, SchemaRef, TimeUnit};
use arrow_select::concat::concat;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion, TreeNodeVisitor};
use datafusion::common::DFSchema;
use datafusion::config::ConfigOptions;
use datafusion::error::Result as DFResult;
use datafusion::execution::config::SessionConfig;
use datafusion::execution::context::SessionState;
use datafusion::execution::runtime_env::{RuntimeConfig, RuntimeEnv};
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::{
    AggregateUDF, ColumnarValue, GetFieldAccess, GetIndexedField, ScalarUDF, ScalarUDFImpl,
    Signature, Volatility, WindowUDF,
};
use datafusion::optimizer::simplify_expressions::SimplifyContext;
use datafusion::physical_optimizer::optimizer::PhysicalOptimizer;
use datafusion::sql::planner::{ContextProvider, ParserOptions, PlannerContext, SqlToRel};
use datafusion::sql::sqlparser::ast::{
    Array as SQLArray, BinaryOperator, DataType as SQLDataType, ExactNumberInfo, Expr as SQLExpr,
    Function, FunctionArg, FunctionArgExpr, Ident, TimezoneInfo, UnaryOperator, Value,
};
use datafusion::{
    common::Column,
    logical_expr::{col, BinaryExpr, Like, Operator},
    physical_expr::execution_props::ExecutionProps,
    physical_plan::PhysicalExpr,
    prelude::Expr,
    scalar::ScalarValue,
};
use datafusion_functions::core::getfield::GetFieldFunc;
use lance_arrow::cast::cast_with_options;
use lance_datafusion::expr::safe_coerce_scalar;
use lance_index::scalar::expression::{
    apply_scalar_indices, IndexInformationProvider, ScalarIndexExpr,
};
use snafu::{location, Location};

use crate::datafusion::logical_expr::{coerce_filter_type_to_boolean, get_as_string_scalar_opt};
use crate::utils::sql::parse_sql_expr;
use crate::{
    datafusion::logical_expr::resolve_expr, datatypes::Schema, utils::sql::parse_sql_filter, Error,
    Result,
};

#[derive(Default, Debug)]
pub struct FilterPlan {
    pub index_query: Option<ScalarIndexExpr>,
    pub refine_expr: Option<Expr>,
    pub full_expr: Option<Expr>,
}

impl FilterPlan {
    pub fn refine_columns(&self) -> Vec<String> {
        self.refine_expr
            .as_ref()
            .map(Planner::column_names_in_expr)
            .unwrap_or_default()
    }

    /// Return true if this has a refine step, regardless of the status of prefilter
    pub fn has_refine(&self) -> bool {
        self.refine_expr.is_some()
    }
}

#[derive(Debug, Clone)]
struct CastListF16Udf {
    signature: Signature,
}

impl CastListF16Udf {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for CastListF16Udf {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "_cast_list_f16"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[ArrowDataType]) -> DFResult<ArrowDataType> {
        let input = &arg_types[0];
        match input {
            ArrowDataType::FixedSizeList(field, size) => {
                if field.data_type() != &ArrowDataType::Float32
                    && field.data_type() != &ArrowDataType::Float16
                {
                    return Err(datafusion::error::DataFusionError::Execution(
                        "cast_list_f16 only supports list of float32 or float16".to_string(),
                    ));
                }
                Ok(ArrowDataType::FixedSizeList(
                    Arc::new(Field::new(
                        field.name(),
                        ArrowDataType::Float16,
                        field.is_nullable(),
                    )),
                    *size,
                ))
            }
            ArrowDataType::List(field) => {
                if field.data_type() != &ArrowDataType::Float32
                    && field.data_type() != &ArrowDataType::Float16
                {
                    return Err(datafusion::error::DataFusionError::Execution(
                        "cast_list_f16 only supports list of float32 or float16".to_string(),
                    ));
                }
                Ok(ArrowDataType::List(Arc::new(Field::new(
                    field.name(),
                    ArrowDataType::Float16,
                    field.is_nullable(),
                ))))
            }
            _ => Err(datafusion::error::DataFusionError::Execution(
                "cast_list_f16 only supports FixedSizeList/List arguments".to_string(),
            )),
        }
    }

    fn invoke(&self, args: &[ColumnarValue]) -> DFResult<ColumnarValue> {
        let ColumnarValue::Array(arr) = &args[0] else {
            return Err(datafusion::error::DataFusionError::Execution(
                "cast_list_f16 only supports array arguments".to_string(),
            ));
        };

        let to_type = match arr.data_type() {
            ArrowDataType::FixedSizeList(field, size) => ArrowDataType::FixedSizeList(
                Arc::new(Field::new(
                    field.name(),
                    ArrowDataType::Float16,
                    field.is_nullable(),
                )),
                *size,
            ),
            ArrowDataType::List(field) => ArrowDataType::List(Arc::new(Field::new(
                field.name(),
                ArrowDataType::Float16,
                field.is_nullable(),
            ))),
            _ => {
                return Err(datafusion::error::DataFusionError::Execution(
                    "cast_list_f16 only supports array arguments".to_string(),
                ));
            }
        };

        let res = cast_with_options(arr.as_ref(), &to_type, &CastOptions::default())?;
        Ok(ColumnarValue::Array(res))
    }
}

// Adapter that instructs datafusion how lance expects expressions to be interpreted
struct LanceContextProvider {
    options: datafusion::config::ConfigOptions,
    state: SessionState,
}

impl Default for LanceContextProvider {
    fn default() -> Self {
        let config = SessionConfig::new();
        let runtime_config = RuntimeConfig::new();
        let runtime = Arc::new(RuntimeEnv::new(runtime_config).unwrap());
        let state = SessionState::new_with_config_rt(config, runtime);
        Self {
            options: ConfigOptions::default(),
            state,
        }
    }
}

impl ContextProvider for LanceContextProvider {
    fn get_table_source(
        &self,
        name: datafusion::sql::TableReference,
    ) -> DFResult<Arc<dyn datafusion::logical_expr::TableSource>> {
        Err(datafusion::error::DataFusionError::NotImplemented(format!(
            "Attempt to reference inner table {} not supported",
            name
        )))
    }

    fn get_aggregate_meta(&self, name: &str) -> Option<Arc<AggregateUDF>> {
        self.state.aggregate_functions().get(name).cloned()
    }

    fn get_window_meta(&self, name: &str) -> Option<Arc<WindowUDF>> {
        self.state.window_functions().get(name).cloned()
    }

    fn get_function_meta(&self, f: &str) -> Option<Arc<ScalarUDF>> {
        match f {
            // TODO: cast should go thru CAST syntax instead of UDF
            // Going thru UDF makes it hard for the optimizer to find no-ops
            "_cast_list_f16" => Some(Arc::new(ScalarUDF::new_from_impl(CastListF16Udf::new()))),
            _ => self.state.scalar_functions().get(f).cloned(),
        }
    }

    fn get_variable_type(&self, _: &[String]) -> Option<ArrowDataType> {
        // Variables (things like @@LANGUAGE) not supported
        None
    }

    fn options(&self) -> &datafusion::config::ConfigOptions {
        &self.options
    }

    fn udfs_names(&self) -> Vec<String> {
        self.state.scalar_functions().keys().cloned().collect()
    }

    fn udafs_names(&self) -> Vec<String> {
        self.state.aggregate_functions().keys().cloned().collect()
    }

    fn udwfs_names(&self) -> Vec<String> {
        self.state.window_functions().keys().cloned().collect()
    }
}

pub struct Planner {
    schema: SchemaRef,
}

impl Planner {
    pub fn new(schema: SchemaRef) -> Self {
        Self { schema }
    }

    fn column(idents: &[Ident]) -> Expr {
        let mut column = col(&idents[0].value);
        for ident in &idents[1..] {
            column = Expr::ScalarFunction(ScalarFunction {
                args: vec![
                    column,
                    Expr::Literal(ScalarValue::Utf8(Some(ident.value.clone()))),
                ],
                func_def: datafusion::logical_expr::ScalarFunctionDefinition::UDF(Arc::new(
                    ScalarUDF::new_from_impl(GetFieldFunc::default()),
                )),
            });
        }
        column
    }

    fn binary_op(&self, op: &BinaryOperator) -> Result<Operator> {
        Ok(match op {
            BinaryOperator::Plus => Operator::Plus,
            BinaryOperator::Minus => Operator::Minus,
            BinaryOperator::Multiply => Operator::Multiply,
            BinaryOperator::Divide => Operator::Divide,
            BinaryOperator::Modulo => Operator::Modulo,
            BinaryOperator::StringConcat => Operator::StringConcat,
            BinaryOperator::Gt => Operator::Gt,
            BinaryOperator::Lt => Operator::Lt,
            BinaryOperator::GtEq => Operator::GtEq,
            BinaryOperator::LtEq => Operator::LtEq,
            BinaryOperator::Eq => Operator::Eq,
            BinaryOperator::NotEq => Operator::NotEq,
            BinaryOperator::And => Operator::And,
            BinaryOperator::Or => Operator::Or,
            _ => {
                return Err(Error::io(
                    //TODO: Define more granular Error and wrap it in here.
                    format!("Operator {op} is not supported"),
                    location!(),
                ));
            }
        })
    }

    fn binary_expr(&self, left: &SQLExpr, op: &BinaryOperator, right: &SQLExpr) -> Result<Expr> {
        Ok(Expr::BinaryExpr(BinaryExpr::new(
            Box::new(self.parse_sql_expr(left)?),
            self.binary_op(op)?,
            Box::new(self.parse_sql_expr(right)?),
        )))
    }

    fn unary_expr(&self, op: &UnaryOperator, expr: &SQLExpr) -> Result<Expr> {
        Ok(match op {
            UnaryOperator::Not | UnaryOperator::PGBitwiseNot => {
                Expr::Not(Box::new(self.parse_sql_expr(expr)?))
            }

            UnaryOperator::Minus => {
                use datafusion::logical_expr::lit;
                match expr {
                    SQLExpr::Value(Value::Number(n, _)) => match n.parse::<i64>() {
                        Ok(n) => lit(-n),
                        Err(_) => lit(-n
                            .parse::<f64>()
                            .map_err(|_e| {
                                Error::io(          // TODO: Define more granular Error and wrap it in here.
                                    format!("negative operator can be only applied to integer and float operands, got: {n}"),
                                    location!(),
                                )
                            })?),
                    },
                    _ => {
                        Expr::Negative(Box::new(self.parse_sql_expr(expr)?))
                    }
                }
            }

            _ => {
                return Err(Error::io(
                    // TODO: Define more granular Error and wrap it in here.
                    format!("Unary operator '{:?}' is not supported", op),
                    location!(),
                ));
            }
        })
    }

    // See datafusion `sqlToRel::parse_sql_number()`
    fn number(&self, value: &str) -> Result<Expr> {
        use datafusion::logical_expr::lit;
        if let Ok(n) = value.parse::<i64>() {
            Ok(lit(n))
        } else {
            value.parse::<f64>().map(lit).map_err(|_| {
                Error::io(
                    // TODO: Define more granular Error and wrap it in here.
                    format!("'{value}' is not supported number value."),
                    location!(),
                )
            })
        }
    }

    fn value(&self, value: &Value) -> Result<Expr> {
        Ok(match value {
            Value::Number(v, _) => self.number(v.as_str())?,
            Value::SingleQuotedString(s) => Expr::Literal(ScalarValue::Utf8(Some(s.clone()))),
            Value::DollarQuotedString(_) => todo!(),
            Value::EscapedStringLiteral(_) => todo!(),
            Value::NationalStringLiteral(_) => todo!(),
            Value::HexStringLiteral(_) => todo!(),
            Value::DoubleQuotedString(s) => Expr::Literal(ScalarValue::Utf8(Some(s.clone()))),
            Value::Boolean(v) => Expr::Literal(ScalarValue::Boolean(Some(*v))),
            Value::Null => Expr::Literal(ScalarValue::Null),
            Value::Placeholder(_) => todo!(),
            Value::UnQuotedString(_) => todo!(),
            Value::SingleQuotedByteStringLiteral(_) => todo!(),
            Value::DoubleQuotedByteStringLiteral(_) => todo!(),
            Value::RawStringLiteral(_) => todo!(),
        })
    }

    fn parse_function_args(&self, func_args: &FunctionArg) -> Result<Expr> {
        match func_args {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => self.parse_sql_expr(expr),
            _ => Err(Error::io(
                // TODO: Define more granular Error and wrap it in here.
                format!("Unsupported function args: {:?}", func_args),
                location!(),
            )),
        }
    }

    // We now use datafusion to parse functions.  This allows us to use datafusion's
    // entire collection of functions (previously we had just hard-coded support for two functions).
    //
    // Unfortunately, one of those two functions was is_valid and the reason we needed it was because
    // this is a function that comes from duckdb.  Datafusion does not consider is_valid to be a function
    // but rather an AST node (Expr::IsNotNull) and so we need to handle this case specially.
    fn legacy_parse_function(&self, func: &Function) -> Result<Expr> {
        if func.args.len() != 1 {
            return Err(Error::io(
                // TODO: Define more granular Error and wrap it in here.
                format!("is_valid only support 1 args, got {}", func.args.len()),
                location!(),
            ));
        }
        Ok(Expr::IsNotNull(Box::new(
            self.parse_function_args(&func.args[0])?,
        )))
    }

    fn parse_function(&self, function: SQLExpr) -> Result<Expr> {
        if let SQLExpr::Function(function) = &function {
            if !function.name.0.is_empty() && function.name.0[0].value == "is_valid" {
                return self.legacy_parse_function(function);
            }
        }
        let context_provider = LanceContextProvider::default();
        let sql_to_rel = SqlToRel::new_with_options(
            &context_provider,
            ParserOptions {
                parse_float_as_decimal: false,
                enable_ident_normalization: false,
            },
        );
        let mut planner_context = PlannerContext::default();
        let schema = DFSchema::try_from(self.schema.as_ref().clone())?;
        Ok(sql_to_rel.sql_to_expr(function, &schema, &mut planner_context)?)
    }

    fn parse_type(&self, data_type: &SQLDataType) -> Result<ArrowDataType> {
        const SUPPORTED_TYPES: [&str; 13] = [
            "int [unsigned]",
            "tinyint [unsigned]",
            "smallint [unsigned]",
            "bigint [unsigned]",
            "float",
            "double",
            "string",
            "binary",
            "date",
            "timestamp(precision)",
            "datetime(precision)",
            "decimal(precision,scale)",
            "boolean",
        ];
        match data_type {
            SQLDataType::String(_) => Ok(ArrowDataType::Utf8),
            SQLDataType::Binary(_) => Ok(ArrowDataType::Binary),
            SQLDataType::Float(_) => Ok(ArrowDataType::Float32),
            SQLDataType::Double => Ok(ArrowDataType::Float64),
            SQLDataType::Boolean => Ok(ArrowDataType::Boolean),
            SQLDataType::TinyInt(_) => Ok(ArrowDataType::Int8),
            SQLDataType::SmallInt(_) => Ok(ArrowDataType::Int16),
            SQLDataType::Int(_) | SQLDataType::Integer(_) => Ok(ArrowDataType::Int32),
            SQLDataType::BigInt(_) => Ok(ArrowDataType::Int64),
            SQLDataType::UnsignedTinyInt(_) => Ok(ArrowDataType::UInt8),
            SQLDataType::UnsignedSmallInt(_) => Ok(ArrowDataType::UInt16),
            SQLDataType::UnsignedInt(_) | SQLDataType::UnsignedInteger(_) => {
                Ok(ArrowDataType::UInt32)
            }
            SQLDataType::UnsignedBigInt(_) => Ok(ArrowDataType::UInt64),
            SQLDataType::Date => Ok(ArrowDataType::Date32),
            SQLDataType::Timestamp(resolution, tz) => {
                match tz {
                    TimezoneInfo::None => {}
                    _ => {
                        return Err(Error::io(
                            // TODO: Define more granular Error and wrap it in here.
                            "Timezone not supported in timestamp".to_string(),
                            location!(),
                        ));
                    }
                };
                let time_unit = match resolution {
                    // Default to microsecond to match PyArrow
                    None => TimeUnit::Microsecond,
                    Some(0) => TimeUnit::Second,
                    Some(3) => TimeUnit::Millisecond,
                    Some(6) => TimeUnit::Microsecond,
                    Some(9) => TimeUnit::Nanosecond,
                    _ => {
                        return Err(Error::io(
                            // TODO: Define more granular Error and wrap it in here.
                            format!("Unsupported datetime resolution: {:?}", resolution),
                            location!(),
                        ));
                    }
                };
                Ok(ArrowDataType::Timestamp(time_unit, None))
            }
            SQLDataType::Datetime(resolution) => {
                let time_unit = match resolution {
                    None => TimeUnit::Microsecond,
                    Some(0) => TimeUnit::Second,
                    Some(3) => TimeUnit::Millisecond,
                    Some(6) => TimeUnit::Microsecond,
                    Some(9) => TimeUnit::Nanosecond,
                    _ => {
                        return Err(Error::io(
                            // TODO: Define more granular Error and wrap it in here.
                            format!("Unsupported datetime resolution: {:?}", resolution),
                            location!(),
                        ));
                    }
                };
                Ok(ArrowDataType::Timestamp(time_unit, None))
            }
            SQLDataType::Decimal(number_info) => match number_info {
                ExactNumberInfo::PrecisionAndScale(precision, scale) => {
                    Ok(ArrowDataType::Decimal128(*precision as u8, *scale as i8))
                }
                _ => Err(Error::io(
                    format!(
                        "Must provide precision and scale for decimal: {:?}",
                        number_info
                    ),
                    location!(),
                )),
            },
            _ => Err(Error::io(
                // TODO: Define more granular Error and wrap it in here.
                format!(
                    "Unsupported data type: {:?}. Supported types: {:?}",
                    data_type, SUPPORTED_TYPES
                ),
                location!(),
            )),
        }
    }

    fn parse_sql_expr(&self, expr: &SQLExpr) -> Result<Expr> {
        match expr {
            SQLExpr::Identifier(id) => {
                // Users can pass string literals wrapped in `"`.
                // (Normally SQL only allows single quotes.)
                if id.quote_style == Some('"') {
                    Ok(Expr::Literal(ScalarValue::Utf8(Some(id.value.clone()))))
                // Users can wrap identifiers with ` to reference non-standard
                // names, such as uppercase or spaces.
                } else if id.quote_style == Some('`') {
                    Ok(Expr::Column(Column::from_name(id.value.clone())))
                } else {
                    Ok(Self::column(vec![id.clone()].as_slice()))
                }
            }
            SQLExpr::CompoundIdentifier(ids) => Ok(Self::column(ids.as_slice())),
            SQLExpr::BinaryOp { left, op, right } => self.binary_expr(left, op, right),
            SQLExpr::UnaryOp { op, expr } => self.unary_expr(op, expr),
            SQLExpr::Value(value) => self.value(value),
            SQLExpr::Array(SQLArray { elem, .. }) => {
                let mut values = vec![];

                for expr in elem {
                    if let SQLExpr::Value(value) = expr {
                        if let Expr::Literal(value) = self.value(value)? {
                            values.push(value);
                        } else {
                            return Err(Error::Internal {
                                message: "Expected a literal value in array.".into(),
                                location: location!(),
                            });
                        }
                    } else {
                        return Err(Error::io(
                            // TODO: Define more granular Error and wrap it in here.
                            "Only arrays of literals are supported in lance.",
                            location!(),
                        ));
                    }
                }

                let field = if !values.is_empty() {
                    let data_type = values[0].data_type();

                    for value in &mut values {
                        if value.data_type() != data_type {
                            *value = safe_coerce_scalar(value, &data_type).ok_or_else(|| Error::io(     // TODO: Define more granular Error and wrap it in here.
                                format!("Array expressions must have a consistent datatype. Expected: {}, got: {}", data_type, value.data_type()),
                                location!()
                            ))?;
                        }
                    }
                    Field::new("item", data_type.clone(), true)
                } else {
                    Field::new("item", ArrowDataType::Null, true)
                };

                let values = values
                    .into_iter()
                    .map(|v| v.to_array().map_err(Error::from))
                    .collect::<Result<Vec<_>>>()?;
                let array_refs = values.iter().map(|v| v.as_ref()).collect::<Vec<_>>();
                let values = concat(&array_refs)?;
                let values = ListArray::try_new(
                    field.into(),
                    OffsetBuffer::from_lengths([values.len()]),
                    values,
                    None,
                )?;

                Ok(Expr::Literal(ScalarValue::List(Arc::new(values))))
            }
            // For example, DATE '2020-01-01'
            SQLExpr::TypedString { data_type, value } => {
                Ok(Expr::Cast(datafusion::logical_expr::Cast {
                    expr: Box::new(Expr::Literal(ScalarValue::Utf8(Some(value.clone())))),
                    data_type: self.parse_type(data_type)?,
                }))
            }
            SQLExpr::IsFalse(expr) => Ok(Expr::IsFalse(Box::new(self.parse_sql_expr(expr)?))),
            SQLExpr::IsNotFalse(_) => Ok(Expr::IsNotFalse(Box::new(self.parse_sql_expr(expr)?))),
            SQLExpr::IsTrue(expr) => Ok(Expr::IsTrue(Box::new(self.parse_sql_expr(expr)?))),
            SQLExpr::IsNotTrue(expr) => Ok(Expr::IsNotTrue(Box::new(self.parse_sql_expr(expr)?))),
            SQLExpr::IsNull(expr) => Ok(Expr::IsNull(Box::new(self.parse_sql_expr(expr)?))),
            SQLExpr::IsNotNull(expr) => Ok(Expr::IsNotNull(Box::new(self.parse_sql_expr(expr)?))),
            SQLExpr::InList {
                expr,
                list,
                negated,
            } => {
                let value_expr = self.parse_sql_expr(expr)?;
                let list_exprs = list
                    .iter()
                    .map(|e| self.parse_sql_expr(e))
                    .collect::<Result<Vec<_>>>()?;
                Ok(value_expr.in_list(list_exprs, *negated))
            }
            SQLExpr::Nested(inner) => self.parse_sql_expr(inner.as_ref()),
            SQLExpr::Function(_) => self.parse_function(expr.clone()),
            SQLExpr::ILike {
                negated,
                expr,
                pattern,
                escape_char,
            } => Ok(Expr::Like(Like::new(
                *negated,
                Box::new(self.parse_sql_expr(expr)?),
                Box::new(self.parse_sql_expr(pattern)?),
                *escape_char,
                true,
            ))),
            SQLExpr::Like {
                negated,
                expr,
                pattern,
                escape_char,
            } => Ok(Expr::Like(Like::new(
                *negated,
                Box::new(self.parse_sql_expr(expr)?),
                Box::new(self.parse_sql_expr(pattern)?),
                *escape_char,
                false,
            ))),
            SQLExpr::Cast {
                expr, data_type, ..
            } => Ok(Expr::Cast(datafusion::logical_expr::Cast {
                expr: Box::new(self.parse_sql_expr(expr)?),
                data_type: self.parse_type(data_type)?,
            })),
            _ => Err(Error::io(
                format!("Expression '{expr}' is not supported SQL in lance"), // TODO: Define more granular Error and wrap it in here.
                location!(),
            )),
        }
    }

    /// Create Logical [Expr] from a SQL filter clause.
    ///
    /// Note: the returned expression must be passed through [optimize_expr()]
    /// before being passed to [create_physical_expr()].
    pub fn parse_filter(&self, filter: &str) -> Result<Expr> {
        // Allow sqlparser to parse filter as part of ONE SQL statement.
        let ast_expr = parse_sql_filter(filter)?;
        let expr = self.parse_sql_expr(&ast_expr)?;
        let schema = Schema::try_from(self.schema.as_ref())?;
        let resolved = resolve_expr(&expr, &schema)?;
        coerce_filter_type_to_boolean(resolved)
    }

    /// Create Logical [Expr] from a SQL expression.
    ///
    /// Note: the returned expression must be passed through [optimize_filter()]
    /// before being passed to [create_physical_expr()].
    pub fn parse_expr(&self, expr: &str) -> Result<Expr> {
        let ast_expr = parse_sql_expr(expr)?;
        let expr = self.parse_sql_expr(&ast_expr)?;
        let schema = Schema::try_from(self.schema.as_ref())?;
        let resolved = resolve_expr(&expr, &schema)?;
        Ok(resolved)
    }

    /// Optimize the filter expression and coerce data types.
    pub fn optimize_expr(&self, expr: Expr) -> Result<Expr> {
        let df_schema = Arc::new(DFSchema::try_from(self.schema.as_ref().clone())?);

        // DataFusion needs the simplify and coerce passes to be applied before
        // expressions can be handled by the physical planner.
        let props = ExecutionProps::default();
        let simplify_context = SimplifyContext::new(&props).with_schema(df_schema.clone());
        let simplifier =
            datafusion::optimizer::simplify_expressions::ExprSimplifier::new(simplify_context);

        let expr = simplifier.simplify(expr.clone())?;
        let expr = simplifier.coerce(expr, df_schema.clone())?;

        Ok(expr)
    }

    /// Create the [`PhysicalExpr`] from a logical [`Expr`]
    pub fn create_physical_expr(&self, expr: &Expr) -> Result<Arc<dyn PhysicalExpr>> {
        let df_schema = Arc::new(DFSchema::try_from(self.schema.as_ref().clone())?);

        Ok(datafusion::physical_expr::create_physical_expr(
            expr,
            df_schema.as_ref(),
            &Default::default(),
        )?)
    }

    /// Collect the columns in the expression.
    ///
    /// The columns are returned in sorted order.
    pub fn column_names_in_expr(expr: &Expr) -> Vec<String> {
        let mut visitor = ColumnCapturingVisitor {
            current_path: VecDeque::new(),
            columns: BTreeSet::new(),
        };
        expr.visit(&mut visitor).unwrap();
        visitor.columns.into_iter().collect()
    }

    /// Determine how to apply a provided filter
    ///
    /// We parse the filter into a logical expression.  We then
    /// split the logical expression into a portion that can be
    /// satisfied by an index search (of one or more indices) and
    /// a refine portion that must be applied after the index search
    pub fn create_filter_plan(
        &self,
        filter: Expr,
        index_info: &dyn IndexInformationProvider,
        use_scalar_index: bool,
    ) -> Result<FilterPlan> {
        let logical_expr = self.optimize_expr(filter)?;
        if use_scalar_index {
            let indexed_expr = apply_scalar_indices(logical_expr.clone(), index_info);
            Ok(FilterPlan {
                index_query: indexed_expr.scalar_query,
                refine_expr: indexed_expr.refine_expr,
                full_expr: Some(logical_expr),
            })
        } else {
            Ok(FilterPlan {
                index_query: None,
                refine_expr: Some(logical_expr.clone()),
                full_expr: Some(logical_expr),
            })
        }
    }

    pub fn get_physical_optimizer() -> PhysicalOptimizer {
        PhysicalOptimizer::with_rules(vec![
            Arc::new(crate::io::exec::optimizer::CoalesceTake),
            Arc::new(crate::io::exec::optimizer::SimplifyProjection),
        ])
    }
}

struct ColumnCapturingVisitor {
    // Current column path. If this is empty, we are not in a column expression.
    current_path: VecDeque<String>,
    columns: BTreeSet<String>,
}

impl TreeNodeVisitor for ColumnCapturingVisitor {
    type Node = Expr;

    fn f_down(&mut self, node: &Self::Node) -> DFResult<TreeNodeRecursion> {
        match node {
            Expr::Column(Column { name, .. }) => {
                let mut path = name.clone();
                for part in self.current_path.drain(..) {
                    path.push('.');
                    path.push_str(&part);
                }
                self.columns.insert(path);
                self.current_path.clear();
            }
            Expr::ScalarFunction(udf) => {
                if udf.name() == GetFieldFunc::default().name() {
                    if let Some(name) = get_as_string_scalar_opt(&udf.args[1]) {
                        self.current_path.push_front(name.to_string())
                    } else {
                        self.current_path.clear();
                    }
                } else {
                    self.current_path.clear();
                }
            }
            Expr::GetIndexedField(GetIndexedField {
                expr: _,
                field: GetFieldAccess::NamedStructField { name },
            }) => {
                self.current_path.push_front(name.to_string());
            }
            _ => {
                self.current_path.clear();
            }
        }

        Ok(TreeNodeRecursion::Continue)
    }
}

#[cfg(test)]
mod tests {

    use crate::datafusion::logical_expr::tests::ExprExt;

    use super::*;

    use arrow_array::{
        ArrayRef, BooleanArray, Float32Array, Int32Array, Int64Array, RecordBatch, StringArray,
        StructArray, TimestampMicrosecondArray, TimestampMillisecondArray,
        TimestampNanosecondArray, TimestampSecondArray,
    };
    use arrow_schema::{DataType, Fields, Schema};
    use datafusion::logical_expr::{lit, Cast, ScalarFunctionDefinition};

    #[test]
    fn test_parse_filter_simple() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("s", DataType::Utf8, true),
            Field::new(
                "st",
                DataType::Struct(Fields::from(vec![
                    Field::new("x", DataType::Float32, false),
                    Field::new("y", DataType::Float32, false),
                ])),
                true,
            ),
        ]));

        let planner = Planner::new(schema.clone());

        let expected = col("i")
            .gt(lit(3_i32))
            .and(col("st").field_newstyle("x").lt_eq(lit(5.0_f32)))
            .and(
                col("s")
                    .eq(lit("str-4"))
                    .or(col("s").in_list(vec![lit("str-4"), lit("str-5")], false)),
            );

        // double quotes
        let expr = planner
            .parse_filter("i > 3 AND st.x <= 5.0 AND (s == 'str-4' OR s in ('str-4', 'str-5'))")
            .unwrap();
        assert_eq!(expr, expected);

        // single quote
        let expr = planner
            .parse_filter("i > 3 AND st.x <= 5.0 AND (s = 'str-4' OR s in ('str-4', 'str-5'))")
            .unwrap();

        let physical_expr = planner.create_physical_expr(&expr).unwrap();
        println!("Physical expr: {:#?}", physical_expr);

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from_iter_values(0..10)) as ArrayRef,
                Arc::new(StringArray::from_iter_values(
                    (0..10).map(|v| format!("str-{}", v)),
                )),
                Arc::new(StructArray::from(vec![
                    (
                        Arc::new(Field::new("x", DataType::Float32, false)),
                        Arc::new(Float32Array::from_iter_values((0..10).map(|v| v as f32)))
                            as ArrayRef,
                    ),
                    (
                        Arc::new(Field::new("y", DataType::Float32, false)),
                        Arc::new(Float32Array::from_iter_values(
                            (0..10).map(|v| (v * 10) as f32),
                        )),
                    ),
                ])),
            ],
        )
        .unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).unwrap().as_ref(),
            &BooleanArray::from(vec![
                false, false, false, false, true, true, false, false, false, false
            ])
        );
    }

    #[test]
    fn test_nested_col_refs() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("s0", DataType::Utf8, true),
            Field::new(
                "st",
                DataType::Struct(Fields::from(vec![
                    Field::new("s1", DataType::Utf8, true),
                    Field::new(
                        "st",
                        DataType::Struct(Fields::from(vec![Field::new(
                            "s2",
                            DataType::Utf8,
                            true,
                        )])),
                        true,
                    ),
                ])),
                true,
            ),
        ]));

        let planner = Planner::new(schema.clone());

        fn assert_column_eq(planner: &Planner, expr: &str, expected: &Expr) {
            let expr = planner.parse_filter(&format!("{expr} = 'val'")).unwrap();
            assert!(matches!(
                expr,
                Expr::BinaryExpr(BinaryExpr {
                    left: _,
                    op: Operator::Eq,
                    right: _
                })
            ));
            if let Expr::BinaryExpr(BinaryExpr { left, .. }) = expr {
                assert_eq!(left.as_ref(), expected);
            }
        }

        let expected = Expr::Column(Column {
            relation: None,
            name: "s0".to_string(),
        });
        assert_column_eq(&planner, "s0", &expected);
        assert_column_eq(&planner, "`s0`", &expected);

        let expected = Expr::ScalarFunction(ScalarFunction {
            func_def: ScalarFunctionDefinition::UDF(Arc::new(ScalarUDF::new_from_impl(
                GetFieldFunc::default(),
            ))),
            args: vec![
                Expr::Column(Column {
                    relation: None,
                    name: "st".to_string(),
                }),
                Expr::Literal(ScalarValue::Utf8(Some("s1".to_string()))),
            ],
        });
        assert_column_eq(&planner, "st.s1", &expected);
        assert_column_eq(&planner, "`st`.`s1`", &expected);
        assert_column_eq(&planner, "st.`s1`", &expected);

        let expected = Expr::ScalarFunction(ScalarFunction {
            func_def: ScalarFunctionDefinition::UDF(Arc::new(ScalarUDF::new_from_impl(
                GetFieldFunc::default(),
            ))),
            args: vec![
                Expr::ScalarFunction(ScalarFunction {
                    func_def: ScalarFunctionDefinition::UDF(Arc::new(ScalarUDF::new_from_impl(
                        GetFieldFunc::default(),
                    ))),
                    args: vec![
                        Expr::Column(Column {
                            relation: None,
                            name: "st".to_string(),
                        }),
                        Expr::Literal(ScalarValue::Utf8(Some("st".to_string()))),
                    ],
                }),
                Expr::Literal(ScalarValue::Utf8(Some("s2".to_string()))),
            ],
        });

        assert_column_eq(&planner, "st.st.s2", &expected);
        assert_column_eq(&planner, "`st`.`st`.`s2`", &expected);
        assert_column_eq(&planner, "st.st.`s2`", &expected);
    }

    #[test]
    fn test_negative_expressions() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));

        let planner = Planner::new(schema.clone());

        let expected = col("x")
            .gt(lit(-3_i64))
            .and(col("x").lt(-(lit(-5_i64) + lit(3_i64))));

        let expr = planner.parse_filter("x > -3 AND x < -(-5 + 3)").unwrap();

        assert_eq!(expr, expected);

        let physical_expr = planner.create_physical_expr(&expr).unwrap();

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from_iter_values(-5..5)) as ArrayRef],
        )
        .unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).unwrap().as_ref(),
            &BooleanArray::from(vec![
                false, false, false, true, true, true, true, false, false, false
            ])
        );
    }

    #[test]
    fn test_sql_like() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));

        let planner = Planner::new(schema.clone());

        let expected = col("s").like(lit("str-4"));
        // single quote
        let expr = planner.parse_filter("s LIKE 'str-4'").unwrap();
        assert_eq!(expr, expected);
        let physical_expr = planner.create_physical_expr(&expr).unwrap();

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from_iter_values(
                (0..10).map(|v| format!("str-{}", v)),
            ))],
        )
        .unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).unwrap().as_ref(),
            &BooleanArray::from(vec![
                false, false, false, false, true, false, false, false, false, false
            ])
        );
    }

    #[test]
    fn test_not_like() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));

        let planner = Planner::new(schema.clone());

        let expected = col("s").not_like(lit("str-4"));
        // single quote
        let expr = planner.parse_filter("s NOT LIKE 'str-4'").unwrap();
        assert_eq!(expr, expected);
        let physical_expr = planner.create_physical_expr(&expr).unwrap();

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from_iter_values(
                (0..10).map(|v| format!("str-{}", v)),
            ))],
        )
        .unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).unwrap().as_ref(),
            &BooleanArray::from(vec![
                true, true, true, true, false, true, true, true, true, true
            ])
        );
    }

    #[test]
    fn test_sql_is_in() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));

        let planner = Planner::new(schema.clone());

        let expected = col("s").in_list(vec![lit("str-4"), lit("str-5")], false);
        // single quote
        let expr = planner.parse_filter("s IN ('str-4', 'str-5')").unwrap();
        assert_eq!(expr, expected);
        let physical_expr = planner.create_physical_expr(&expr).unwrap();

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from_iter_values(
                (0..10).map(|v| format!("str-{}", v)),
            ))],
        )
        .unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).unwrap().as_ref(),
            &BooleanArray::from(vec![
                false, false, false, false, true, true, false, false, false, false
            ])
        );
    }

    #[test]
    fn test_sql_is_null() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));

        let planner = Planner::new(schema.clone());

        let expected = col("s").is_null();
        let expr = planner.parse_filter("s IS NULL").unwrap();
        assert_eq!(expr, expected);
        let physical_expr = planner.create_physical_expr(&expr).unwrap();

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from_iter((0..10).map(|v| {
                if v % 3 == 0 {
                    Some(format!("str-{}", v))
                } else {
                    None
                }
            })))],
        )
        .unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).unwrap().as_ref(),
            &BooleanArray::from(vec![
                false, true, true, false, true, true, false, true, true, false
            ])
        );

        let expr = planner.parse_filter("s IS NOT NULL").unwrap();
        let physical_expr = planner.create_physical_expr(&expr).unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).unwrap().as_ref(),
            &BooleanArray::from(vec![
                true, false, false, true, false, false, true, false, false, true,
            ])
        );
    }

    #[test]
    fn test_sql_invert() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Boolean, true)]));

        let planner = Planner::new(schema.clone());

        let expr = planner.parse_filter("NOT s").unwrap();
        let physical_expr = planner.create_physical_expr(&expr).unwrap();

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(BooleanArray::from_iter(
                (0..10).map(|v| Some(v % 3 == 0)),
            ))],
        )
        .unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).unwrap().as_ref(),
            &BooleanArray::from(vec![
                false, true, true, false, true, true, false, true, true, false
            ])
        );
    }

    #[test]
    fn test_sql_cast() {
        let cases = &[
            (
                "x = cast('2021-01-01 00:00:00' as timestamp)",
                ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
            ),
            (
                "x = cast('2021-01-01 00:00:00' as timestamp(0))",
                ArrowDataType::Timestamp(TimeUnit::Second, None),
            ),
            (
                "x = cast('2021-01-01 00:00:00.123' as timestamp(9))",
                ArrowDataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            (
                "x = cast('2021-01-01 00:00:00.123' as datetime(9))",
                ArrowDataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            ("x = cast('2021-01-01' as date)", ArrowDataType::Date32),
            (
                "x = cast('1.238' as decimal(9,3))",
                ArrowDataType::Decimal128(9, 3),
            ),
            ("x = cast(1 as float)", ArrowDataType::Float32),
            ("x = cast(1 as double)", ArrowDataType::Float64),
            ("x = cast(1 as tinyint)", ArrowDataType::Int8),
            ("x = cast(1 as smallint)", ArrowDataType::Int16),
            ("x = cast(1 as int)", ArrowDataType::Int32),
            ("x = cast(1 as integer)", ArrowDataType::Int32),
            ("x = cast(1 as bigint)", ArrowDataType::Int64),
            ("x = cast(1 as tinyint unsigned)", ArrowDataType::UInt8),
            ("x = cast(1 as smallint unsigned)", ArrowDataType::UInt16),
            ("x = cast(1 as int unsigned)", ArrowDataType::UInt32),
            ("x = cast(1 as integer unsigned)", ArrowDataType::UInt32),
            ("x = cast(1 as bigint unsigned)", ArrowDataType::UInt64),
            ("x = cast(1 as boolean)", ArrowDataType::Boolean),
            ("x = cast(1 as string)", ArrowDataType::Utf8),
        ];

        for (sql, expected_data_type) in cases {
            let schema = Arc::new(Schema::new(vec![Field::new(
                "x",
                expected_data_type.clone(),
                true,
            )]));
            let planner = Planner::new(schema.clone());
            let expr = planner.parse_filter(sql).unwrap();

            // Get the thing after 'cast(` but before ' as'.
            let expected_value_str = sql
                .split("cast(")
                .nth(1)
                .unwrap()
                .split(" as")
                .next()
                .unwrap();
            // Remove any quote marks
            let expected_value_str = expected_value_str.trim_matches('\'');

            match expr {
                Expr::BinaryExpr(BinaryExpr { right, .. }) => match right.as_ref() {
                    Expr::Cast(Cast { expr, data_type }) => {
                        match expr.as_ref() {
                            Expr::Literal(ScalarValue::Utf8(Some(value_str))) => {
                                assert_eq!(value_str, expected_value_str);
                            }
                            Expr::Literal(ScalarValue::Int64(Some(value))) => {
                                assert_eq!(*value, 1);
                            }
                            _ => panic!("Expected cast to be applied to literal"),
                        }
                        assert_eq!(data_type, expected_data_type);
                    }
                    _ => panic!("Expected right to be a cast"),
                },
                _ => panic!("Expected binary expression"),
            }
        }
    }

    #[test]
    fn test_sql_literals() {
        let cases = &[
            (
                "x = timestamp '2021-01-01 00:00:00'",
                ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
            ),
            (
                "x = timestamp(0) '2021-01-01 00:00:00'",
                ArrowDataType::Timestamp(TimeUnit::Second, None),
            ),
            (
                "x = timestamp(9) '2021-01-01 00:00:00.123'",
                ArrowDataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            ("x = date '2021-01-01'", ArrowDataType::Date32),
            ("x = decimal(9,3) '1.238'", ArrowDataType::Decimal128(9, 3)),
        ];

        for (sql, expected_data_type) in cases {
            let schema = Arc::new(Schema::new(vec![Field::new(
                "x",
                expected_data_type.clone(),
                true,
            )]));
            let planner = Planner::new(schema.clone());
            let expr = planner.parse_filter(sql).unwrap();

            let expected_value_str = sql.split('\'').nth(1).unwrap();

            match expr {
                Expr::BinaryExpr(BinaryExpr { right, .. }) => match right.as_ref() {
                    Expr::Cast(Cast { expr, data_type }) => {
                        match expr.as_ref() {
                            Expr::Literal(ScalarValue::Utf8(Some(value_str))) => {
                                assert_eq!(value_str, expected_value_str);
                            }
                            _ => panic!("Expected cast to be applied to literal"),
                        }
                        assert_eq!(data_type, expected_data_type);
                    }
                    _ => panic!("Expected right to be a cast"),
                },
                _ => panic!("Expected binary expression"),
            }
        }
    }

    #[test]
    fn test_sql_array_literals() {
        let cases = [
            (
                "x = [1, 2, 3]",
                ArrowDataType::List(Arc::new(Field::new("item", ArrowDataType::Int64, true))),
            ),
            (
                "x = [1, 2, 3]",
                ArrowDataType::FixedSizeList(
                    Arc::new(Field::new("item", ArrowDataType::Int64, true)),
                    3,
                ),
            ),
        ];

        for (sql, expected_data_type) in cases {
            let schema = Arc::new(Schema::new(vec![Field::new(
                "x",
                expected_data_type.clone(),
                true,
            )]));
            let planner = Planner::new(schema.clone());
            let expr = planner.parse_filter(sql).unwrap();
            let expr = planner.optimize_expr(expr).unwrap();

            match expr {
                Expr::BinaryExpr(BinaryExpr { right, .. }) => match right.as_ref() {
                    Expr::Literal(value) => {
                        assert_eq!(&value.data_type(), &expected_data_type);
                    }
                    _ => panic!("Expected right to be a literal"),
                },
                _ => panic!("Expected binary expression"),
            }
        }
    }

    #[test]
    fn test_sql_comparison() {
        // Create a batch with all data types
        let batch: Vec<(&str, ArrayRef)> = vec![
            (
                "timestamp_s",
                Arc::new(TimestampSecondArray::from_iter_values(0..10)),
            ),
            (
                "timestamp_ms",
                Arc::new(TimestampMillisecondArray::from_iter_values(0..10)),
            ),
            (
                "timestamp_us",
                Arc::new(TimestampMicrosecondArray::from_iter_values(0..10)),
            ),
            (
                "timestamp_ns",
                Arc::new(TimestampNanosecondArray::from_iter_values(4995..5005)),
            ),
        ];
        let batch = RecordBatch::try_from_iter(batch).unwrap();

        let planner = Planner::new(batch.schema());

        // Each expression is meant to select the final 5 rows
        let expressions = &[
            "timestamp_s >= TIMESTAMP '1970-01-01 00:00:05'",
            "timestamp_ms >= TIMESTAMP '1970-01-01 00:00:00.005'",
            "timestamp_us >= TIMESTAMP '1970-01-01 00:00:00.000005'",
            "timestamp_ns >= TIMESTAMP '1970-01-01 00:00:00.000005'",
        ];

        let expected: ArrayRef = Arc::new(BooleanArray::from_iter(
            std::iter::repeat(Some(false))
                .take(5)
                .chain(std::iter::repeat(Some(true)).take(5)),
        ));
        for expression in expressions {
            // convert to physical expression
            let logical_expr = planner.parse_filter(expression).unwrap();
            let logical_expr = planner.optimize_expr(logical_expr).unwrap();
            let physical_expr = planner.create_physical_expr(&logical_expr).unwrap();

            // Evaluate and assert they have correct results
            let result = physical_expr.evaluate(&batch).unwrap();
            let result = result.into_array(batch.num_rows()).unwrap();
            assert_eq!(&expected, &result, "unexpected result for {}", expression);
        }
    }

    #[test]
    fn test_columns_in_expr() {
        let expr = col("s0").gt(lit("value")).and(
            col("st")
                .field("st")
                .field("s2")
                .eq(lit("value"))
                .or(col("st")
                    .field("s1")
                    .in_list(vec![lit("value 1"), lit("value 2")], false)),
        );

        let columns = Planner::column_names_in_expr(&expr);
        assert_eq!(columns, vec!["s0", "st.s1", "st.st.s2"]);
    }
}
