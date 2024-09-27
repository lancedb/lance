// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_schema::Schema as ArrowSchema;
use bytes::Bytes;
use datafusion_common::ScalarValue;
use datafusion_expr::Expr;
use futures::FutureExt;
use lance_core::datatypes::Schema;
use lance_core::Result;
use lance_datafusion::substrait::encode_substrait;
use lance_datafusion::substrait::parse_substrait;
use lance_encoding::decoder::FilterExpression;

/// Helper trait to bridge lance-encoding and substrait
pub trait FilterExpressionExt {
    /// Convert a lance-encoding filter expression (which we assume is
    /// substrait encoded) into a datafusion expr
    fn substrait_to_df(&self, schema: Arc<ArrowSchema>) -> Result<Expr>;
    /// Convert a datafusion filter expression into a lance-encoding
    /// filter expression (using substrait)
    fn df_to_substrait(expr: Expr, schema: &Schema) -> Result<Self>
    where
        Self: Sized;
}

impl FilterExpressionExt for FilterExpression {
    fn substrait_to_df(&self, schema: Arc<ArrowSchema>) -> Result<Expr> {
        if self.0.is_empty() {
            return Ok(Expr::Literal(ScalarValue::Boolean(Some(true))));
        }
        let expr = parse_substrait(&self.0, schema).now_or_never().unwrap()?;
        Ok(expr)
    }

    fn df_to_substrait(expr: Expr, schema: &Schema) -> Result<Self>
    where
        Self: Sized,
    {
        let schema = Arc::new(ArrowSchema::from(schema));
        let bytes = Bytes::from(encode_substrait(expr, schema)?);
        Ok(Self(bytes))
    }
}
