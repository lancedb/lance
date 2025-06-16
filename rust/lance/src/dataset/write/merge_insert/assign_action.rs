// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use super::{MergeInsertParams, WhenNotMatchedBySource};
use crate::{dataset::WhenMatched, error::Result};
use datafusion::scalar::ScalarValue;
use datafusion_expr::{
    col, expr::ScalarFunction, sqlparser::keywords::ROWID, Case, Expr, ScalarUDF,
};
use datafusion_functions::core::named_struct::NamedStructFunc;

// Note: right now, this is a fixed enum. In the future, this will need to be
// dynamic to support multiple merge insert update clauses like:
// ```sql
// MERGE my_table USING input ON table.id = input.id
// WHEN MATCHED AND input.event = "new_date" THEN UPDATE SET my_table.date = input.date
// WHEN MATCHED AND input.event = "new_name" THEN UPDATE SET my_table.name = input.new_name
// ```
// At that point we will have a variable number of actions.
#[derive(Clone, Copy)]
#[repr(u8)]
enum Action {
    Nothing = 0,
    /// Update all columns with source values
    UpdateAll = 1,
    Insert = 2,
    Delete = 3,
}

impl Action {
    fn as_literal_expr(&self) -> Expr {
        Expr::Literal(ScalarValue::UInt8(Some(*self as u8)))
    }
}

/// Transforms merge insert parameters into a logical expression. The output
/// is a single "action" column, that describes what to do with each row.
pub fn merge_insert_action(params: &MergeInsertParams) -> Result<Expr> {
    let source_has_key: Expr = datafusion_expr::lit(true); // TODO: Make sure at least one key column is non-null

    let row_id_is_not_null = col("target._rowid").is_not_null();
    let matched = source_has_key.clone().and(row_id_is_not_null);

    let row_id_is_null = col("target._rowid").is_null();
    let not_matched_in_target = source_has_key.and(row_id_is_null);

    let not_matched_in_source = col("target._rowid").is_null().is_not_true();

    let mut cases = vec![];

    if params.insert_not_matched {
        cases.push((not_matched_in_target, Action::Insert.as_literal_expr()));
    }

    match &params.when_matched {
        WhenMatched::UpdateAll => {
            cases.push((matched, Action::UpdateAll.as_literal_expr()));
        }
        WhenMatched::UpdateIf(condition) => {
            cases.push((matched.and(condition.clone()), Action::UpdateAll.as_literal_expr()));
        }
        WhenMatched::DoNothing => {}
    }

    match &params.delete_not_matched_by_source {
        WhenNotMatchedBySource::Delete => {
            cases.push((not_matched_in_source, Action::Delete.as_literal_expr()));
        }
        WhenNotMatchedBySource::DeleteIf(condition) => {
            cases.push((
                not_matched_in_source.and(condition.clone()),
                Action::Delete.as_literal_expr(),
            ));
        }
        WhenNotMatchedBySource::Keep => {}
    }

    Ok(Expr::Case(Case {
        expr: None,
        when_then_expr: cases
            .into_iter()
            .map(|(when, then)| (Box::new(when), Box::new(then)))
            .collect(),
        else_expr: Some(Box::new(Action::Nothing.as_literal_expr())),
    }))
}

// Collect given columns into a struct
///
/// This is the inverse of [datafusion_expr::logical_plan::builder::get_struct_unnested_columns]
pub fn collect_to_struct(columns: &[&str], prefix: Option<&str>) -> Expr {
    let mut args = Vec::with_capacity(columns.len() * 2);
    for column in columns {
        let prefix_len = prefix.map(|p| p.len()).unwrap_or(0);
        args.push(Expr::Literal(ScalarValue::Utf8(Some(
            column[prefix_len..column.len()].to_string(),
        ))));
        args.push(col(*column));
    }

    Expr::ScalarFunction(ScalarFunction {
        func: Arc::new(ScalarUDF::new_from_impl(NamedStructFunc::new())),
        args,
    })
}
