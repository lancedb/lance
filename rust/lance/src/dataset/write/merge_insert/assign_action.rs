// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{MergeInsertParams, WhenNotMatchedBySource};
use crate::{dataset::WhenMatched, error::Result};
use datafusion::scalar::ScalarValue;
use datafusion_expr::{col, Case, Expr};
use snafu::location;

// Note: right now, this is a fixed enum. In the future, this will need to be
// dynamic to support multiple merge insert update clauses like:
// ```sql
// MERGE my_table USING input ON table.id = input.id
// WHEN MATCHED AND input.event = "new_date" THEN UPDATE SET my_table.date = input.date
// WHEN MATCHED AND input.event = "new_name" THEN UPDATE SET my_table.name = input.new_name
// ```
// At that point we will have a variable number of actions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Action {
    Nothing = 0,
    /// Update all columns with source values
    UpdateAll = 1,
    Insert = 2,
    Delete = 3,
}

impl TryFrom<u8> for Action {
    type Error = crate::Error;

    fn try_from(value: u8) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Nothing),
            1 => Ok(Self::UpdateAll),
            2 => Ok(Self::Insert),
            3 => Ok(Self::Delete),
            _ => Err(crate::Error::InvalidInput {
                source: format!("Invalid action code: {value}").into(),
                location: location!(),
            }),
        }
    }
}

impl Action {
    fn as_literal_expr(&self) -> Expr {
        Expr::Literal(ScalarValue::UInt8(Some(*self as u8)))
    }
}

/// Transforms merge insert parameters into a logical expression. The output
/// is a single "action" column, that describes what to do with each row.
pub fn merge_insert_action(params: &MergeInsertParams) -> Result<Expr> {
    // Check that at least one key column is non-null in the source
    // This ensures we only process rows that have valid join keys
    let source_has_key: Expr = if params.on.len() == 1 {
        // Single key column case - check if the source key column is not null
        // Need to qualify the column to avoid ambiguity between target.key and source.key
        col(format!("source.{}", &params.on[0])).is_not_null()
    } else {
        // Multiple key columns - require that ALL key columns are non-null
        // This is a stricter requirement than "at least one" to ensure proper joins
        let key_conditions: Vec<Expr> = params
            .on
            .iter()
            .map(|key| col(format!("source.{key}")).is_not_null())
            .collect();

        // Use AND to combine all key column checks (all must be non-null)
        key_conditions
            .into_iter()
            .reduce(|acc, expr| acc.and(expr))
            .unwrap_or_else(|| datafusion_expr::lit(false))
    };

    let row_addr_is_not_null = col("target._rowaddr").is_not_null();
    let matched = source_has_key.clone().and(row_addr_is_not_null);

    let row_addr_is_null = col("target._rowaddr").is_null();
    let not_matched_in_target = source_has_key.and(row_addr_is_null);

    let not_matched_in_source = col("target._rowaddr").is_null().is_not_true();

    let mut cases = vec![];

    if params.insert_not_matched {
        cases.push((not_matched_in_target, Action::Insert.as_literal_expr()));
    }

    match &params.when_matched {
        WhenMatched::UpdateAll => {
            cases.push((matched, Action::UpdateAll.as_literal_expr()));
        }
        WhenMatched::UpdateIf(condition) => {
            cases.push((
                matched.and(condition.as_ref().clone()),
                Action::UpdateAll.as_literal_expr(),
            ));
        }
        WhenMatched::DoNothing => {}
    }

    match &params.delete_not_matched_by_source {
        WhenNotMatchedBySource::Delete => {
            cases.push((not_matched_in_source, Action::Delete.as_literal_expr()));
        }
        WhenNotMatchedBySource::DeleteIf(condition) => {
            cases.push((
                not_matched_in_source.and(condition.as_ref().clone()),
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
