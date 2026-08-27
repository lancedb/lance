// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Rewrites of comparisons against a floating point zero literal.

use datafusion::logical_expr::{BinaryExpr, Operator, expr::InList};
use datafusion::prelude::Expr;
use datafusion::scalar::ScalarValue::{self, Float16, Float32, Float64};
use datafusion_common::tree_node::{Transformed, TreeNode};
use half::f16;
use lance_core::Result;

/// Rewrite every comparison against a floating point zero literal into the form
/// that Arrow's total-order kernels answer the way IEEE 754 and SQL define it.
///
/// Arrow sorts `-0.0` strictly below `+0.0` and compares the two encodings for
/// equality by bit pattern, while IEEE 754 and SQL treat them as one number.
/// Each comparison against a zero literal has an equivalent total-order form,
/// reached by moving the literal to whichever encoding sits on the correct side
/// of the pair:
///
/// | written              | evaluated              |
/// |----------------------|------------------------|
/// | `x < 0`, `x >= 0`    | literal becomes `-0.0` |
/// | `x <= 0`, `x > 0`    | literal becomes `+0.0` |
/// | `x = 0`              | `x IN (-0.0, 0.0)`     |
/// | `x != 0`             | `x NOT IN (-0.0, 0.0)` |
///
/// The scalar indices order candidates by the same total order, so a rewritten
/// predicate is served exactly by an index search and needs no recheck.
///
/// NaN is out of scope. Arrow sorts it above every other value, so `x >= -0.0`
/// admits NaN where IEEE would not, and that holds for every comparison rather
/// than only the ones against zero.
pub fn rewrite_signed_zero_comparisons(expr: Expr) -> Result<Expr> {
    Ok(expr
        .transform_up(|node| {
            Ok(match rewrite_node(&node) {
                Some(rewritten) => Transformed::yes(rewritten),
                None => Transformed::no(node),
            })
        })?
        .data)
}

/// Both encodings of a floating point zero, negative first.
///
/// Returns `None` for anything else, including NULL, NaN, and integer zero.
fn zero_encodings(value: &ScalarValue) -> Option<(ScalarValue, ScalarValue)> {
    match value {
        Float16(Some(v)) if *v == f16::ZERO => {
            Some((Float16(Some(f16::NEG_ZERO)), Float16(Some(f16::ZERO))))
        }
        Float32(Some(v)) if *v == 0.0 => Some((Float32(Some(-0.0)), Float32(Some(0.0)))),
        Float64(Some(v)) if *v == 0.0 => Some((Float64(Some(-0.0)), Float64(Some(0.0)))),
        _ => None,
    }
}

/// The rewritten expression, or `None` when `expr` is not a comparison against a
/// floating point zero.
fn rewrite_node(expr: &Expr) -> Option<Expr> {
    match expr {
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            // `resolve_expr` accepts the literal on either side, and the
            // operator mirrors when it sits on the left.
            let (literal, other, op) = match (left.as_ref(), right.as_ref()) {
                (_, Expr::Literal(..)) => (right.as_ref(), left.as_ref(), *op),
                (Expr::Literal(..), _) => (left.as_ref(), right.as_ref(), op.swap()?),
                _ => return None,
            };
            let Expr::Literal(value, metadata) = literal else {
                return None;
            };
            let (negative, positive) = zero_encodings(value)?;
            let zero = match op {
                Operator::Lt | Operator::GtEq => negative,
                Operator::LtEq | Operator::Gt => positive,
                Operator::Eq | Operator::NotEq => {
                    return Some(Expr::InList(InList {
                        expr: Box::new(other.clone()),
                        list: vec![
                            Expr::Literal(negative, metadata.clone()),
                            Expr::Literal(positive, metadata.clone()),
                        ],
                        negated: op == Operator::NotEq,
                    }));
                }
                _ => return None,
            };
            Some(Expr::BinaryExpr(BinaryExpr {
                left: Box::new(other.clone()),
                op,
                right: Box::new(Expr::Literal(zero, metadata.clone())),
            }))
        }
        Expr::InList(InList {
            expr,
            list,
            negated,
        }) => Some(Expr::InList(InList {
            expr: expr.clone(),
            list: widen_zero_list(list)?,
            negated: *negated,
        })),
        _ => None,
    }
}

/// Add the missing encoding next to every floating point zero in an `IN` list.
///
/// Returns `None` when the list holds no zero, or already spells out both
/// encodings of each zero it holds.
fn widen_zero_list(list: &[Expr]) -> Option<Vec<Expr>> {
    let mut widened = list.to_vec();
    let original_len = widened.len();
    for item in list {
        let Expr::Literal(value, metadata) = item else {
            continue;
        };
        let Some((negative, positive)) = zero_encodings(value) else {
            continue;
        };
        let counterpart = if *value == negative {
            positive
        } else {
            negative
        };
        // `ScalarValue` compares floats by bit pattern, so this distinguishes
        // the two encodings rather than collapsing them.
        let present = widened
            .iter()
            .any(|other| matches!(other, Expr::Literal(v, _) if *v == counterpart));
        if !present {
            widened.push(Expr::Literal(counterpart, metadata.clone()));
        }
    }
    (widened.len() > original_len).then_some(widened)
}

#[cfg(test)]
mod tests {
    use datafusion::prelude::{col, lit};
    use rstest::rstest;

    use super::*;

    fn rewrite(expr: Expr) -> Expr {
        rewrite_signed_zero_comparisons(expr).unwrap()
    }

    fn compare(left: Expr, op: Operator, right: Expr) -> Expr {
        Expr::BinaryExpr(BinaryExpr {
            left: Box::new(left),
            op,
            right: Box::new(right),
        })
    }

    #[rstest]
    #[case::lt_from_positive(Operator::Lt, 0.0, -0.0)]
    #[case::lt_from_negative(Operator::Lt, -0.0, -0.0)]
    #[case::lt_eq_from_positive(Operator::LtEq, 0.0, 0.0)]
    #[case::lt_eq_from_negative(Operator::LtEq, -0.0, 0.0)]
    #[case::gt_from_positive(Operator::Gt, 0.0, 0.0)]
    #[case::gt_from_negative(Operator::Gt, -0.0, 0.0)]
    #[case::gt_eq_from_positive(Operator::GtEq, 0.0, -0.0)]
    #[case::gt_eq_from_negative(Operator::GtEq, -0.0, -0.0)]
    fn range_comparison_uses_the_encoding_for_the_operator(
        #[case] op: Operator,
        #[case] written: f64,
        #[case] evaluated: f64,
    ) {
        assert_eq!(
            rewrite(compare(col("x"), op, lit(written))),
            compare(col("x"), op, lit(evaluated))
        );
    }

    #[rstest]
    #[case::eq(Operator::Eq, false)]
    #[case::not_eq(Operator::NotEq, true)]
    fn equality_covers_both_encodings(#[case] op: Operator, #[case] negated: bool) {
        assert_eq!(
            rewrite(compare(col("x"), op, lit(0.0))),
            Expr::InList(InList {
                expr: Box::new(col("x")),
                list: vec![lit(-0.0), lit(0.0)],
                negated,
            })
        );
    }

    #[test]
    fn a_literal_on_the_left_mirrors_the_operator() {
        // `0.0 > x` is `x < 0.0`, which evaluates against the negative encoding.
        assert_eq!(
            rewrite(compare(lit(0.0), Operator::Gt, col("x"))),
            compare(col("x"), Operator::Lt, lit(-0.0))
        );
    }

    #[rstest]
    #[case::float32(Float32(Some(-0.0)), Float32(Some(0.0)))]
    #[case::float16(Float16(Some(f16::NEG_ZERO)), Float16(Some(f16::ZERO)))]
    fn narrow_floats_are_rewritten_too(
        #[case] written: ScalarValue,
        #[case] evaluated: ScalarValue,
    ) {
        assert_eq!(
            rewrite(compare(
                col("x"),
                Operator::LtEq,
                Expr::Literal(written, None)
            )),
            compare(col("x"), Operator::LtEq, Expr::Literal(evaluated, None))
        );
    }

    #[test]
    fn an_in_list_gains_the_missing_encoding() {
        assert_eq!(
            rewrite(Expr::InList(InList {
                expr: Box::new(col("x")),
                list: vec![lit(0.0), lit(5.0)],
                negated: true,
            })),
            Expr::InList(InList {
                expr: Box::new(col("x")),
                list: vec![lit(0.0), lit(5.0), lit(-0.0)],
                negated: true,
            })
        );
    }

    #[test]
    fn only_the_zero_comparison_in_a_conjunction_changes() {
        assert_eq!(
            rewrite(col("x").lt(lit(0.0)).and(col("y").eq(lit(1.0)))),
            col("x").lt(lit(-0.0)).and(col("y").eq(lit(1.0)))
        );
    }

    #[rstest]
    #[case::non_zero(col("x").lt(lit(1.0)))]
    #[case::integer_zero(col("x").eq(lit(0_i64)))]
    #[case::null(compare(col("x"), Operator::Eq, Expr::Literal(Float64(None), None)))]
    #[case::nan(col("x").lt(lit(f64::NAN)))]
    #[case::column_on_both_sides(col("x").lt(col("y")))]
    #[case::both_encodings_listed(Expr::InList(InList {
        expr: Box::new(col("x")),
        list: vec![lit(-0.0), lit(0.0)],
        negated: false,
    }))]
    fn unrelated_comparisons_are_left_alone(#[case] expr: Expr) {
        assert_eq!(rewrite(expr.clone()), expr);
    }
}
