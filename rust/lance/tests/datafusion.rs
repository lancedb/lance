// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance::deps::datafusion::{
    logical_expr::{Expr, col, lit},
    sql::unparser::Unparser,
};

#[test]
fn test_deeply_nested_datafusion_expression_does_not_overflow_stack() {
    const DEPTH: usize = 2_000;

    let handle = std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(|| {
            let mut expr: Expr = col("value");
            for _ in 0..DEPTH {
                expr = expr + lit(1_i64);
            }

            Unparser::default()
                .expr_to_sql(&expr)
                .expect("deeply nested expression should unparse");
            Unparser::default()
                .with_pretty(true)
                .expr_to_sql(&expr)
                .expect("deeply nested expression should unparse in pretty mode");
        })
        .expect("unparsing thread should spawn");

    handle.join().expect("unparsing thread should not panic");
}
