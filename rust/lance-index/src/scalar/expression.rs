// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{ops::Bound, sync::Arc};

use arrow_schema::{DataType, Field};
use datafusion_common::ScalarValue;
use datafusion_expr::{Expr, Operator, ReturnFieldArgs, ScalarUDF, expr::Like};

use super::{BloomFilterQuery, LabelListQuery, SargableQuery, TextQuery, TokenQuery};
#[cfg(feature = "geo")]
use super::{GeoQuery, RelationQuery};
use lance_core::Result;
use lance_datafusion::{expr::safe_coerce_scalar, planner::Planner};

pub use lance_index_core::scalar::expression::{
    IndexInformationProvider, IndexedExpression, MultiQueryParser, ScalarIndexExpr,
    ScalarIndexLoader, ScalarIndexSearch, ScalarQueryParser, apply_scalar_indices,
};

/// A parser for indices that handle SARGable queries
#[derive(Debug)]
pub struct SargableQueryParser {
    index_name: String,
    index_type: String,
    needs_recheck: bool,
}

impl SargableQueryParser {
    pub fn new(index_name: String, index_type: String, needs_recheck: bool) -> Self {
        Self {
            index_name,
            index_type,
            needs_recheck,
        }
    }
}

impl ScalarQueryParser for SargableQueryParser {
    fn is_valid_reference(&self, func: &Expr, data_type: &DataType) -> Option<DataType> {
        match func {
            Expr::Column(_) => Some(data_type.clone()),
            // Also accept get_field expressions for nested field access
            Expr::ScalarFunction(udf) if udf.name() == "get_field" => Some(data_type.clone()),
            _ => None,
        }
    }

    fn visit_between(
        &self,
        column: &str,
        low: &Bound<ScalarValue>,
        high: &Bound<ScalarValue>,
    ) -> Option<IndexedExpression> {
        if let Bound::Included(val) | Bound::Excluded(val) = low
            && val.is_null()
        {
            return None;
        }
        if let Bound::Included(val) | Bound::Excluded(val) = high
            && val.is_null()
        {
            return None;
        }
        let query = SargableQuery::Range(low.clone(), high.clone());
        Some(IndexedExpression::index_query_with_recheck(
            column.to_string(),
            self.index_name.clone(),
            self.index_type.clone(),
            Arc::new(query),
            self.needs_recheck,
        ))
    }

    fn visit_in_list(&self, column: &str, in_list: &[ScalarValue]) -> Option<IndexedExpression> {
        if in_list.iter().any(|val| val.is_null()) {
            return None;
        }
        let query = SargableQuery::IsIn(in_list.to_vec());
        Some(IndexedExpression::index_query_with_recheck(
            column.to_string(),
            self.index_name.clone(),
            self.index_type.clone(),
            Arc::new(query),
            self.needs_recheck,
        ))
    }

    fn visit_is_bool(&self, column: &str, value: bool) -> Option<IndexedExpression> {
        Some(IndexedExpression::index_query_with_recheck(
            column.to_string(),
            self.index_name.clone(),
            self.index_type.clone(),
            Arc::new(SargableQuery::Equals(ScalarValue::Boolean(Some(value)))),
            self.needs_recheck,
        ))
    }

    fn visit_is_null(&self, column: &str) -> Option<IndexedExpression> {
        Some(IndexedExpression::index_query_with_recheck(
            column.to_string(),
            self.index_name.clone(),
            self.index_type.clone(),
            Arc::new(SargableQuery::IsNull()),
            self.needs_recheck,
        ))
    }

    fn visit_comparison(
        &self,
        column: &str,
        value: &ScalarValue,
        op: &Operator,
    ) -> Option<IndexedExpression> {
        if value.is_null() {
            return None;
        }
        let query = match op {
            Operator::Lt => SargableQuery::Range(Bound::Unbounded, Bound::Excluded(value.clone())),
            Operator::LtEq => {
                SargableQuery::Range(Bound::Unbounded, Bound::Included(value.clone()))
            }
            Operator::Gt => SargableQuery::Range(Bound::Excluded(value.clone()), Bound::Unbounded),
            Operator::GtEq => {
                SargableQuery::Range(Bound::Included(value.clone()), Bound::Unbounded)
            }
            Operator::Eq => SargableQuery::Equals(value.clone()),
            // This will be negated by the caller
            Operator::NotEq => SargableQuery::Equals(value.clone()),
            _ => unreachable!(),
        };
        Some(IndexedExpression::index_query_with_recheck(
            column.to_string(),
            self.index_name.clone(),
            self.index_type.clone(),
            Arc::new(query),
            self.needs_recheck,
        ))
    }

    fn visit_scalar_function(
        &self,
        column: &str,
        _data_type: &DataType,
        func: &ScalarUDF,
        args: &[Expr],
    ) -> Option<IndexedExpression> {
        // Handle starts_with(col, 'prefix') -> convert to LikePrefix query
        if func.name() == "starts_with" && args.len() == 2 {
            // Extract the prefix from the second argument
            let prefix = match &args[1] {
                Expr::Literal(ScalarValue::Utf8(Some(s)), _) => ScalarValue::Utf8(Some(s.clone())),
                Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _) => {
                    ScalarValue::LargeUtf8(Some(s.clone()))
                }
                _ => return None,
            };

            let query = SargableQuery::LikePrefix(prefix);
            return Some(IndexedExpression::index_query_with_recheck(
                column.to_string(),
                self.index_name.clone(),
                self.index_type.clone(),
                Arc::new(query),
                self.needs_recheck,
            ));
        }

        None
    }

    fn visit_like(
        &self,
        column: &str,
        like: &Like,
        pattern: &ScalarValue,
    ) -> Option<IndexedExpression> {
        // Case-insensitive LIKE (ILIKE) cannot be efficiently pruned with zone maps
        if like.case_insensitive {
            return None;
        }

        // Extract the pattern string
        let pattern_str = match pattern {
            ScalarValue::Utf8(Some(s)) => s.as_str(),
            ScalarValue::LargeUtf8(Some(s)) => s.as_str(),
            _ => return None,
        };

        // Try to extract a prefix from the LIKE pattern
        let (prefix, needs_refine) = extract_like_leading_prefix(pattern_str, like.escape_char)?;

        // Create the prefix ScalarValue with the same type as the pattern
        let prefix_value = match pattern {
            ScalarValue::Utf8(_) => ScalarValue::Utf8(Some(prefix)),
            ScalarValue::LargeUtf8(_) => ScalarValue::LargeUtf8(Some(prefix)),
            _ => return None,
        };

        let query = SargableQuery::LikePrefix(prefix_value);
        let scalar_query = Some(ScalarIndexExpr::Query(ScalarIndexSearch {
            column: column.to_string(),
            index_name: self.index_name.clone(),
            index_type: self.index_type.clone(),
            query: Arc::new(query),
            needs_recheck: self.needs_recheck,
            fragment_bitmap: None,
        }));

        // If the pattern has wildcards beyond simple prefix, add refine expression
        let refine_expr = if needs_refine {
            Some(Expr::Like(like.clone()))
        } else {
            None
        };

        Some(IndexedExpression {
            scalar_query,
            refine_expr,
        })
    }
}

/// Extract the leading literal prefix from a LIKE pattern.
///
/// Returns `Some((prefix, needs_refine))` where:
/// - `prefix` is the leading literal portion before any wildcards
/// - `needs_refine` is true if the pattern has wildcards beyond a simple trailing `%`
///
/// Returns `None` if the pattern starts with a wildcard (no leading literal).
///
/// Examples:
/// - "foo%" -> Some(("foo", false)) - pure prefix, no recheck needed
/// - "foo%bar%" -> Some(("foo", true)) - can use prefix for pruning, needs recheck
/// - "foo_bar%" -> Some(("foo", true)) - _ is a wildcard, needs recheck
/// - "foo\%bar%" with escape '\' -> Some(("foo%bar", false)) - escaped %, pure prefix
/// - "%foo" -> None - starts with wildcard, cannot prune
/// - "foo" -> None - no wildcard at all, use equality instead
fn extract_like_leading_prefix(pattern: &str, escape_char: Option<char>) -> Option<(String, bool)> {
    let chars: Vec<char> = pattern.chars().collect();
    let len = chars.len();

    if len == 0 {
        return None;
    }

    // DataFusion's starts_with simplification escapes special characters with backslash
    // but doesn't set escape_char. Use backslash as default escape character.
    // Pattern: starts_with(col, 'test_ns$') -> col LIKE 'test\_ns$%' (escape_char: None)
    // See: https://github.com/apache/datafusion/issues/XXXX
    let effective_escape_char = escape_char.or(Some('\\'));

    // Helper to check if a character at position i is escaped
    let is_escaped = |i: usize| -> bool {
        if let Some(esc) = effective_escape_char {
            if i > 0 && chars[i - 1] == esc {
                // Check if the escape char itself is escaped
                if i >= 2 && chars[i - 2] == esc {
                    false // Escape was escaped, so this char is NOT escaped
                } else {
                    true // This char is escaped
                }
            } else {
                false
            }
        } else {
            // No escape character defined - nothing can be escaped
            false
        }
    };

    // Pattern must contain at least one unescaped wildcard
    let has_wildcard = chars.iter().enumerate().any(|(i, &c)| {
        if c != '%' && c != '_' {
            return false;
        }
        !is_escaped(i)
    });

    if !has_wildcard {
        return None; // No wildcards, should use equality
    }

    // Check if pattern starts with an unescaped wildcard
    if chars[0] == '%' || chars[0] == '_' {
        return None; // Starts with wildcard, cannot prune
    }

    // Extract the leading literal prefix (everything before first unescaped wildcard)
    let mut prefix = String::new();
    let mut i = 0;
    let mut found_wildcard = false;

    while i < len {
        let c = chars[i];

        // Check for escape character (using effective escape char which may be inferred)
        if let Some(esc) = effective_escape_char
            && c == esc
            && i + 1 < len
        {
            let next = chars[i + 1];
            if next == '%' || next == '_' || next == esc {
                // Escaped character - add the literal character
                prefix.push(next);
                i += 2;
                continue;
            }
        }

        // Check for unescaped wildcard
        if c == '%' || c == '_' {
            found_wildcard = true;
            break;
        }

        prefix.push(c);
        i += 1;
    }

    if prefix.is_empty() {
        return None;
    }

    // Check if pattern is just a simple prefix (ends with single % and nothing after)
    let needs_refine = if found_wildcard && i < len {
        // Check if we're at a % wildcard
        if chars[i] == '%' && i + 1 == len {
            // Pattern is "prefix%" - pure prefix match, no refine needed
            false
        } else {
            // Pattern has more after first wildcard, or has _ wildcard
            true
        }
    } else {
        // No wildcard found (shouldn't happen due to earlier check)
        false
    };

    Some((prefix, needs_refine))
}

/// A parser for bloom filter indices that only support equals, is_null, and is_in operations
#[derive(Debug)]
pub struct BloomFilterQueryParser {
    index_name: String,
    index_type: String,
    needs_recheck: bool,
}

impl BloomFilterQueryParser {
    pub fn new(index_name: String, index_type: String, needs_recheck: bool) -> Self {
        Self {
            index_name,
            index_type,
            needs_recheck,
        }
    }
}

impl ScalarQueryParser for BloomFilterQueryParser {
    fn visit_between(
        &self,
        _: &str,
        _: &Bound<ScalarValue>,
        _: &Bound<ScalarValue>,
    ) -> Option<IndexedExpression> {
        // Bloom filters don't support range queries
        None
    }

    fn visit_in_list(&self, column: &str, in_list: &[ScalarValue]) -> Option<IndexedExpression> {
        let query = BloomFilterQuery::IsIn(in_list.to_vec());
        Some(IndexedExpression::index_query_with_recheck(
            column.to_string(),
            self.index_name.clone(),
            self.index_type.clone(),
            Arc::new(query),
            self.needs_recheck,
        ))
    }

    fn visit_is_bool(&self, column: &str, value: bool) -> Option<IndexedExpression> {
        Some(IndexedExpression::index_query_with_recheck(
            column.to_string(),
            self.index_name.clone(),
            self.index_type.clone(),
            Arc::new(BloomFilterQuery::Equals(ScalarValue::Boolean(Some(value)))),
            self.needs_recheck,
        ))
    }

    fn visit_is_null(&self, column: &str) -> Option<IndexedExpression> {
        Some(IndexedExpression::index_query_with_recheck(
            column.to_string(),
            self.index_name.clone(),
            self.index_type.clone(),
            Arc::new(BloomFilterQuery::IsNull()),
            self.needs_recheck,
        ))
    }

    fn visit_comparison(
        &self,
        column: &str,
        value: &ScalarValue,
        op: &Operator,
    ) -> Option<IndexedExpression> {
        let query = match op {
            // Bloom filters only support equality comparisons
            Operator::Eq => BloomFilterQuery::Equals(value.clone()),
            // This will be negated by the caller
            Operator::NotEq => BloomFilterQuery::Equals(value.clone()),
            // Bloom filters don't support range operations
            _ => return None,
        };
        Some(IndexedExpression::index_query_with_recheck(
            column.to_string(),
            self.index_name.clone(),
            self.index_type.clone(),
            Arc::new(query),
            self.needs_recheck,
        ))
    }

    fn visit_scalar_function(
        &self,
        _: &str,
        _: &DataType,
        _: &ScalarUDF,
        _: &[Expr],
    ) -> Option<IndexedExpression> {
        // Bloom filters don't support scalar functions
        None
    }
}

/// A parser for indices that handle label list queries
#[derive(Debug)]
pub struct LabelListQueryParser {
    index_name: String,
    index_type: String,
}

impl LabelListQueryParser {
    pub fn new(index_name: String, index_type: String) -> Self {
        Self {
            index_name,
            index_type,
        }
    }
}

impl ScalarQueryParser for LabelListQueryParser {
    fn visit_between(
        &self,
        _: &str,
        _: &Bound<ScalarValue>,
        _: &Bound<ScalarValue>,
    ) -> Option<IndexedExpression> {
        None
    }

    fn visit_in_list(&self, _: &str, _: &[ScalarValue]) -> Option<IndexedExpression> {
        None
    }

    fn visit_is_bool(&self, _: &str, _: bool) -> Option<IndexedExpression> {
        None
    }

    fn visit_is_null(&self, _: &str) -> Option<IndexedExpression> {
        None
    }

    fn visit_comparison(
        &self,
        _: &str,
        _: &ScalarValue,
        _: &Operator,
    ) -> Option<IndexedExpression> {
        None
    }

    fn visit_scalar_function(
        &self,
        column: &str,
        data_type: &DataType,
        func: &ScalarUDF,
        args: &[Expr],
    ) -> Option<IndexedExpression> {
        if args.len() != 2 {
            return None;
        }
        // DataFusion normalizes array_contains to array_has
        if func.name() == "array_has" {
            let inner_type = match data_type {
                DataType::List(field) | DataType::LargeList(field) => field.data_type(),
                _ => return None,
            };
            let scalar = maybe_scalar(&args[1], inner_type)?;
            // array_has(..., NULL) returns no matches in datafusion, but the index would
            // match rows containing NULL. Fallback to match datafusion behavior.
            if scalar.is_null() {
                return None;
            }
            let query = LabelListQuery::HasAnyLabel(vec![scalar]);
            return Some(IndexedExpression::index_query(
                column.to_string(),
                self.index_name.clone(),
                self.index_type.clone(),
                Arc::new(query),
            ));
        }

        let label_list = maybe_scalar(&args[1], data_type)?;
        if let ScalarValue::List(list_arr) = label_list {
            let list_values = list_arr.values();
            if list_values.is_empty() {
                return None;
            }
            let mut scalars = Vec::with_capacity(list_values.len());
            for idx in 0..list_values.len() {
                scalars.push(ScalarValue::try_from_array(list_values.as_ref(), idx).ok()?);
            }
            if func.name() == "array_has_all" {
                let query = LabelListQuery::HasAllLabels(scalars);
                Some(IndexedExpression::index_query(
                    column.to_string(),
                    self.index_name.clone(),
                    self.index_type.clone(),
                    Arc::new(query),
                ))
            } else if func.name() == "array_has_any" {
                let query = LabelListQuery::HasAnyLabel(scalars);
                Some(IndexedExpression::index_query(
                    column.to_string(),
                    self.index_name.clone(),
                    self.index_type.clone(),
                    Arc::new(query),
                ))
            } else {
                None
            }
        } else {
            None
        }
    }
}

/// A parser for indices that handle string `contains` queries, and -- when
/// `supports_regex` is set -- `regexp_like` / `regexp_match` queries.
#[derive(Debug, Clone)]
pub struct TextQueryParser {
    index_name: String,
    index_type: String,
    needs_recheck: bool,
    supports_regex: bool,
}

impl TextQueryParser {
    pub fn new(
        index_name: String,
        index_type: String,
        needs_recheck: bool,
        supports_regex: bool,
    ) -> Self {
        Self {
            index_name,
            index_type,
            needs_recheck,
            supports_regex,
        }
    }
}

impl ScalarQueryParser for TextQueryParser {
    fn visit_between(
        &self,
        _: &str,
        _: &Bound<ScalarValue>,
        _: &Bound<ScalarValue>,
    ) -> Option<IndexedExpression> {
        None
    }

    fn visit_in_list(&self, _: &str, _: &[ScalarValue]) -> Option<IndexedExpression> {
        None
    }

    fn visit_is_bool(&self, _: &str, _: bool) -> Option<IndexedExpression> {
        None
    }

    fn visit_is_null(&self, _: &str) -> Option<IndexedExpression> {
        None
    }

    fn visit_comparison(
        &self,
        _: &str,
        _: &ScalarValue,
        _: &Operator,
    ) -> Option<IndexedExpression> {
        None
    }

    fn visit_scalar_function(
        &self,
        column: &str,
        data_type: &DataType,
        func: &ScalarUDF,
        args: &[Expr],
    ) -> Option<IndexedExpression> {
        // The first argument is the indexed column; the second is the substring
        // / pattern. `contains` takes exactly two arguments; the regex functions
        // optionally take a third flags argument.
        if args.len() < 2 {
            return None;
        }
        // A non-string pattern cannot be handled.
        let (ScalarValue::Utf8(Some(pattern)) | ScalarValue::LargeUtf8(Some(pattern))) =
            maybe_scalar(&args[1], data_type)?
        else {
            return None;
        };

        let query = match func.name() {
            "contains" if args.len() == 2 => TextQuery::StringContains(pattern),
            "regexp_like" | "regexp_match" if self.supports_regex => {
                let pattern = match args.get(2) {
                    Some(flags_expr) => apply_regex_flags(&pattern, flags_expr)?,
                    None => pattern,
                };
                // If the pattern yields no usable trigram (e.g. `a.b`), leave it
                // to a full scan instead of routing it to the index, which could
                // only answer with an unsupported "recheck everything" result.
                if !crate::scalar::ngram::regex_can_use_index(&pattern) {
                    return None;
                }
                TextQuery::Regex(pattern)
            }
            _ => return None,
        };

        Some(IndexedExpression::index_query_with_recheck(
            column.to_string(),
            self.index_name.clone(),
            self.index_type.clone(),
            Arc::new(query),
            self.needs_recheck,
        ))
    }

    fn visit_like(
        &self,
        column: &str,
        like: &Like,
        pattern: &ScalarValue,
    ) -> Option<IndexedExpression> {
        // Infix LIKE is accelerated only by the ngram index (via its regex
        // machinery). A plain-literal `regexp_like(col, 'foo')` is rewritten to
        // `col LIKE '%foo%'` before it reaches the index, so this is the path
        // that accelerates those. ILIKE is skipped because its case folding does
        // not match the index's normalization.
        if !self.supports_regex || like.case_insensitive {
            return None;
        }
        let pattern_str = match pattern {
            ScalarValue::Utf8(Some(s)) | ScalarValue::LargeUtf8(Some(s)) => s.as_str(),
            _ => return None,
        };
        // Translate the LIKE pattern into a loose regex used only for candidate
        // generation; the original LIKE stays as the recheck filter, so the
        // regex only needs to be a sound superset.
        let regex = like_to_regex(pattern_str, like.escape_char)?;
        if !crate::scalar::ngram::regex_can_use_index(&regex) {
            return None;
        }
        Some(IndexedExpression {
            scalar_query: Some(ScalarIndexExpr::Query(ScalarIndexSearch {
                column: column.to_string(),
                index_name: self.index_name.clone(),
                index_type: self.index_type.clone(),
                query: Arc::new(TextQuery::Regex(regex)),
                needs_recheck: self.needs_recheck,
                fragment_bitmap: None,
            })),
            refine_expr: Some(Expr::Like(like.clone())),
        })
    }
}

/// Translate a LIKE pattern into a regular expression used purely for ngram
/// candidate generation: `%` becomes `.*`, `_` becomes `.`, and literal
/// characters are regex-escaped. Returns `None` when no literal run is long
/// enough to yield a trigram (the index could not help, so a full scan is left
/// to handle it).
fn like_to_regex(pattern: &str, escape: Option<char>) -> Option<String> {
    let mut regex = String::new();
    let mut run = 0usize;
    let mut longest_run = 0usize;
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        let literal = if Some(c) == escape {
            // The next character is escaped, i.e. a literal.
            chars.next()
        } else {
            match c {
                '%' => {
                    regex.push_str(".*");
                    run = 0;
                    None
                }
                '_' => {
                    regex.push('.');
                    run = 0;
                    None
                }
                other => Some(other),
            }
        };
        if let Some(lit) = literal {
            if regex_syntax::is_meta_character(lit) {
                regex.push('\\');
            }
            regex.push(lit);
            // Only runs of alphanumeric characters can produce a trigram.
            if lit.is_alphanumeric() {
                run += 1;
                longest_run = longest_run.max(run);
            } else {
                run = 0;
            }
        }
    }
    (longest_run >= 3).then_some(regex)
}

/// Fold the supported `regexp_like` / `regexp_match` flags into an inline prefix
/// on the pattern (e.g. flags `"i"` -> `"(?i)pattern"`). Returns `None` for a
/// non-literal flags argument or an unrecognized flag, so the caller leaves the
/// predicate to a full recheck rather than risk changing its semantics.
fn apply_regex_flags(pattern: &str, flags_expr: &Expr) -> Option<String> {
    let (Expr::Literal(ScalarValue::Utf8(Some(flags)), _)
    | Expr::Literal(ScalarValue::LargeUtf8(Some(flags)), _)) = flags_expr
    else {
        return None;
    };
    let mut inline = String::new();
    for flag in flags.chars() {
        // Only flags expressible as an inline `(?...)` group in the regex crate
        // (which the recheck uses) are safe to fold.
        if ['i', 's', 'm', 'x'].contains(&flag) {
            inline.push(flag);
        } else {
            return None;
        }
    }
    if inline.is_empty() {
        Some(pattern.to_string())
    } else {
        Some(format!("(?{inline}){pattern}"))
    }
}

/// A parser for indices that handle queries with the contains_tokens function
#[derive(Debug, Clone)]
pub struct FtsQueryParser {
    index_name: String,
    index_type: String,
}

impl FtsQueryParser {
    pub fn new(name: String, index_type: String) -> Self {
        Self {
            index_name: name,
            index_type,
        }
    }
}

impl ScalarQueryParser for FtsQueryParser {
    fn visit_between(
        &self,
        _: &str,
        _: &Bound<ScalarValue>,
        _: &Bound<ScalarValue>,
    ) -> Option<IndexedExpression> {
        None
    }

    fn visit_in_list(&self, _: &str, _: &[ScalarValue]) -> Option<IndexedExpression> {
        None
    }

    fn visit_is_bool(&self, _: &str, _: bool) -> Option<IndexedExpression> {
        None
    }

    fn visit_is_null(&self, _: &str) -> Option<IndexedExpression> {
        None
    }

    fn visit_comparison(
        &self,
        _: &str,
        _: &ScalarValue,
        _: &Operator,
    ) -> Option<IndexedExpression> {
        None
    }

    fn visit_scalar_function(
        &self,
        column: &str,
        data_type: &DataType,
        func: &ScalarUDF,
        args: &[Expr],
    ) -> Option<IndexedExpression> {
        if args.len() != 2 {
            return None;
        }
        let scalar = maybe_scalar(&args[1], data_type)?;
        if let ScalarValue::Utf8(Some(scalar_str)) = scalar
            && func.name() == "contains_tokens"
        {
            let query = TokenQuery::TokensContains(scalar_str);
            return Some(IndexedExpression::index_query(
                column.to_string(),
                self.index_name.clone(),
                self.index_type.clone(),
                Arc::new(query),
            ));
        }
        None
    }
}

/// A parser for geo indices that handles spatial queries
#[cfg(feature = "geo")]
#[derive(Debug, Clone)]
pub struct GeoQueryParser {
    index_name: String,
    index_type: String,
}

#[cfg(feature = "geo")]
impl GeoQueryParser {
    pub fn new(index_name: String, index_type: String) -> Self {
        Self {
            index_name,
            index_type,
        }
    }
}

#[cfg(feature = "geo")]
impl ScalarQueryParser for GeoQueryParser {
    fn visit_between(
        &self,
        _: &str,
        _: &Bound<ScalarValue>,
        _: &Bound<ScalarValue>,
    ) -> Option<IndexedExpression> {
        None
    }

    fn visit_in_list(&self, _: &str, _: &[ScalarValue]) -> Option<IndexedExpression> {
        None
    }

    fn visit_is_bool(&self, _: &str, _: bool) -> Option<IndexedExpression> {
        None
    }

    fn visit_is_null(&self, column: &str) -> Option<IndexedExpression> {
        Some(IndexedExpression::index_query_with_recheck(
            column.to_string(),
            self.index_name.clone(),
            self.index_type.clone(),
            Arc::new(GeoQuery::IsNull),
            true,
        ))
    }

    fn visit_comparison(
        &self,
        _: &str,
        _: &ScalarValue,
        _: &Operator,
    ) -> Option<IndexedExpression> {
        None
    }

    fn visit_scalar_function(
        &self,
        column: &str,
        _data_type: &DataType,
        func: &ScalarUDF,
        args: &[Expr],
    ) -> Option<IndexedExpression> {
        if (func.name() == "st_intersects"
            || func.name() == "st_contains"
            || func.name() == "st_within"
            || func.name() == "st_touches"
            || func.name() == "st_crosses"
            || func.name() == "st_overlaps"
            || func.name() == "st_covers"
            || func.name() == "st_coveredby")
            && args.len() == 2
        {
            let left_arg = &args[0];
            let right_arg = &args[1];
            return match (left_arg, right_arg) {
                (Expr::Literal(left_value, metadata), Expr::Column(_)) => {
                    let mut field = Field::new("_geo", left_value.data_type(), false);
                    if let Some(metadata) = metadata {
                        field = field.with_metadata(metadata.to_hashmap());
                    }
                    let query = GeoQuery::IntersectQuery(RelationQuery {
                        value: left_value.clone(),
                        field,
                    });
                    Some(IndexedExpression::index_query_with_recheck(
                        column.to_string(),
                        self.index_name.clone(),
                        self.index_type.clone(),
                        Arc::new(query),
                        true,
                    ))
                }
                (Expr::Column(_), Expr::Literal(right_value, metadata)) => {
                    let mut field = Field::new("_geo", right_value.data_type(), false);
                    if let Some(metadata) = metadata {
                        field = field.with_metadata(metadata.to_hashmap());
                    }
                    let query = GeoQuery::IntersectQuery(RelationQuery {
                        value: right_value.clone(),
                        field,
                    });
                    Some(IndexedExpression::index_query_with_recheck(
                        column.to_string(),
                        self.index_name.clone(),
                        self.index_type.clone(),
                        Arc::new(query),
                        true,
                    ))
                }
                _ => None,
            };
        }
        None
    }
}

// Extract a literal scalar value from an expression, if it is a literal, or None
fn maybe_scalar(expr: &Expr, expected_type: &DataType) -> Option<ScalarValue> {
    match expr {
        Expr::Literal(value, _) => safe_coerce_scalar(value, expected_type),
        // Some literals can't be expressed in datafusion's SQL and can only be expressed with
        // a cast.  For example, there is no way to express a fixed-size-binary literal (which is
        // commonly used for UUID).  As a result the expression could look like...
        //
        // col = arrow_cast(value, 'fixed_size_binary(16)')
        //
        // In this case we need to extract the value, apply the cast, and then test the casted value
        Expr::Cast(cast) => match cast.expr.as_ref() {
            Expr::Literal(value, _) => {
                let casted = value.cast_to(&cast.data_type).ok()?;
                safe_coerce_scalar(&casted, expected_type)
            }
            _ => None,
        },
        Expr::ScalarFunction(scalar_function) => {
            if scalar_function.name() == "arrow_cast" {
                if scalar_function.args.len() != 2 {
                    return None;
                }
                match (&scalar_function.args[0], &scalar_function.args[1]) {
                    (Expr::Literal(value, _), Expr::Literal(cast_type, _)) => {
                        let target_type = scalar_function
                            .func
                            .return_field_from_args(ReturnFieldArgs {
                                arg_fields: &[
                                    Arc::new(Field::new("expression", value.data_type(), false)),
                                    Arc::new(Field::new("datatype", cast_type.data_type(), false)),
                                ],
                                scalar_arguments: &[Some(value), Some(cast_type)],
                            })
                            .ok()?;
                        let casted = value.cast_to(target_type.data_type()).ok()?;
                        safe_coerce_scalar(&casted, expected_type)
                    }
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

#[derive(Clone, Default, Debug)]
pub struct FilterPlan {
    pub index_query: Option<ScalarIndexExpr>,
    /// True if the index query is guaranteed to return exact results
    pub skip_recheck: bool,
    pub refine_expr: Option<Expr>,
    pub full_expr: Option<Expr>,
}

impl FilterPlan {
    pub fn empty() -> Self {
        Self {
            index_query: None,
            skip_recheck: true,
            refine_expr: None,
            full_expr: None,
        }
    }

    pub fn new_refine_only(expr: Expr) -> Self {
        Self {
            index_query: None,
            skip_recheck: true,
            refine_expr: Some(expr.clone()),
            full_expr: Some(expr),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.refine_expr.is_none() && self.index_query.is_none()
    }

    pub fn all_columns(&self) -> Vec<String> {
        self.full_expr
            .as_ref()
            .map(Planner::column_names_in_expr)
            .unwrap_or_default()
    }

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

    /// Return true if this has a scalar index query
    pub fn has_index_query(&self) -> bool {
        self.index_query.is_some()
    }

    pub fn has_any_filter(&self) -> bool {
        self.refine_expr.is_some() || self.index_query.is_some()
    }

    pub fn make_refine_only(&mut self) {
        self.index_query = None;
        self.refine_expr = self.full_expr.clone();
    }

    /// Return true if there is no refine or recheck of any kind and there is an index query
    pub fn is_exact_index_search(&self) -> bool {
        self.index_query.is_some() && self.refine_expr.is_none() && self.skip_recheck
    }
}

pub trait PlannerIndexExt {
    /// Determine how to apply a provided filter
    ///
    /// We parse the filter into a logical expression.  We then
    /// split the logical expression into a portion that can be
    /// satisfied by an index search (of one or more indices) and
    /// a refine portion that must be applied after the index search
    fn create_filter_plan(
        &self,
        filter: Expr,
        index_info: &dyn IndexInformationProvider,
        use_scalar_index: bool,
    ) -> Result<FilterPlan>;
}

impl PlannerIndexExt for Planner {
    fn create_filter_plan(
        &self,
        filter: Expr,
        index_info: &dyn IndexInformationProvider,
        use_scalar_index: bool,
    ) -> Result<FilterPlan> {
        let logical_expr = self.optimize_expr(filter)?;
        if use_scalar_index {
            let indexed_expr = apply_scalar_indices(logical_expr.clone(), index_info)?;
            let mut skip_recheck = false;
            if let Some(scalar_query) = indexed_expr.scalar_query.as_ref() {
                skip_recheck = !scalar_query.needs_recheck();
            }
            Ok(FilterPlan {
                index_query: indexed_expr.scalar_query,
                refine_expr: indexed_expr.refine_expr,
                full_expr: Some(logical_expr),
                skip_recheck,
            })
        } else {
            Ok(FilterPlan {
                index_query: None,
                skip_recheck: true,
                refine_expr: Some(logical_expr.clone()),
                full_expr: Some(logical_expr),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use arrow_array::Array;
    use arrow_schema::{Field, Schema};
    use chrono::Utc;
    use datafusion_common::{Column, DFSchema};
    use datafusion_expr::simplify::SimplifyContext;
    use lance_datafusion::exec::{LanceExecutionOptions, get_session_context};
    use lance_select::result::IndexExprResultWireFormat;
    use lance_select::{IndexExprResult, NullableIndexExprResult, NullableRowAddrMask};
    use roaring::RoaringBitmap;

    use crate::scalar::{
        AnyQuery,
        json::{JsonQuery, JsonQueryParser},
    };

    use super::*;

    struct ColInfo {
        data_type: DataType,
        parser: Box<MultiQueryParser>,
    }

    impl ColInfo {
        fn new(data_type: DataType, parser: Box<dyn ScalarQueryParser>) -> Self {
            Self {
                data_type,
                parser: Box::new(MultiQueryParser::single(parser)),
            }
        }

        fn with_multi(data_type: DataType, parser: Box<MultiQueryParser>) -> Self {
            Self { data_type, parser }
        }
    }

    struct MockIndexInfoProvider {
        indexed_columns: HashMap<String, ColInfo>,
    }

    impl MockIndexInfoProvider {
        fn new(indexed_columns: Vec<(&str, ColInfo)>) -> Self {
            Self {
                indexed_columns: HashMap::from_iter(
                    indexed_columns
                        .into_iter()
                        .map(|(s, ty)| (s.to_string(), ty)),
                ),
            }
        }
    }

    impl IndexInformationProvider for MockIndexInfoProvider {
        fn get_index(&self, col: &str) -> Option<(&DataType, &MultiQueryParser)> {
            self.indexed_columns
                .get(col)
                .map(|col_info| (&col_info.data_type, col_info.parser.as_ref()))
        }
    }

    fn check(
        index_info: &dyn IndexInformationProvider,
        expr: &str,
        expected: Option<IndexedExpression>,
        optimize: bool,
    ) {
        let schema = Schema::new(vec![
            Field::new("color", DataType::Utf8, false),
            Field::new("size", DataType::Float32, false),
            Field::new("aisle", DataType::UInt32, false),
            Field::new("on_sale", DataType::Boolean, false),
            Field::new("price", DataType::Float32, false),
            Field::new("json", DataType::LargeBinary, false),
        ]);
        let df_schema: DFSchema = schema.try_into().unwrap();

        let ctx = get_session_context(&LanceExecutionOptions::default());
        let state = ctx.state();
        let mut expr = state.create_logical_expr(expr, &df_schema).unwrap();
        if optimize {
            let simplify_context = SimplifyContext::default()
                .with_schema(Arc::new(df_schema))
                .with_query_execution_start_time(Some(Utc::now()));
            let simplifier =
                datafusion::optimizer::simplify_expressions::ExprSimplifier::new(simplify_context);
            expr = simplifier.simplify(expr).unwrap();
        }

        let actual = apply_scalar_indices(expr.clone(), index_info).unwrap();
        if let Some(expected) = expected {
            assert_eq!(actual, expected);
        } else {
            assert!(actual.scalar_query.is_none());
            assert_eq!(actual.refine_expr.unwrap(), expr);
        }
    }

    fn check_no_index(index_info: &dyn IndexInformationProvider, expr: &str) {
        check(index_info, expr, None, false)
    }

    fn check_simple(
        index_info: &dyn IndexInformationProvider,
        expr: &str,
        col: &str,
        query: impl AnyQuery,
    ) {
        check(
            index_info,
            expr,
            Some(IndexedExpression::index_query(
                col.to_string(),
                format!("{}_idx", col),
                "BTree".to_string(),
                Arc::new(query),
            )),
            false,
        )
    }

    fn check_range(
        index_info: &dyn IndexInformationProvider,
        expr: &str,
        col: &str,
        query: SargableQuery,
    ) {
        check(
            index_info,
            expr,
            Some(IndexedExpression::index_query(
                col.to_string(),
                format!("{}_idx", col),
                "BTree".to_string(),
                Arc::new(query),
            )),
            true,
        )
    }

    fn check_simple_negated(
        index_info: &dyn IndexInformationProvider,
        expr: &str,
        col: &str,
        query: SargableQuery,
    ) {
        check(
            index_info,
            expr,
            Some(
                IndexedExpression::index_query(
                    col.to_string(),
                    format!("{}_idx", col),
                    "BTree".to_string(),
                    Arc::new(query),
                )
                .maybe_not()
                .unwrap(),
            ),
            false,
        )
    }

    #[test]
    fn test_expressions() {
        let index_info = MockIndexInfoProvider::new(vec![
            (
                "color",
                ColInfo::new(
                    DataType::Utf8,
                    Box::new(SargableQueryParser::new(
                        "color_idx".to_string(),
                        "BTree".to_string(),
                        false,
                    )),
                ),
            ),
            (
                "aisle",
                ColInfo::new(
                    DataType::UInt32,
                    Box::new(SargableQueryParser::new(
                        "aisle_idx".to_string(),
                        "BTree".to_string(),
                        false,
                    )),
                ),
            ),
            (
                "on_sale",
                ColInfo::new(
                    DataType::Boolean,
                    Box::new(SargableQueryParser::new(
                        "on_sale_idx".to_string(),
                        "BTree".to_string(),
                        false,
                    )),
                ),
            ),
            (
                "price",
                ColInfo::new(
                    DataType::Float32,
                    Box::new(SargableQueryParser::new(
                        "price_idx".to_string(),
                        "BTree".to_string(),
                        false,
                    )),
                ),
            ),
            (
                "json",
                ColInfo::new(
                    DataType::LargeBinary,
                    Box::new(JsonQueryParser::new(
                        "$.name".to_string(),
                        Box::new(SargableQueryParser::new(
                            "json_idx".to_string(),
                            "BTree".to_string(),
                            false,
                        )),
                    )),
                ),
            ),
        ]);

        check_simple(
            &index_info,
            "json_extract(json, '$.name') = 'foo'",
            "json",
            JsonQuery::new(
                Arc::new(SargableQuery::Equals(ScalarValue::Utf8(Some(
                    "foo".to_string(),
                )))),
                "$.name".to_string(),
            ),
        );

        check_no_index(&index_info, "size BETWEEN 5 AND 10");
        // Cast case.  We will cast 5 (an int64) to Int16 and then coerce to UInt32
        check_simple(
            &index_info,
            "aisle = arrow_cast(5, 'Int16')",
            "aisle",
            SargableQuery::Equals(ScalarValue::UInt32(Some(5))),
        );
        // 5 different ways of writing BETWEEN (all should be recognized)
        check_range(
            &index_info,
            "aisle BETWEEN 5 AND 10",
            "aisle",
            SargableQuery::Range(
                Bound::Included(ScalarValue::UInt32(Some(5))),
                Bound::Included(ScalarValue::UInt32(Some(10))),
            ),
        );
        check_range(
            &index_info,
            "aisle >= 5 AND aisle <= 10",
            "aisle",
            SargableQuery::Range(
                Bound::Included(ScalarValue::UInt32(Some(5))),
                Bound::Included(ScalarValue::UInt32(Some(10))),
            ),
        );

        check_range(
            &index_info,
            "aisle <= 10 AND aisle >= 5",
            "aisle",
            SargableQuery::Range(
                Bound::Included(ScalarValue::UInt32(Some(5))),
                Bound::Included(ScalarValue::UInt32(Some(10))),
            ),
        );

        check_range(
            &index_info,
            "5 <= aisle AND 10 >= aisle",
            "aisle",
            SargableQuery::Range(
                Bound::Included(ScalarValue::UInt32(Some(5))),
                Bound::Included(ScalarValue::UInt32(Some(10))),
            ),
        );

        check_range(
            &index_info,
            "10 >= aisle AND 5 <= aisle",
            "aisle",
            SargableQuery::Range(
                Bound::Included(ScalarValue::UInt32(Some(5))),
                Bound::Included(ScalarValue::UInt32(Some(10))),
            ),
        );
        check_range(
            &index_info,
            "aisle <= 10 AND aisle > 5",
            "aisle",
            SargableQuery::Range(
                Bound::Excluded(ScalarValue::UInt32(Some(5))),
                Bound::Included(ScalarValue::UInt32(Some(10))),
            ),
        );
        check_range(
            &index_info,
            "aisle < 10 AND aisle >= 5",
            "aisle",
            SargableQuery::Range(
                Bound::Included(ScalarValue::UInt32(Some(5))),
                Bound::Excluded(ScalarValue::UInt32(Some(10))),
            ),
        );
        check_simple(
            &index_info,
            "on_sale IS TRUE",
            "on_sale",
            SargableQuery::Equals(ScalarValue::Boolean(Some(true))),
        );
        check_simple(
            &index_info,
            "on_sale",
            "on_sale",
            SargableQuery::Equals(ScalarValue::Boolean(Some(true))),
        );
        check_simple_negated(
            &index_info,
            "NOT on_sale",
            "on_sale",
            SargableQuery::Equals(ScalarValue::Boolean(Some(true))),
        );
        check_simple(
            &index_info,
            "on_sale IS FALSE",
            "on_sale",
            SargableQuery::Equals(ScalarValue::Boolean(Some(false))),
        );
        check_simple_negated(
            &index_info,
            "aisle NOT BETWEEN 5 AND 10",
            "aisle",
            SargableQuery::Range(
                Bound::Included(ScalarValue::UInt32(Some(5))),
                Bound::Included(ScalarValue::UInt32(Some(10))),
            ),
        );
        // Small in-list (in-list with 3 or fewer items optimizes into or-chain)
        check_simple(
            &index_info,
            "aisle IN (5, 6, 7)",
            "aisle",
            SargableQuery::IsIn(vec![
                ScalarValue::UInt32(Some(5)),
                ScalarValue::UInt32(Some(6)),
                ScalarValue::UInt32(Some(7)),
            ]),
        );
        check_simple_negated(
            &index_info,
            "NOT aisle IN (5, 6, 7)",
            "aisle",
            SargableQuery::IsIn(vec![
                ScalarValue::UInt32(Some(5)),
                ScalarValue::UInt32(Some(6)),
                ScalarValue::UInt32(Some(7)),
            ]),
        );
        check_simple_negated(
            &index_info,
            "aisle NOT IN (5, 6, 7)",
            "aisle",
            SargableQuery::IsIn(vec![
                ScalarValue::UInt32(Some(5)),
                ScalarValue::UInt32(Some(6)),
                ScalarValue::UInt32(Some(7)),
            ]),
        );
        check_simple(
            &index_info,
            "aisle IN (5, 6, 7, 8, 9)",
            "aisle",
            SargableQuery::IsIn(vec![
                ScalarValue::UInt32(Some(5)),
                ScalarValue::UInt32(Some(6)),
                ScalarValue::UInt32(Some(7)),
                ScalarValue::UInt32(Some(8)),
                ScalarValue::UInt32(Some(9)),
            ]),
        );
        check_simple_negated(
            &index_info,
            "NOT aisle IN (5, 6, 7, 8, 9)",
            "aisle",
            SargableQuery::IsIn(vec![
                ScalarValue::UInt32(Some(5)),
                ScalarValue::UInt32(Some(6)),
                ScalarValue::UInt32(Some(7)),
                ScalarValue::UInt32(Some(8)),
                ScalarValue::UInt32(Some(9)),
            ]),
        );
        check_simple_negated(
            &index_info,
            "aisle NOT IN (5, 6, 7, 8, 9)",
            "aisle",
            SargableQuery::IsIn(vec![
                ScalarValue::UInt32(Some(5)),
                ScalarValue::UInt32(Some(6)),
                ScalarValue::UInt32(Some(7)),
                ScalarValue::UInt32(Some(8)),
                ScalarValue::UInt32(Some(9)),
            ]),
        );
        check_simple(
            &index_info,
            "on_sale is false",
            "on_sale",
            SargableQuery::Equals(ScalarValue::Boolean(Some(false))),
        );
        check_simple(
            &index_info,
            "on_sale is true",
            "on_sale",
            SargableQuery::Equals(ScalarValue::Boolean(Some(true))),
        );
        check_simple(
            &index_info,
            "aisle < 10",
            "aisle",
            SargableQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::UInt32(Some(10))),
            ),
        );
        check_simple(
            &index_info,
            "aisle <= 10",
            "aisle",
            SargableQuery::Range(
                Bound::Unbounded,
                Bound::Included(ScalarValue::UInt32(Some(10))),
            ),
        );
        check_simple(
            &index_info,
            "aisle > 10",
            "aisle",
            SargableQuery::Range(
                Bound::Excluded(ScalarValue::UInt32(Some(10))),
                Bound::Unbounded,
            ),
        );
        // In the future we can handle this case if we need to.  For
        // now let's make sure we don't accidentally do the wrong thing
        // (we were getting this backwards in the past)
        check_no_index(&index_info, "10 > aisle");
        check_simple(
            &index_info,
            "aisle >= 10",
            "aisle",
            SargableQuery::Range(
                Bound::Included(ScalarValue::UInt32(Some(10))),
                Bound::Unbounded,
            ),
        );
        check_simple(
            &index_info,
            "aisle = 10",
            "aisle",
            SargableQuery::Equals(ScalarValue::UInt32(Some(10))),
        );
        check_simple_negated(
            &index_info,
            "aisle <> 10",
            "aisle",
            SargableQuery::Equals(ScalarValue::UInt32(Some(10))),
        );
        // // Common compound case, AND'd clauses
        let left = Box::new(ScalarIndexExpr::Query(ScalarIndexSearch {
            column: "aisle".to_string(),
            index_name: "aisle_idx".to_string(),
            index_type: "BTree".to_string(),
            query: Arc::new(SargableQuery::Equals(ScalarValue::UInt32(Some(10)))),
            needs_recheck: false,
            fragment_bitmap: None,
        }));
        let right = Box::new(ScalarIndexExpr::Query(ScalarIndexSearch {
            column: "color".to_string(),
            index_name: "color_idx".to_string(),
            index_type: "BTree".to_string(),
            query: Arc::new(SargableQuery::Equals(ScalarValue::Utf8(Some(
                "blue".to_string(),
            )))),
            needs_recheck: false,
            fragment_bitmap: None,
        }));
        check(
            &index_info,
            "aisle = 10 AND color = 'blue'",
            Some(IndexedExpression {
                scalar_query: Some(ScalarIndexExpr::And(left.clone(), right.clone())),
                refine_expr: None,
            }),
            false,
        );
        // Compound AND's and not all of them are indexed columns
        let refine = Expr::Column(Column::new_unqualified("size")).gt(datafusion_expr::lit(30_i64));
        check(
            &index_info,
            "aisle = 10 AND color = 'blue' AND size > 30",
            Some(IndexedExpression {
                scalar_query: Some(ScalarIndexExpr::And(left.clone(), right.clone())),
                refine_expr: Some(refine.clone()),
            }),
            false,
        );
        // Compounded OR's where ALL columns are indexed
        check(
            &index_info,
            "aisle = 10 OR color = 'blue'",
            Some(IndexedExpression {
                scalar_query: Some(ScalarIndexExpr::Or(left.clone(), right.clone())),
                refine_expr: None,
            }),
            false,
        );
        // Compounded OR's with one or more unindexed columns
        check_no_index(&index_info, "aisle = 10 OR color = 'blue' OR size > 30");
        // AND'd group of OR
        check(
            &index_info,
            "(aisle = 10 OR color = 'blue') AND size > 30",
            Some(IndexedExpression {
                scalar_query: Some(ScalarIndexExpr::Or(left, right)),
                refine_expr: Some(refine),
            }),
            false,
        );
        // Examples of things that are not yet supported but should be supportable someday

        // OR'd group of refined index searches (see IndexedExpression::or for details)
        check_no_index(
            &index_info,
            "(aisle = 10 AND size > 30) OR (color = 'blue' AND size > 20)",
        );

        // Non-normalized arithmetic (can use expression simplification)
        check_no_index(&index_info, "aisle + 3 < 10");

        // Currently we assume that the return of an index search tells us which rows are
        // TRUE and all other rows are FALSE.  This will need to change but for now it is
        // safer to not support the following cases because the return value of non-matched
        // rows is NULL and not FALSE.
        check_no_index(&index_info, "aisle IN (5, 6, NULL)");
        // OR-list with NULL (in future DF version this will be optimized repr of
        // small in-list with NULL so let's get ready for it)
        check_no_index(&index_info, "aisle = 5 OR aisle = 6 OR NULL");
        check_no_index(&index_info, "aisle IN (5, 6, 7, 8, NULL)");
        check_no_index(&index_info, "aisle = NULL");
        check_no_index(&index_info, "aisle BETWEEN 5 AND NULL");
        check_no_index(&index_info, "aisle BETWEEN NULL AND 10");
    }

    #[tokio::test]
    async fn test_not_flips_certainty() {
        use lance_select::{NullableRowAddrSet, RowAddrTreeMap};

        // Test that NOT flips certainty for inexact index results.
        // Under the {lower, upper} form, `!{l, u} = {!u, !l}`, which
        // preserves the AtMost ↔ AtLeast swap and leaves Exact as Exact.

        // AtMost: superset of matches (e.g., bloom filter says "might be in [1,2]")
        let at_most = NullableIndexExprResult::at_most(NullableRowAddrMask::AllowList(
            NullableRowAddrSet::new(RowAddrTreeMap::from_iter(&[1, 2]), RowAddrTreeMap::new()),
        ));
        // NOT(AtMost) should be AtLeast (definitely NOT in [1,2], might be elsewhere)
        assert!((!at_most).is_at_least());

        // AtLeast: subset of matches (e.g., definitely in [1,2], might be more)
        let at_least = NullableIndexExprResult::at_least(NullableRowAddrMask::AllowList(
            NullableRowAddrSet::new(RowAddrTreeMap::from_iter(&[1, 2]), RowAddrTreeMap::new()),
        ));
        // NOT(AtLeast) should be AtMost (might NOT be in [1,2], definitely elsewhere)
        assert!((!at_least).is_at_most());

        // Exact should stay Exact
        let exact = NullableIndexExprResult::exact(NullableRowAddrMask::AllowList(
            NullableRowAddrSet::new(RowAddrTreeMap::from_iter(&[1, 2]), RowAddrTreeMap::new()),
        ));
        assert!((!exact).is_exact());
    }

    #[tokio::test]
    async fn test_and_or_preserve_certainty() {
        use lance_select::{NullableRowAddrSet, RowAddrTreeMap};

        // Test that AND/OR correctly propagate certainty under the
        // {lower, upper} algebra. Each binary op is elementwise on the
        // endpoints, so degenerate shapes (Exact / AtMost / AtLeast)
        // combine into a result that lands in one of those same shapes
        // in every case exercised below.
        let make_at_most = || {
            NullableIndexExprResult::at_most(NullableRowAddrMask::AllowList(
                NullableRowAddrSet::new(
                    RowAddrTreeMap::from_iter(&[1, 2, 3]),
                    RowAddrTreeMap::new(),
                ),
            ))
        };

        let make_at_least = || {
            NullableIndexExprResult::at_least(NullableRowAddrMask::AllowList(
                NullableRowAddrSet::new(
                    RowAddrTreeMap::from_iter(&[2, 3, 4]),
                    RowAddrTreeMap::new(),
                ),
            ))
        };

        let make_exact = || {
            NullableIndexExprResult::exact(NullableRowAddrMask::AllowList(NullableRowAddrSet::new(
                RowAddrTreeMap::from_iter(&[1, 2]),
                RowAddrTreeMap::new(),
            )))
        };

        // AtMost & AtMost → AtMost
        assert!((make_at_most() & make_at_most()).is_at_most());

        // AtLeast & AtLeast → AtLeast
        assert!((make_at_least() & make_at_least()).is_at_least());

        // AtMost & AtLeast → AtMost (the lower side stays empty)
        assert!((make_at_most() & make_at_least()).is_at_most());

        // AtMost | AtMost → AtMost
        assert!((make_at_most() | make_at_most()).is_at_most());

        // AtLeast | AtLeast → AtLeast
        assert!((make_at_least() | make_at_least()).is_at_least());

        // AtMost | AtLeast → AtLeast (upper stays universe)
        assert!((make_at_most() | make_at_least()).is_at_least());

        // Exact & AtMost → AtMost
        assert!((make_exact() & make_at_most()).is_at_most());

        // Exact | AtLeast → AtLeast
        assert!((make_exact() | make_at_least()).is_at_least());
    }

    /// The whole point of the `{lower, upper}` representation is that it
    /// can express a Refined result — a non-empty `lower` strictly inside
    /// a non-universe `upper` — which the old enum couldn't. This test
    /// constructs one through the algebra and verifies the endpoints.
    #[tokio::test]
    async fn test_refined_result_constructed_through_algebra() {
        use lance_select::{NullableRowAddrSet, RowAddrTreeMap};

        let allow_set = |rows: &[u64]| {
            NullableRowAddrMask::AllowList(NullableRowAddrSet::new(
                RowAddrTreeMap::from_iter(rows),
                RowAddrTreeMap::new(),
            ))
        };

        // AtLeast({1,2}) & Exact({1,2,3}) is Refined, because:
        //   lower = {1,2} ∩ {1,2,3} = {1,2}        (non-empty)
        //   upper = universe ∩ {1,2,3} = {1,2,3}   (not universe)
        //   lower ≠ upper                          (not Exact)
        let at_least_12 = NullableIndexExprResult::at_least(allow_set(&[1, 2]));
        let exact_123 = NullableIndexExprResult::exact(allow_set(&[1, 2, 3]));
        let refined = at_least_12 & exact_123;

        // None of the shape predicates should fire — that's what makes
        // this a Refined result.
        assert!(
            !refined.is_exact(),
            "Refined must not be classified as Exact"
        );
        assert!(
            !refined.is_at_most(),
            "Refined must not be classified as AtMost"
        );
        assert!(
            !refined.is_at_least(),
            "Refined must not be classified as AtLeast"
        );

        // Check the actual endpoints.
        assert_eq!(refined.lower, allow_set(&[1, 2]));
        assert_eq!(refined.upper, allow_set(&[1, 2, 3]));

        // NOT swaps the endpoints, preserving the Refined shape.
        let negated = !refined;
        assert!(!negated.is_exact());
        assert!(!negated.is_at_most());
        assert!(!negated.is_at_least());

        // !{l, u} = {!u, !l}. AllowList → BlockList.
        assert!(matches!(negated.lower, NullableRowAddrMask::BlockList(_)));
        assert!(matches!(negated.upper, NullableRowAddrMask::BlockList(_)));
    }

    #[test]
    fn test_like_to_regex() {
        // `%` -> `.*`, `_` -> `.`, with a literal run of at least three chars.
        assert_eq!(like_to_regex("%foo%", None).as_deref(), Some(".*foo.*"));
        assert_eq!(like_to_regex("foo%bar", None).as_deref(), Some("foo.*bar"));
        assert_eq!(like_to_regex("foo_bar", None).as_deref(), Some("foo.bar"));
        assert_eq!(like_to_regex("foobar", None).as_deref(), Some("foobar"));

        // Regex metacharacters in the literal portion are escaped.
        assert_eq!(
            like_to_regex("%a.bcd%", None).as_deref(),
            Some(".*a\\.bcd.*")
        );

        // No literal run of three alphanumeric characters -> no index help.
        assert_eq!(like_to_regex("%ab%", None), None);
        assert_eq!(like_to_regex("%a%b%c%", None), None);
        assert_eq!(like_to_regex("%", None), None);

        // The escape character makes the following character a literal.
        assert_eq!(
            like_to_regex(r"%foo\%bar%", Some('\\')).as_deref(),
            Some(".*foo%bar.*")
        );
    }

    #[test]
    fn test_apply_regex_flags() {
        fn flags(s: &str) -> Expr {
            Expr::Literal(ScalarValue::Utf8(Some(s.to_string())), None)
        }

        // Empty flags leave the pattern untouched (no inline group emitted).
        assert_eq!(apply_regex_flags("foo", &flags("")).as_deref(), Some("foo"));
        // Supported flags are folded into an inline `(?...)` prefix.
        assert_eq!(
            apply_regex_flags("foo", &flags("i")).as_deref(),
            Some("(?i)foo")
        );
        assert_eq!(
            apply_regex_flags("foo", &flags("is")).as_deref(),
            Some("(?is)foo")
        );
        // An unrecognized flag bails out so the caller leaves the predicate to a
        // full recheck rather than risk changing its semantics.
        assert_eq!(apply_regex_flags("foo", &flags("g")), None);
        // A non-string (hence non-literal-flags) argument cannot be folded.
        assert_eq!(
            apply_regex_flags("foo", &Expr::Literal(ScalarValue::Int32(Some(1)), None)),
            None
        );
    }

    #[test]
    fn test_extract_like_leading_prefix() {
        // Simple prefix patterns (no recheck needed)
        assert_eq!(
            extract_like_leading_prefix("foo%", None),
            Some(("foo".to_string(), false))
        );
        assert_eq!(
            extract_like_leading_prefix("abc%", None),
            Some(("abc".to_string(), false))
        );

        // Patterns with wildcards in the middle (need recheck)
        assert_eq!(
            extract_like_leading_prefix("foo%bar%", None),
            Some(("foo".to_string(), true))
        );
        assert_eq!(
            extract_like_leading_prefix("foo_bar%", None),
            Some(("foo".to_string(), true))
        );
        assert_eq!(
            extract_like_leading_prefix("foo%bar", None),
            Some(("foo".to_string(), true))
        );
        assert_eq!(
            extract_like_leading_prefix("foo_", None),
            Some(("foo".to_string(), true))
        );

        // Not prefix patterns (starts with wildcard)
        assert_eq!(extract_like_leading_prefix("%foo", None), None);
        assert_eq!(extract_like_leading_prefix("_foo%", None), None);
        assert_eq!(extract_like_leading_prefix("%", None), None);

        // No wildcard at all (should use equality)
        assert_eq!(extract_like_leading_prefix("foo", None), None);

        // With escape character
        assert_eq!(
            extract_like_leading_prefix(r"foo\%bar%", Some('\\')),
            Some(("foo%bar".to_string(), false))
        );
        assert_eq!(
            extract_like_leading_prefix(r"foo\_bar%", Some('\\')),
            Some(("foo_bar".to_string(), false))
        );
        assert_eq!(
            extract_like_leading_prefix(r"foo\\bar%", Some('\\')),
            Some(("foo\\bar".to_string(), false))
        );

        // Escaped trailing % is not a wildcard (no wildcards)
        assert_eq!(extract_like_leading_prefix(r"foo\%", Some('\\')), None);

        // With backslash as default escape (for DataFusion starts_with compatibility):
        // "foo\%" means escaped %, no wildcard -> None (should use equality)
        assert_eq!(extract_like_leading_prefix(r"foo\%", None), None);
        // "foo\bar%" - \b is not a valid escape sequence, so \ and b are literals, % is wildcard
        assert_eq!(
            extract_like_leading_prefix(r"foo\bar%", None),
            Some(("foo\\bar".to_string(), false))
        );

        // Empty pattern
        assert_eq!(extract_like_leading_prefix("", None), None);

        // Mixed escaped and unescaped
        assert_eq!(
            extract_like_leading_prefix(r"foo\%bar%baz%", Some('\\')),
            Some(("foo%bar".to_string(), true))
        );
    }

    #[test]
    fn test_like_expression_parsing() {
        // Test that LIKE expressions are parsed correctly with refine_expr for complex patterns

        let index_info = MockIndexInfoProvider::new(vec![(
            "color",
            ColInfo::new(
                DataType::Utf8,
                Box::new(SargableQueryParser::new(
                    "color_idx".to_string(),
                    "BTree".to_string(),
                    false,
                )),
            ),
        )]);

        // Simple prefix pattern: LIKE 'foo%' -> LikePrefix("foo"), no refine_expr
        let schema = Schema::new(vec![Field::new("color", DataType::Utf8, false)]);
        let df_schema: DFSchema = schema.try_into().unwrap();
        let ctx = get_session_context(&LanceExecutionOptions::default());
        let state = ctx.state();

        let expr = state
            .create_logical_expr("color LIKE 'foo%'", &df_schema)
            .unwrap();
        let result = apply_scalar_indices(expr, &index_info).unwrap();

        assert!(result.scalar_query.is_some(), "Should have scalar_query");
        assert!(
            result.refine_expr.is_none(),
            "Simple prefix should not need refine_expr"
        );

        // Extract the query and verify it's LikePrefix
        if let Some(ScalarIndexExpr::Query(search)) = &result.scalar_query {
            let query = search.query.as_any().downcast_ref::<SargableQuery>();
            assert!(query.is_some(), "Query should be SargableQuery");
            match query.unwrap() {
                SargableQuery::LikePrefix(prefix) => {
                    assert_eq!(prefix, &ScalarValue::Utf8(Some("foo".to_string())));
                }
                _ => panic!("Expected LikePrefix query"),
            }
        } else {
            panic!("Expected Query variant");
        }

        // Complex pattern: LIKE 'foo%bar%' -> LikePrefix("foo"), with refine_expr
        let expr = state
            .create_logical_expr("color LIKE 'foo%bar%'", &df_schema)
            .unwrap();
        let result = apply_scalar_indices(expr, &index_info).unwrap();

        assert!(result.scalar_query.is_some(), "Should have scalar_query");
        assert!(
            result.refine_expr.is_some(),
            "Complex pattern should have refine_expr"
        );

        // Verify the query is still LikePrefix("foo")
        if let Some(ScalarIndexExpr::Query(search)) = &result.scalar_query {
            let query = search.query.as_any().downcast_ref::<SargableQuery>();
            assert!(query.is_some(), "Query should be SargableQuery");
            match query.unwrap() {
                SargableQuery::LikePrefix(prefix) => {
                    assert_eq!(prefix, &ScalarValue::Utf8(Some("foo".to_string())));
                }
                _ => panic!("Expected LikePrefix query"),
            }
        }

        // Verify the refine_expr is the original LIKE expression
        let refine = result.refine_expr.unwrap();
        match refine {
            Expr::Like(like) => {
                assert!(!like.negated);
                assert!(!like.case_insensitive);
                if let Expr::Literal(ScalarValue::Utf8(Some(pattern)), _) = like.pattern.as_ref() {
                    assert_eq!(pattern, "foo%bar%");
                } else {
                    panic!("Expected Utf8 literal pattern");
                }
            }
            _ => panic!("Expected Like expression in refine_expr"),
        }

        // Pattern starting with wildcard: LIKE '%foo' -> no index, only refine
        let expr = state
            .create_logical_expr("color LIKE '%foo'", &df_schema)
            .unwrap();
        let result = apply_scalar_indices(expr, &index_info).unwrap();

        assert!(
            result.scalar_query.is_none(),
            "Pattern starting with wildcard should not use index"
        );
        assert!(result.refine_expr.is_some(), "Should fall back to refine");
    }

    #[test]
    fn test_starts_with_with_underscore_after_optimization() {
        // Test that starts_with with underscore in prefix works correctly after DataFusion optimization
        // DataFusion simplifies starts_with(col, 'test_ns$') to col LIKE 'test_ns$%'
        // The underscore in the prefix should NOT be treated as a wildcard!
        let index_info = MockIndexInfoProvider::new(vec![(
            "object_id",
            ColInfo::new(
                DataType::Utf8,
                Box::new(SargableQueryParser::new(
                    "object_id_idx".to_string(),
                    "BTree".to_string(),
                    false,
                )),
            ),
        )]);

        let schema = Schema::new(vec![Field::new("object_id", DataType::Utf8, false)]);
        let df_schema: DFSchema = schema.try_into().unwrap();
        let ctx = get_session_context(&LanceExecutionOptions::default());
        let state = ctx.state();

        // Create the expression with starts_with containing underscore
        let expr = state
            .create_logical_expr("starts_with(object_id, 'test_ns$')", &df_schema)
            .unwrap();

        // Apply DataFusion simplification (this may convert starts_with to LIKE)
        let simplify_context = SimplifyContext::default()
            .with_schema(Arc::new(df_schema))
            .with_query_execution_start_time(Some(Utc::now()));
        let simplifier =
            datafusion::optimizer::simplify_expressions::ExprSimplifier::new(simplify_context);
        let simplified_expr = simplifier.simplify(expr).unwrap();

        // Apply scalar indices
        let result = apply_scalar_indices(simplified_expr, &index_info).unwrap();

        // The prefix should be "test_ns$", NOT "test"
        // This test documents the current (potentially broken) behavior
        if let Some(ScalarIndexExpr::Query(search)) = &result.scalar_query {
            let query = search
                .query
                .as_any()
                .downcast_ref::<SargableQuery>()
                .unwrap();
            match query {
                SargableQuery::LikePrefix(prefix) => {
                    let prefix_str = match prefix {
                        ScalarValue::Utf8(Some(s)) => s.clone(),
                        _ => panic!("Expected Utf8 prefix"),
                    };
                    // Verify the prefix is correctly extracted with underscore as literal
                    assert_eq!(
                        prefix_str, "test_ns$",
                        "Prefix should be 'test_ns$', not 'test' (underscore should not be a wildcard)"
                    );
                }
                _ => panic!("Expected LikePrefix query"),
            }
        } else {
            // If no scalar query, it means the pattern was not recognized
            panic!("Expected scalar_query to be present");
        }
    }

    #[test]
    fn test_starts_with_to_like_conversion() {
        // Test that starts_with(col, 'prefix') is converted to LikePrefix query
        let index_info = MockIndexInfoProvider::new(vec![(
            "color",
            ColInfo::new(
                DataType::Utf8,
                Box::new(SargableQueryParser::new(
                    "color_idx".to_string(),
                    "BTree".to_string(),
                    false,
                )),
            ),
        )]);

        let schema = Schema::new(vec![Field::new("color", DataType::Utf8, false)]);
        let df_schema: DFSchema = schema.try_into().unwrap();
        let ctx = get_session_context(&LanceExecutionOptions::default());
        let state = ctx.state();

        // starts_with(color, 'foo') should be converted to LikePrefix("foo")
        let expr = state
            .create_logical_expr("starts_with(color, 'foo')", &df_schema)
            .unwrap();
        let result = apply_scalar_indices(expr, &index_info).unwrap();

        assert!(
            result.scalar_query.is_some(),
            "starts_with should use index"
        );
        assert!(
            result.refine_expr.is_none(),
            "Pure prefix starts_with should not need refine_expr"
        );

        // Extract the query and verify it's LikePrefix
        if let Some(ScalarIndexExpr::Query(search)) = &result.scalar_query {
            let query = search.query.as_any().downcast_ref::<SargableQuery>();
            assert!(query.is_some(), "Query should be SargableQuery");
            match query.unwrap() {
                SargableQuery::LikePrefix(prefix) => {
                    assert_eq!(prefix, &ScalarValue::Utf8(Some("foo".to_string())));
                }
                _ => panic!("Expected LikePrefix query"),
            }
        } else {
            panic!("Expected Query variant");
        }

        // Both starts_with and LIKE 'prefix%' should produce the same LikePrefix query
        let like_expr = state
            .create_logical_expr("color LIKE 'foo%'", &df_schema)
            .unwrap();
        let like_result = apply_scalar_indices(like_expr, &index_info).unwrap();

        // Compare the queries - both should be LikePrefix("foo")
        if let (
            Some(ScalarIndexExpr::Query(starts_with_search)),
            Some(ScalarIndexExpr::Query(like_search)),
        ) = (&result.scalar_query, &like_result.scalar_query)
        {
            let sw_query = starts_with_search
                .query
                .as_any()
                .downcast_ref::<SargableQuery>()
                .unwrap();
            let like_query = like_search
                .query
                .as_any()
                .downcast_ref::<SargableQuery>()
                .unwrap();
            assert_eq!(
                sw_query, like_query,
                "starts_with and LIKE 'prefix%' should produce identical queries"
            );
        }
    }

    #[test]
    fn test_serialize_index_expr_result_round_trip() {
        use lance_select::{RowAddrMask, RowAddrTreeMap};

        for format in [
            IndexExprResultWireFormat::TwoMask,
            IndexExprResultWireFormat::ThreeVariant,
        ] {
            let mut addrs = RowAddrTreeMap::new();
            addrs.insert_range(0..5);
            addrs.insert_range(100..103);

            let mut fragments_covered = RoaringBitmap::new();
            fragments_covered.insert(0);
            fragments_covered.insert(7);

            let cases = [
                (
                    "exact",
                    IndexExprResult::exact(RowAddrMask::from_allowed(addrs.clone())),
                ),
                (
                    "at_most",
                    IndexExprResult::at_most(RowAddrMask::from_allowed(addrs.clone())),
                ),
                (
                    "at_least",
                    IndexExprResult::at_least(RowAddrMask::from_allowed(addrs)),
                ),
            ];

            for (label, original) in cases {
                let batch = original.serialize(&fragments_covered, format).unwrap();
                assert_eq!(
                    batch.schema(),
                    *format.schema(),
                    "format {format:?}, case {label}"
                );
                assert_eq!(batch.num_rows(), 2, "format {format:?}, case {label}");

                let (round_tripped, round_tripped_frags) =
                    IndexExprResult::deserialize(&batch).unwrap();
                assert_eq!(
                    round_tripped.lower, original.lower,
                    "format {format:?}, case {label}: lower"
                );
                assert_eq!(
                    round_tripped.upper, original.upper,
                    "format {format:?}, case {label}: upper"
                );
                assert_eq!(
                    round_tripped_frags, fragments_covered,
                    "format {format:?}, case {label}: frags"
                );
                assert_eq!(
                    round_tripped.is_exact(),
                    original.is_exact(),
                    "format {format:?}, case {label}"
                );
                assert_eq!(
                    round_tripped.is_at_most(),
                    original.is_at_most(),
                    "format {format:?}, case {label}"
                );
                assert_eq!(
                    round_tripped.is_at_least(),
                    original.is_at_least(),
                    "format {format:?}, case {label}"
                );
            }
        }
    }

    /// Exact results encode `upper` as a fully-null column on the wire — the
    /// payload only needs to ship once. `RowAddrMask::into_arrow` never
    /// produces a fully-null array (it always sets exactly one of the two
    /// rows), so the sentinel can't collide with a real mask. This pins
    /// both halves: exact ⇒ upper fully null, non-exact ⇒ upper carries the
    /// real mask.
    #[test]
    fn test_serialize_omits_upper_when_exact() {
        use lance_select::{RowAddrMask, RowAddrTreeMap};

        let mask = RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(0u64..5));
        let fragments_covered = RoaringBitmap::from_iter([0u32]);

        use arrow::array::AsArray;

        // Exact: upper column must be fully null on the wire.
        let exact_batch = IndexExprResult::exact(mask.clone())
            .serialize(&fragments_covered, IndexExprResultWireFormat::TwoMask)
            .unwrap();
        let exact_upper = exact_batch.column(1).as_binary::<i32>();
        assert!(exact_upper.is_null(0) && exact_upper.is_null(1));

        // Non-exact (at_most): upper column must carry the upper mask, so at
        // least one row is non-null (`AllowList(mask)` puts the payload at
        // row 1).
        let at_most_batch = IndexExprResult::at_most(mask.clone())
            .serialize(&fragments_covered, IndexExprResultWireFormat::TwoMask)
            .unwrap();
        let at_most_upper = at_most_batch.column(1).as_binary::<i32>();
        assert!(!(at_most_upper.is_null(0) && at_most_upper.is_null(1)));

        // Non-exact (at_least): upper = all_rows, which `into_arrow`
        // encodes as `BlockList(empty)` — row 0 holds the empty-tree bytes,
        // row 1 is null. Round-trip must preserve `is_at_least`.
        let at_least_batch = IndexExprResult::at_least(mask)
            .serialize(&fragments_covered, IndexExprResultWireFormat::TwoMask)
            .unwrap();
        let at_least_upper = at_least_batch.column(1).as_binary::<i32>();
        assert!(!at_least_upper.is_null(0));
        let (round_tripped, _) = IndexExprResult::deserialize(&at_least_batch).unwrap();
        assert!(round_tripped.is_at_least());
        assert!(!round_tripped.is_exact());
    }

    /// A refined `IndexExprResult` (`lower` strictly inside a non-universe
    /// `upper`) has no legacy three-shape encoding. The serializer
    /// must not error in that case — it must degrade to `AtMost(upper)` so
    /// older read planners still see a valid superset and recheck.
    #[test]
    fn test_three_variant_serialize_refined_degrades_to_at_most() {
        use lance_select::{RowAddrMask, RowAddrTreeMap};

        let lower_addrs = RowAddrTreeMap::from_iter(0u64..3);
        let upper_addrs = RowAddrTreeMap::from_iter(0u64..10);
        let refined = IndexExprResult::new(
            RowAddrMask::from_allowed(lower_addrs),
            RowAddrMask::from_allowed(upper_addrs.clone()),
        );
        assert!(!refined.is_exact() && !refined.is_at_most() && !refined.is_at_least());

        let fragments_covered = RoaringBitmap::from_iter([0u32, 1]);

        let batch = refined
            .serialize(&fragments_covered, IndexExprResultWireFormat::ThreeVariant)
            .unwrap();
        assert_eq!(
            batch.schema(),
            *IndexExprResultWireFormat::ThreeVariant.schema()
        );

        // Discriminant 1 == AtMost; the round-tripped result carries the
        // original `upper` as the AtMost mask (empty lower, upper = upper).
        let (round_tripped, round_tripped_frags) = IndexExprResult::deserialize(&batch).unwrap();
        assert!(round_tripped.is_at_most());
        assert_eq!(round_tripped.upper, RowAddrMask::from_allowed(upper_addrs));
        assert_eq!(round_tripped_frags, fragments_covered);
    }

    /// Regression test: when two JSON indices target different paths on the same
    /// column, a query against one path must be routed to its own index instead
    /// of being intercepted by whichever parser was registered first.
    #[test]
    fn test_multi_json_indices_route_by_path() {
        // Build a MultiQueryParser containing two JSON sub-parsers: one for
        // path "$.a" and one for path "$.b".
        let mut multi = MultiQueryParser::single(Box::new(JsonQueryParser::new(
            "$.a".to_string(),
            Box::new(SargableQueryParser::new(
                "json_a_idx".to_string(),
                "Json".to_string(),
                false,
            )),
        )));
        multi.add(Box::new(JsonQueryParser::new(
            "$.b".to_string(),
            Box::new(SargableQueryParser::new(
                "json_b_idx".to_string(),
                "Json".to_string(),
                false,
            )),
        )));

        let index_info = MockIndexInfoProvider::new(vec![(
            "json",
            ColInfo::with_multi(DataType::LargeBinary, Box::new(multi)),
        )]);

        // Query against path "$.b" must hit the "$.b" index.
        let expected_b = IndexedExpression::index_query(
            "json".to_string(),
            "json_b_idx".to_string(),
            "Json".to_string(),
            Arc::new(JsonQuery::new(
                Arc::new(SargableQuery::Equals(ScalarValue::Utf8(Some(
                    "foo".to_string(),
                )))),
                "$.b".to_string(),
            )),
        );
        check(
            &index_info,
            "json_extract(json, '$.b') = 'foo'",
            Some(expected_b),
            false,
        );

        // Query against path "$.a" must hit the "$.a" index.
        let expected_a = IndexedExpression::index_query(
            "json".to_string(),
            "json_a_idx".to_string(),
            "Json".to_string(),
            Arc::new(JsonQuery::new(
                Arc::new(SargableQuery::Equals(ScalarValue::Utf8(Some(
                    "foo".to_string(),
                )))),
                "$.a".to_string(),
            )),
        );
        check(
            &index_info,
            "json_extract(json, '$.a') = 'foo'",
            Some(expected_a),
            false,
        );

        // Query against an unindexed path must not bind to either index.
        check_no_index(&index_info, "json_extract(json, '$.c') = 'foo'");
    }
}
