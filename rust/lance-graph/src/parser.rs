// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cypher query parser
//!
//! This module provides parsing functionality for Cypher queries using nom parser combinators.
//! It supports a subset of Cypher syntax focused on graph pattern matching and property access.

use crate::ast::*;
use crate::error::{GraphError, Result};
use nom::{
    branch::alt,
    bytes::complete::{tag, tag_no_case, take_while1},
    character::complete::{char, multispace0, multispace1},
    combinator::{map, opt, recognize},
    multi::{many0, separated_list0},
    sequence::{delimited, pair, preceded, tuple},
    IResult,
};
use std::collections::HashMap;

/// Parse a complete Cypher query
pub fn parse_cypher_query(input: &str) -> Result<CypherQuery> {
    let (remaining, query) = cypher_query(input).map_err(|e| GraphError::ParseError {
        message: format!("Failed to parse Cypher query: {}", e),
        position: 0,
        location: snafu::Location::new(file!(), line!(), column!()),
    })?;

    if !remaining.trim().is_empty() {
        return Err(GraphError::ParseError {
            message: format!("Unexpected input after query: {}", remaining),
            position: input.len() - remaining.len(),
            location: snafu::Location::new(file!(), line!(), column!()),
        });
    }

    Ok(query)
}

// Top-level parser for a complete Cypher query
fn cypher_query(input: &str) -> IResult<&str, CypherQuery> {
    let (input, _) = multispace0(input)?;
    let (input, match_clauses) = many0(match_clause)(input)?;
    let (input, where_clause) = opt(where_clause)(input)?;
    let (input, return_clause) = return_clause(input)?;
    let (input, order_by) = opt(order_by_clause)(input)?;
    let (input, limit) = opt(limit_clause)(input)?;
    let (input, _) = multispace0(input)?;

    Ok((
        input,
        CypherQuery {
            match_clauses,
            where_clause,
            return_clause,
            limit,
            order_by,
        },
    ))
}

// Parse a MATCH clause
fn match_clause(input: &str) -> IResult<&str, MatchClause> {
    let (input, _) = multispace0(input)?;
    let (input, _) = tag_no_case("MATCH")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, patterns) = separated_list0(comma_ws, graph_pattern)(input)?;

    Ok((input, MatchClause { patterns }))
}

// Parse a graph pattern (node or path)
fn graph_pattern(input: &str) -> IResult<&str, GraphPattern> {
    alt((
        map(path_pattern, GraphPattern::Path),
        map(node_pattern, GraphPattern::Node),
    ))(input)
}

// Parse a path pattern (only if there are segments)
fn path_pattern(input: &str) -> IResult<&str, PathPattern> {
    let (input, start_node) = node_pattern(input)?;
    let (input, segments) = many0(path_segment)(input)?;

    // Only succeed if we actually have path segments
    if segments.is_empty() {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Tag,
        )));
    }

    Ok((
        input,
        PathPattern {
            start_node,
            segments,
        },
    ))
}

// Parse a path segment (relationship + node)
fn path_segment(input: &str) -> IResult<&str, PathSegment> {
    let (input, relationship) = relationship_pattern(input)?;
    let (input, end_node) = node_pattern(input)?;

    Ok((
        input,
        PathSegment {
            relationship,
            end_node,
        },
    ))
}

// Parse a node pattern: (variable:Label {prop: value})
fn node_pattern(input: &str) -> IResult<&str, NodePattern> {
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(')(input)?;
    let (input, _) = multispace0(input)?;
    let (input, variable) = opt(identifier)(input)?;
    let (input, labels) = many0(preceded(char(':'), identifier))(input)?;
    let (input, _) = multispace0(input)?;
    let (input, properties) = opt(property_map)(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')')(input)?;

    Ok((
        input,
        NodePattern {
            variable: variable.map(|s| s.to_string()),
            labels: labels.into_iter().map(|s| s.to_string()).collect(),
            properties: properties.unwrap_or_default(),
        },
    ))
}

// Parse a relationship pattern: -[variable:TYPE {prop: value}]->
fn relationship_pattern(input: &str) -> IResult<&str, RelationshipPattern> {
    let (input, _) = multispace0(input)?;

    // Parse direction and bracket content
    let (input, (direction, content)) = alt((
        // Outgoing: -[...]->
        map(
            tuple((
                char('-'),
                delimited(char('['), relationship_content, char(']')),
                tag("->"),
            )),
            |(_, content, _)| (RelationshipDirection::Outgoing, content),
        ),
        // Incoming: <-[...]-
        map(
            tuple((
                tag("<-"),
                delimited(char('['), relationship_content, char(']')),
                char('-'),
            )),
            |(_, content, _)| (RelationshipDirection::Incoming, content),
        ),
        // Undirected: -[...]-
        map(
            tuple((
                char('-'),
                delimited(char('['), relationship_content, char(']')),
                char('-'),
            )),
            |(_, content, _)| (RelationshipDirection::Undirected, content),
        ),
    ))(input)?;

    let (variable, types, properties, length) = content;

    Ok((
        input,
        RelationshipPattern {
            variable: variable.map(|s| s.to_string()),
            types: types.into_iter().map(|s| s.to_string()).collect(),
            direction,
            properties: properties.unwrap_or_default(),
            length,
        },
    ))
}

// Type alias for complex relationship content return type
type RelationshipContentResult<'a> = (
    Option<&'a str>,
    Vec<&'a str>,
    Option<HashMap<String, PropertyValue>>,
    Option<LengthRange>,
);

// Parse relationship content inside brackets
fn relationship_content(input: &str) -> IResult<&str, RelationshipContentResult<'_>> {
    let (input, _) = multispace0(input)?;
    let (input, variable) = opt(identifier)(input)?;
    let (input, types) = many0(preceded(char(':'), identifier))(input)?;
    let (input, _) = multispace0(input)?;
    let (input, length) = opt(length_range)(input)?;
    let (input, _) = multispace0(input)?;
    let (input, properties) = opt(property_map)(input)?;
    let (input, _) = multispace0(input)?;

    Ok((input, (variable, types, properties, length)))
}

// Parse a property map: {key: value, key2: value2}
fn property_map(input: &str) -> IResult<&str, HashMap<String, PropertyValue>> {
    let (input, _) = multispace0(input)?;
    let (input, _) = char('{')(input)?;
    let (input, _) = multispace0(input)?;
    let (input, pairs) = separated_list0(comma_ws, property_pair)(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('}')(input)?;

    Ok((input, pairs.into_iter().collect()))
}

// Parse a property key-value pair
fn property_pair(input: &str) -> IResult<&str, (String, PropertyValue)> {
    let (input, _) = multispace0(input)?;
    let (input, key) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(':')(input)?;
    let (input, _) = multispace0(input)?;
    let (input, value) = property_value(input)?;

    Ok((input, (key.to_string(), value)))
}

// Parse a property value
fn property_value(input: &str) -> IResult<&str, PropertyValue> {
    alt((
        map(string_literal, PropertyValue::String),
        map(integer_literal, PropertyValue::Integer),
        map(float_literal, PropertyValue::Float),
        map(boolean_literal, PropertyValue::Boolean),
        map(tag("null"), |_| PropertyValue::Null),
        map(parameter, PropertyValue::Parameter),
    ))(input)
}

// Parse a WHERE clause
fn where_clause(input: &str) -> IResult<&str, WhereClause> {
    let (input, _) = multispace0(input)?;
    let (input, _) = tag_no_case("WHERE")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, expression) = boolean_expression(input)?;

    Ok((input, WhereClause { expression }))
}

// Parse a boolean expression (simplified)
fn boolean_expression(input: &str) -> IResult<&str, BooleanExpression> {
    // For now, just parse simple comparisons
    let (input, left) = value_expression(input)?;
    let (input, _) = multispace0(input)?;
    let (input, operator) = comparison_operator(input)?;
    let (input, _) = multispace0(input)?;
    let (input, right) = value_expression(input)?;

    Ok((
        input,
        BooleanExpression::Comparison {
            left,
            operator,
            right,
        },
    ))
}

// Parse a comparison operator
fn comparison_operator(input: &str) -> IResult<&str, ComparisonOperator> {
    alt((
        map(tag("="), |_| ComparisonOperator::Equal),
        map(tag("<>"), |_| ComparisonOperator::NotEqual),
        map(tag("!="), |_| ComparisonOperator::NotEqual),
        map(tag("<="), |_| ComparisonOperator::LessThanOrEqual),
        map(tag(">="), |_| ComparisonOperator::GreaterThanOrEqual),
        map(tag("<"), |_| ComparisonOperator::LessThan),
        map(tag(">"), |_| ComparisonOperator::GreaterThan),
    ))(input)
}

// Parse a value expression
fn value_expression(input: &str) -> IResult<&str, ValueExpression> {
    alt((
        map(property_reference, ValueExpression::Property),
        map(property_value, ValueExpression::Literal),
        map(identifier, |id| ValueExpression::Variable(id.to_string())),
    ))(input)
}

// Parse a property reference: variable.property
fn property_reference(input: &str) -> IResult<&str, PropertyRef> {
    let (input, variable) = identifier(input)?;
    let (input, _) = char('.')(input)?;
    let (input, property) = identifier(input)?;

    Ok((
        input,
        PropertyRef {
            variable: variable.to_string(),
            property: property.to_string(),
        },
    ))
}

// Parse a RETURN clause
fn return_clause(input: &str) -> IResult<&str, ReturnClause> {
    let (input, _) = multispace0(input)?;
    let (input, _) = tag_no_case("RETURN")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, distinct) = opt(tag_no_case("DISTINCT"))(input)?;
    let (input, _) = if distinct.is_some() {
        multispace1(input)?
    } else {
        (input, "")
    };
    let (input, items) = separated_list0(comma_ws, return_item)(input)?;

    Ok((
        input,
        ReturnClause {
            distinct: distinct.is_some(),
            items,
        },
    ))
}

// Parse a return item
fn return_item(input: &str) -> IResult<&str, ReturnItem> {
    let (input, expression) = value_expression(input)?;
    let (input, _) = multispace0(input)?;
    let (input, alias) = opt(preceded(
        tuple((tag_no_case("AS"), multispace1)),
        identifier,
    ))(input)?;

    Ok((
        input,
        ReturnItem {
            expression,
            alias: alias.map(|s| s.to_string()),
        },
    ))
}

// Parse an ORDER BY clause
fn order_by_clause(input: &str) -> IResult<&str, OrderByClause> {
    let (input, _) = multispace0(input)?;
    let (input, _) = tag_no_case("ORDER")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, _) = tag_no_case("BY")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, items) = separated_list0(comma_ws, order_by_item)(input)?;

    Ok((input, OrderByClause { items }))
}

// Parse an order by item
fn order_by_item(input: &str) -> IResult<&str, OrderByItem> {
    let (input, expression) = value_expression(input)?;
    let (input, _) = multispace0(input)?;
    let (input, direction) = opt(alt((
        map(tag_no_case("ASC"), |_| SortDirection::Ascending),
        map(tag_no_case("DESC"), |_| SortDirection::Descending),
    )))(input)?;

    Ok((
        input,
        OrderByItem {
            expression,
            direction: direction.unwrap_or(SortDirection::Ascending),
        },
    ))
}

// Parse a LIMIT clause
fn limit_clause(input: &str) -> IResult<&str, u64> {
    let (input, _) = multispace0(input)?;
    let (input, _) = tag_no_case("LIMIT")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, limit) = integer_literal(input)?;

    Ok((input, limit as u64))
}

// Helper parsers

// Parse an identifier
fn identifier(input: &str) -> IResult<&str, &str> {
    take_while1(|c: char| c.is_alphanumeric() || c == '_')(input)
}

// Parse a string literal
fn string_literal(input: &str) -> IResult<&str, String> {
    let (input, _) = char('"')(input)?;
    let (input, content) = take_while1(|c| c != '"')(input)?;
    let (input, _) = char('"')(input)?;
    Ok((input, content.to_string()))
}

// Parse an integer literal
fn integer_literal(input: &str) -> IResult<&str, i64> {
    let (input, digits) = recognize(pair(
        opt(char('-')),
        take_while1(|c: char| c.is_ascii_digit()),
    ))(input)?;

    Ok((input, digits.parse().unwrap()))
}

// Parse a float literal
fn float_literal(input: &str) -> IResult<&str, f64> {
    let (input, number) = recognize(tuple((
        opt(char('-')),
        take_while1(|c: char| c.is_ascii_digit()),
        char('.'),
        take_while1(|c: char| c.is_ascii_digit()),
    )))(input)?;

    Ok((input, number.parse().unwrap()))
}

// Parse a boolean literal
fn boolean_literal(input: &str) -> IResult<&str, bool> {
    alt((
        map(tag_no_case("true"), |_| true),
        map(tag_no_case("false"), |_| false),
    ))(input)
}

// Parse a parameter reference
fn parameter(input: &str) -> IResult<&str, String> {
    let (input, _) = char('$')(input)?;
    let (input, name) = identifier(input)?;
    Ok((input, name.to_string()))
}

// Parse comma with optional whitespace
fn comma_ws(input: &str) -> IResult<&str, ()> {
    let (input, _) = multispace0(input)?;
    let (input, _) = char(',')(input)?;
    let (input, _) = multispace0(input)?;
    Ok((input, ()))
}

// Parse variable-length path syntax: *1..2, *..3, *2.., *
fn length_range(input: &str) -> IResult<&str, LengthRange> {
    let (input, _) = char('*')(input)?;
    let (input, _) = multispace0(input)?;

    // Parse different length patterns
    alt((
        // *min..max (e.g., *1..3)
        map(
            tuple((
                nom::character::complete::u32,
                tag(".."),
                nom::character::complete::u32,
            )),
            |(min, _, max)| LengthRange {
                min: Some(min),
                max: Some(max),
            },
        ),
        // *..max (e.g., *..3)
        map(preceded(tag(".."), nom::character::complete::u32), |max| {
            LengthRange {
                min: None,
                max: Some(max),
            }
        }),
        // *min.. (e.g., *2..)
        map(
            tuple((nom::character::complete::u32, tag(".."))),
            |(min, _)| LengthRange {
                min: Some(min),
                max: None,
            },
        ),
        // *min (e.g., *2)
        map(nom::character::complete::u32, |min| LengthRange {
            min: Some(min),
            max: Some(min),
        }),
        // * (unlimited)
        map(multispace0, |_| LengthRange {
            min: None,
            max: None,
        }),
    ))(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_node_query() {
        let query = "MATCH (n:Person) RETURN n.name";
        let result = parse_cypher_query(query).unwrap();

        assert_eq!(result.match_clauses.len(), 1);
        assert_eq!(result.return_clause.items.len(), 1);
    }

    #[test]
    fn test_parse_node_with_properties() {
        let query = r#"MATCH (n:Person {name: "John", age: 30}) RETURN n"#;
        let result = parse_cypher_query(query).unwrap();

        if let GraphPattern::Node(node) = &result.match_clauses[0].patterns[0] {
            assert_eq!(node.labels, vec!["Person"]);
            assert_eq!(node.properties.len(), 2);
        } else {
            panic!("Expected node pattern");
        }
    }

    #[test]
    fn test_parse_simple_relationship_query() {
        let query = "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.name, b.name";
        let result = parse_cypher_query(query).unwrap();

        assert_eq!(result.match_clauses.len(), 1);
        assert_eq!(result.return_clause.items.len(), 2);

        if let GraphPattern::Path(path) = &result.match_clauses[0].patterns[0] {
            assert_eq!(path.segments.len(), 1);
            assert_eq!(path.segments[0].relationship.types, vec!["KNOWS"]);
        } else {
            panic!("Expected path pattern");
        }
    }

    #[test]
    fn test_parse_variable_length_path() {
        let query = "MATCH (a:Person)-[:FRIEND_OF*1..2]-(b:Person) RETURN a.name, b.name";
        let result = parse_cypher_query(query).unwrap();

        assert_eq!(result.match_clauses.len(), 1);

        if let GraphPattern::Path(path) = &result.match_clauses[0].patterns[0] {
            assert_eq!(path.segments.len(), 1);
            assert_eq!(path.segments[0].relationship.types, vec!["FRIEND_OF"]);

            let length = path.segments[0].relationship.length.as_ref().unwrap();
            assert_eq!(length.min, Some(1));
            assert_eq!(length.max, Some(2));
        } else {
            panic!("Expected path pattern");
        }
    }

    #[test]
    fn test_parse_query_with_where_clause() {
        let query = "MATCH (n:Person) WHERE n.age > 30 RETURN n.name";
        let result = parse_cypher_query(query).unwrap();

        assert!(result.where_clause.is_some());
    }

    #[test]
    fn test_parse_query_with_limit() {
        let query = "MATCH (n:Person) RETURN n.name LIMIT 10";
        let result = parse_cypher_query(query).unwrap();

        assert_eq!(result.limit, Some(10));
    }
}
