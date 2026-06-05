// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Read-path shard pruning for MemWAL queries.
//!
//! Given a query filter and a [`ShardingSpec`], this module determines which
//! shards can be skipped because their field values cannot match the filter.
//! When the filter contains an equality (`col = lit`) or `IN` predicate on
//! the sharding column, each literal value is evaluated through the sharding
//! transform (bucket / identity / unsharded) and the resulting set of target
//! shard IDs is intersected with the available shards.

use std::collections::{HashMap, HashSet};

use arrow_schema::SchemaRef;
use datafusion::common::ScalarValue;
use datafusion::logical_expr::Operator;
use datafusion::prelude::Expr;
use lance_index::mem_wal::ShardingSpec;
use uuid::Uuid;

use super::data_source::ShardSnapshot;
use crate::dataset::mem_wal::sharding::{
    hash_scalar_to_bucket, source_column_for_field,
};

/// Attempt to prune shards based on a query filter and the sharding spec.
///
/// Returns `Some(shard_ids)` when the filter contains an equality or `IN`
/// predicate on the sharding column; only shards whose computed field values
/// match will be in the returned set. Returns `None` when pruning is not
/// possible (e.g. the filter does not reference the sharding column, or the
/// spec is unsharded).
///
/// `base_schema` is used to coerce filter literals (e.g. SQL `Int64`) to the
/// column's actual Arrow type before hashing, so the bucket id matches the
/// one stored in the shard manifest.
pub fn prune_shards(
    filter: &Expr,
    spec: &ShardingSpec,
    snapshots: &[ShardSnapshot],
    source_id_to_column: &HashMap<i32, String>,
    base_schema: Option<&SchemaRef>,
) -> Option<HashSet<Uuid>> {
    // We only prune single-field bucket/identity specs today.
    let field = spec.fields.first()?;
    let transform = field.transform.as_deref()?;
    if transform == "unsharded" {
        return None; // All rows go to shard 0; nothing to prune.
    }

    let column_name = source_column_for_field(field, source_id_to_column).ok()?;

    // Extract literal values from the filter for the sharding column.
    let literals = extract_column_literals(filter, &column_name)?;
    if literals.is_empty() {
        return None;
    }

    // Coerce literals to the column's Arrow type so the hash matches what
    // the write path stored. SQL parsing often produces Int64 for integer
    // literals even when the column is Int32.
    let coerced = coerce_literals(&literals, &column_name, base_schema);

    match transform {
        "bucket" => {
            let num_buckets: i32 = field
                .parameters
                .get("num_buckets")?
                .parse()
                .ok()?;
            if num_buckets <= 0 {
                return None;
            }

            // Compute bucket id for each literal value.
            let mut target_bucket_bytes: HashSet<Vec<u8>> = HashSet::new();
            for lit in &coerced {
                if let Some(bucket) = hash_scalar_to_bucket(lit, num_buckets) {
                    target_bucket_bytes.insert(bucket.to_le_bytes().to_vec());
                }
            }

            let field_id = &field.field_id;
            let matching: HashSet<Uuid> = snapshots
                .iter()
                .filter(|s| {
                    s.shard_field_values
                        .get(field_id)
                        .map(|v| target_bucket_bytes.contains(v))
                        .unwrap_or(true) // If no field value recorded, don't prune.
                })
                .map(|s| s.shard_id)
                .collect();
            Some(matching)
        }
        "identity" => {
            // Identity sharding: the shard field value IS the column value.
            let mut target_values: HashSet<Vec<u8>> = HashSet::new();
            for lit in &coerced {
                if let Some(bytes) = scalar_to_identity_bytes(lit) {
                    target_values.insert(bytes);
                }
            }

            let field_id = &field.field_id;
            let matching: HashSet<Uuid> = snapshots
                .iter()
                .filter(|s| {
                    s.shard_field_values
                        .get(field_id)
                        .map(|v| target_values.contains(v))
                        .unwrap_or(true)
                })
                .map(|s| s.shard_id)
                .collect();
            Some(matching)
        }
        _ => None,
    }
}

/// Extract literal values from equality (`col = lit`) or `IN (lit, ...)`
/// predicates on `column_name`. Returns `None` if the filter does not
/// reference the column in a prunable shape.
fn extract_column_literals(filter: &Expr, column_name: &str) -> Option<Vec<ScalarValue>> {
    match filter {
        // col = lit  or  lit = col
        Expr::BinaryExpr(b) if matches!(b.op, Operator::Eq) => {
            match (b.left.as_ref(), b.right.as_ref()) {
                (Expr::Column(c), Expr::Literal(lit, _))
                | (Expr::Literal(lit, _), Expr::Column(c))
                    if c.name == column_name =>
                {
                    Some(vec![lit.clone()])
                }
                _ => None,
            }
        }
        // col IN (lit, lit, ...)
        Expr::InList(in_list) if !in_list.negated => {
            let Expr::Column(c) = in_list.expr.as_ref() else {
                return None;
            };
            if c.name != column_name {
                return None;
            }
            let mut vals = Vec::with_capacity(in_list.list.len());
            for e in &in_list.list {
                let Expr::Literal(lit, _) = e else {
                    return None;
                };
                vals.push(lit.clone());
            }
            (!vals.is_empty()).then_some(vals)
        }
        // AND: recurse into both sides and union the results.
        Expr::BinaryExpr(b) if matches!(b.op, Operator::And) => {
            let left = extract_column_literals(&b.left, column_name);
            let right = extract_column_literals(&b.right, column_name);
            // For AND, if either side constrains the column, that's our match.
            // If both do, use the intersection (fewer target shards). In
            // practice, only one side of an AND constrains the same column.
            left.or(right)
        }
        _ => None,
    }
}

/// Convert a [`ScalarValue`] to its Arrow little-endian byte representation
/// for identity sharding comparison.
fn scalar_to_identity_bytes(scalar: &ScalarValue) -> Option<Vec<u8>> {
    match scalar {
        ScalarValue::Int8(Some(v)) => Some(v.to_le_bytes().to_vec()),
        ScalarValue::Int16(Some(v)) => Some(v.to_le_bytes().to_vec()),
        ScalarValue::Int32(Some(v)) => Some(v.to_le_bytes().to_vec()),
        ScalarValue::Int64(Some(v)) => Some(v.to_le_bytes().to_vec()),
        ScalarValue::UInt8(Some(v)) => Some(v.to_le_bytes().to_vec()),
        ScalarValue::UInt16(Some(v)) => Some(v.to_le_bytes().to_vec()),
        ScalarValue::UInt32(Some(v)) => Some(v.to_le_bytes().to_vec()),
        ScalarValue::UInt64(Some(v)) => Some(v.to_le_bytes().to_vec()),
        ScalarValue::Utf8(Some(v)) | ScalarValue::LargeUtf8(Some(v)) => {
            Some(v.as_bytes().to_vec())
        }
        _ => None,
    }
}

/// Coerce filter literals to match the column's Arrow type. SQL parsing
/// often produces `Int64` for integer literals even when the column is
/// `Int32`. Without coercion the Murmur3 hash would differ (Int32 vs Int64
/// code paths) and shard pruning would silently fail to match.
fn coerce_literals(
    literals: &[ScalarValue],
    column_name: &str,
    base_schema: Option<&SchemaRef>,
) -> Vec<ScalarValue> {
    let Some(schema) = base_schema else {
        return literals.to_vec();
    };
    let Ok(field) = schema.field_with_name(column_name) else {
        return literals.to_vec();
    };
    let target_type = field.data_type();
    literals
        .iter()
        .map(|lit| {
            if &lit.data_type() == target_type {
                lit.clone()
            } else {
                lit.cast_to(target_type).unwrap_or_else(|_| lit.clone())
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::{col, lit};
    use lance_index::mem_wal::ShardingField;

    fn bucket_spec(column: &str, num_buckets: i32) -> ShardingSpec {
        ShardingSpec {
            spec_id: 1,
            fields: vec![ShardingField {
                field_id: "bucket".to_string(),
                source_ids: vec![],
                transform: Some("bucket".to_string()),
                expression: None,
                result_type: "int32".to_string(),
                parameters: HashMap::from([
                    ("num_buckets".to_string(), num_buckets.to_string()),
                    ("column".to_string(), column.to_string()),
                ]),
            }],
        }
    }

    fn identity_spec(column: &str) -> ShardingSpec {
        ShardingSpec {
            spec_id: 1,
            fields: vec![ShardingField {
                field_id: "ident".to_string(),
                source_ids: vec![],
                transform: Some("identity".to_string()),
                expression: None,
                result_type: "utf8".to_string(),
                parameters: HashMap::from([("column".to_string(), column.to_string())]),
            }],
        }
    }

    fn snapshot_with_bucket(shard_id: Uuid, field_id: &str, bucket: i32) -> ShardSnapshot {
        ShardSnapshot::new(shard_id).with_shard_field_values(HashMap::from([(
            field_id.to_string(),
            bucket.to_le_bytes().to_vec(),
        )]))
    }

    fn snapshot_with_identity(shard_id: Uuid, field_id: &str, value: &str) -> ShardSnapshot {
        ShardSnapshot::new(shard_id).with_shard_field_values(HashMap::from([(
            field_id.to_string(),
            value.as_bytes().to_vec(),
        )]))
    }

    #[test]
    fn test_bucket_pruning_equality() {
        let spec = bucket_spec("region", 4);
        let shard_a = Uuid::new_v4();
        let shard_b = Uuid::new_v4();
        let shard_c = Uuid::new_v4();

        // Compute the bucket for "us-east" with 4 buckets.
        let target_bucket =
            hash_scalar_to_bucket(&ScalarValue::Utf8(Some("us-east".to_string())), 4).unwrap();

        let snapshots = vec![
            snapshot_with_bucket(shard_a, "bucket", target_bucket),
            snapshot_with_bucket(shard_b, "bucket", (target_bucket + 1) % 4),
            snapshot_with_bucket(shard_c, "bucket", target_bucket), // same bucket as a
        ];

        let filter = col("region").eq(lit("us-east"));
        let result = prune_shards(&filter, &spec, &snapshots, &HashMap::new(), None);

        let pruned = result.expect("should prune");
        assert!(pruned.contains(&shard_a));
        assert!(!pruned.contains(&shard_b));
        assert!(pruned.contains(&shard_c));
    }

    #[test]
    fn test_bucket_pruning_in_list() {
        let spec = bucket_spec("id", 8);
        let shard_a = Uuid::new_v4();
        let shard_b = Uuid::new_v4();

        let bucket_for_1 = hash_scalar_to_bucket(&ScalarValue::Int32(Some(1)), 8).unwrap();
        let bucket_for_2 = hash_scalar_to_bucket(&ScalarValue::Int32(Some(2)), 8).unwrap();

        // Make shard_a match bucket_for_1, shard_b a different bucket.
        let other_bucket = (0..8)
            .find(|b| *b != bucket_for_1 && *b != bucket_for_2)
            .unwrap();
        let snapshots = vec![
            snapshot_with_bucket(shard_a, "bucket", bucket_for_1),
            snapshot_with_bucket(shard_b, "bucket", other_bucket),
        ];

        let filter = col("id").in_list(vec![lit(1i32), lit(2i32)], false);
        let result = prune_shards(&filter, &spec, &snapshots, &HashMap::new(), None);

        let pruned = result.expect("should prune");
        assert!(pruned.contains(&shard_a));
        // shard_b has a bucket that matches neither 1 nor 2
        assert!(!pruned.contains(&shard_b));
    }

    #[test]
    fn test_identity_pruning() {
        let spec = identity_spec("tenant");
        let shard_a = Uuid::new_v4();
        let shard_b = Uuid::new_v4();

        let snapshots = vec![
            snapshot_with_identity(shard_a, "ident", "acme"),
            snapshot_with_identity(shard_b, "ident", "globex"),
        ];

        let filter = col("tenant").eq(lit("acme"));
        let result = prune_shards(&filter, &spec, &snapshots, &HashMap::new(), None);
        let pruned = result.expect("should prune");
        assert!(pruned.contains(&shard_a));
        assert!(!pruned.contains(&shard_b));
    }

    #[test]
    fn test_no_pruning_for_unsharded() {
        let spec = ShardingSpec {
            spec_id: 1,
            fields: vec![ShardingField {
                field_id: "u".to_string(),
                source_ids: vec![],
                transform: Some("unsharded".to_string()),
                expression: None,
                result_type: "int32".to_string(),
                parameters: HashMap::new(),
            }],
        };
        let filter = col("x").eq(lit(1i32));
        assert!(prune_shards(&filter, &spec, &[], &HashMap::new(), None).is_none());
    }

    #[test]
    fn test_no_pruning_for_non_sharding_column() {
        let spec = bucket_spec("region", 4);
        // Filter is on "name", not "region".
        let filter = col("name").eq(lit("foo"));
        assert!(prune_shards(&filter, &spec, &[], &HashMap::new(), None).is_none());
    }

    #[test]
    fn test_snapshot_without_field_values_not_pruned() {
        let spec = bucket_spec("id", 4);
        let shard_a = Uuid::new_v4();
        let shard_b = Uuid::new_v4();

        let bucket_for_1 = hash_scalar_to_bucket(&ScalarValue::Int32(Some(1)), 4).unwrap();
        let snapshots = vec![
            snapshot_with_bucket(shard_a, "bucket", bucket_for_1),
            ShardSnapshot::new(shard_b), // no field values -- must NOT be pruned
        ];

        let filter = col("id").eq(lit(1i32));
        let result = prune_shards(&filter, &spec, &snapshots, &HashMap::new(), None);
        let pruned = result.expect("should prune");
        assert!(pruned.contains(&shard_a));
        assert!(pruned.contains(&shard_b)); // kept because no field values
    }
}
