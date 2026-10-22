// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fragment-local planning and execution of encoded primitive predicates.

use super::{
    FilteredReadGlobalMetrics, FilteredReadStream, RangeMetricsTracker, ScopedFragmentRead,
};
use crate::dataset::fragment::FragmentReader;
use arrow_array::BooleanArray;
use datafusion_expr::{Expr, Operator};
use futures::StreamExt;
use lance_core::{Error, Result, datatypes::OnMissing};
use lance_encoding::{
    decoder::FilterExpression,
    predicate::{
        ComparisonOperator, PrimitiveLiteral, PrimitivePredicate, PrimitivePredicateEncoding,
        PrimitivePredicateOutcome, PrimitivePredicatePlan,
    },
};
use std::{ops::Range, sync::Arc};

pub(super) fn encoded_primitive_predicate(expr: &Expr) -> Option<PrimitivePredicate> {
    let Expr::BinaryExpr(binary) = expr else {
        return None;
    };
    let (column, literal, operator) = match (binary.left.as_ref(), binary.right.as_ref()) {
        (Expr::Column(column), Expr::Literal(literal, _)) => (&column.name, literal, binary.op),
        (Expr::Literal(literal, _), Expr::Column(column)) => {
            (&column.name, literal, reverse_operator(binary.op)?)
        }
        _ => return None,
    };
    let operator = match operator {
        Operator::Eq => ComparisonOperator::Equal,
        Operator::NotEq => ComparisonOperator::NotEqual,
        Operator::Lt => ComparisonOperator::LessThan,
        Operator::LtEq => ComparisonOperator::LessThanOrEqual,
        Operator::Gt => ComparisonOperator::GreaterThan,
        Operator::GtEq => ComparisonOperator::GreaterThanOrEqual,
        _ => return None,
    };
    let literal = match literal {
        datafusion::scalar::ScalarValue::Int32(Some(value)) => PrimitiveLiteral::Int32(*value),
        datafusion::scalar::ScalarValue::UInt32(Some(value)) => PrimitiveLiteral::UInt32(*value),
        _ => return None,
    };
    Some(PrimitivePredicate {
        column: column.clone(),
        operator,
        literal,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EncodedPredicateMode {
    Auto,
    Disabled,
    Forced,
}

pub(super) fn encoded_predicate_mode() -> EncodedPredicateMode {
    encoded_predicate_mode_from_value(
        std::env::var("LANCE_ENCODED_PRIMITIVE_PREDICATE")
            .ok()
            .as_deref(),
    )
}

fn encoded_predicate_mode_from_value(value: Option<&str>) -> EncodedPredicateMode {
    match value.unwrap_or("auto").to_ascii_lowercase().as_str() {
        "0" | "false" | "off" | "disabled" => EncodedPredicateMode::Disabled,
        "1" | "true" | "force" | "forced" => EncodedPredicateMode::Forced,
        "auto" => EncodedPredicateMode::Auto,
        value => {
            tracing::warn!(
                value,
                "Unrecognized LANCE_ENCODED_PRIMITIVE_PREDICATE value; using auto"
            );
            EncodedPredicateMode::Auto
        }
    }
}

const ENCODED_PREDICATE_AUTO_SAMPLE_ROWS: u64 = 32 * 1024;
const ENCODED_PREDICATE_AUTO_SAMPLE_WINDOWS: u64 = 2;
const ENCODED_PREDICATE_RANGE_AMPLIFICATION_ROWS: u64 = 1024;
const ENCODED_PREDICATE_MAX_SELECTED_RANGES: usize = 4096;

pub(super) fn total_range_rows(ranges: &[Range<u64>]) -> u64 {
    ranges.iter().map(|range| range.end - range.start).sum()
}

fn ordinal_range_slice(ranges: &[Range<u64>], mut offset: u64, mut length: u64) -> Vec<Range<u64>> {
    let mut selected = Vec::new();
    for range in ranges {
        let range_len = range.end - range.start;
        if offset >= range_len {
            offset -= range_len;
            continue;
        }
        let start = range.start + offset;
        let take = length.min(range.end - start);
        selected.push(start..start + take);
        length -= take;
        offset = 0;
        if length == 0 {
            break;
        }
    }
    selected
}

fn encoded_predicate_sample_ranges(ranges: &[Range<u64>]) -> Vec<Range<u64>> {
    let total_rows = total_range_rows(ranges);
    if total_rows <= ENCODED_PREDICATE_AUTO_SAMPLE_ROWS {
        return ranges.to_vec();
    }

    let window_rows =
        ENCODED_PREDICATE_AUTO_SAMPLE_ROWS.div_ceil(ENCODED_PREDICATE_AUTO_SAMPLE_WINDOWS);
    let max_start = total_rows - window_rows;
    let mut sampled = Vec::new();
    for window in 0..ENCODED_PREDICATE_AUTO_SAMPLE_WINDOWS {
        let offset = window * max_start / (ENCODED_PREDICATE_AUTO_SAMPLE_WINDOWS - 1);
        sampled.extend(ordinal_range_slice(ranges, offset, window_rows));
    }
    sampled.sort_by_key(|range| range.start);
    let mut merged: Vec<Range<u64>> = Vec::with_capacity(sampled.len());
    for range in sampled {
        if let Some(previous) = merged.last_mut()
            && range.start <= previous.end
        {
            previous.end = previous.end.max(range.end);
        } else {
            merged.push(range);
        }
    }
    merged
}

fn encoded_predicate_auto_accepts(
    encoding: PrimitivePredicateEncoding,
    sampled_rows: u64,
    selected_rows: u64,
    selected_ranges: usize,
) -> bool {
    if sampled_rows == 0 || selected_rows == 0 {
        return true;
    }
    let predicted_read_rows = encoded_predicate_predicted_read_rows(selected_rows, selected_ranges);
    encoded_predicate_maximum_read_rows(encoding, sampled_rows)
        .is_some_and(|maximum_read_rows| predicted_read_rows <= maximum_read_rows)
}

fn encoded_predicate_predicted_read_rows(selected_rows: u64, selected_ranges: usize) -> u64 {
    selected_rows.saturating_add(
        (selected_ranges as u64).saturating_mul(ENCODED_PREDICATE_RANGE_AMPLIFICATION_ROWS),
    )
}

fn encoded_predicate_maximum_read_rows(
    encoding: PrimitivePredicateEncoding,
    total_rows: u64,
) -> Option<u64> {
    match encoding {
        PrimitivePredicateEncoding::Constant => None,
        PrimitivePredicateEncoding::Rle => Some(total_rows.saturating_mul(4) / 5),
        PrimitivePredicateEncoding::Bitpacked => Some(total_rows.saturating_mul(11) / 20),
    }
}

fn encoded_predicate_selection_exceeds_limit(
    selected_rows: u64,
    selected_ranges: usize,
    limit: EncodedPredicateRangeLimit,
) -> bool {
    selected_ranges > limit.maximum_ranges
        || encoded_predicate_predicted_read_rows(selected_rows, selected_ranges)
            > limit.maximum_read_rows
}

fn reverse_operator(operator: Operator) -> Option<Operator> {
    match operator {
        Operator::Eq | Operator::NotEq => Some(operator),
        Operator::Lt => Some(Operator::Gt),
        Operator::LtEq => Some(Operator::GtEq),
        Operator::Gt => Some(Operator::Lt),
        Operator::GtEq => Some(Operator::LtEq),
        _ => None,
    }
}

pub(super) struct EncodedPredicateSelection {
    pub(super) ranges: Vec<Range<u64>>,
    pub(super) selected_rows: u64,
}

enum EncodedPredicateRangeEvaluation {
    Selection(Vec<Range<u64>>),
    AmplificationLimitExceeded,
}

#[derive(Clone, Copy)]
struct EncodedPredicateRangeLimit {
    maximum_read_rows: u64,
    maximum_ranges: usize,
}

impl FilteredReadStream {
    async fn evaluate_encoded_predicate_ranges(
        predicate_reader: &FragmentReader,
        ranges: Vec<Range<u64>>,
        batch_size: u32,
        predicate_schema: Arc<lance_core::datatypes::Schema>,
        predicate: &PrimitivePredicate,
        limit: Option<EncodedPredicateRangeLimit>,
        metrics: &FilteredReadGlobalMetrics,
    ) -> Result<Option<EncodedPredicateRangeEvaluation>> {
        let filter = FilterExpression::primitive(predicate)?;
        let mask_stream = predicate_reader
            .read_ranges_predicate(ranges.clone().into(), batch_size, predicate_schema, filter)
            .await;
        let Some(mut mask_stream) = (match mask_stream {
            Ok(stream) => stream,
            Err(Error::NotSupported { .. }) => return Ok(None),
            Err(error) => return Err(error),
        }) else {
            return Ok(None);
        };

        let mut range_tracker = RangeMetricsTracker::new(ranges.clone());
        let mut physical_rows = ranges.iter().flat_map(|range| range.clone());
        let mut selected = Vec::<Range<u64>>::new();
        let mut selected_rows = 0_u64;
        while let Some(batch) = mask_stream.next().await {
            let batch = match batch.await {
                Ok(batch) => batch,
                Err(Error::NotSupported { .. }) => return Ok(None),
                Err(error) => return Err(error),
            };
            metrics.encoded_predicate_rows_scanned.add(batch.num_rows());
            metrics
                .encoded_predicate_ranges_scanned
                .add(range_tracker.incremental_ranges_scanned(batch.num_rows()));
            let mask = batch
                .column(0)
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    Error::internal("Encoded predicate did not produce a Boolean mask")
                })?;
            for matches in mask.iter() {
                let row = physical_rows.next().ok_or_else(|| {
                    Error::internal("Encoded predicate produced more rows than requested")
                })?;
                if matches == Some(true) {
                    let row_end = row.checked_add(1).ok_or_else(|| {
                        Error::internal("Encoded predicate row offset exceeds u64::MAX")
                    })?;
                    if let Some(previous) = selected.last_mut()
                        && previous.end == row
                    {
                        previous.end = row_end;
                    } else {
                        selected.push(row..row_end);
                    }
                    selected_rows += 1;
                    if let Some(limit) = limit
                        && encoded_predicate_selection_exceeds_limit(
                            selected_rows,
                            selected.len(),
                            limit,
                        )
                    {
                        return Ok(Some(
                            EncodedPredicateRangeEvaluation::AmplificationLimitExceeded,
                        ));
                    }
                }
            }
        }
        if physical_rows.next().is_some() {
            return Err(Error::internal(
                "Encoded predicate produced fewer rows than requested",
            ));
        }
        Ok(Some(EncodedPredicateRangeEvaluation::Selection(selected)))
    }

    pub(super) async fn try_encoded_predicate_ranges(
        fragment_read_task: &ScopedFragmentRead,
        predicate: &PrimitivePredicate,
        mode: EncodedPredicateMode,
        global_metrics: &FilteredReadGlobalMetrics,
    ) -> Result<Option<EncodedPredicateSelection>> {
        if predicate.column.contains('.') {
            return Ok(None);
        }
        // Predicate range reads bypass deletion handling. Deleted rows must instead
        // pass through the ordinary filter, where their predicate values become null.
        if fragment_read_task.with_deleted_rows
            && fragment_read_task
                .fragment
                .get_deletion_vector()
                .await?
                .is_some_and(|deletions| !deletions.is_empty())
        {
            return Ok(None);
        }
        global_metrics.encoded_predicate_attempts.add(1);
        let predicate_projection = fragment_read_task
            .projection
            .as_ref()
            .clone()
            .union_column(&predicate.column, OnMissing::Error)?;
        let predicate_schema = Arc::new(
            predicate_projection
                .to_bare_schema()
                .project(&[predicate.column.as_str()])?,
        );
        let mut predicate_reader = fragment_read_task
            .fragment
            .open(&predicate_schema, fragment_read_task.frag_read_config())
            .await?;
        if fragment_read_task.with_deleted_rows {
            predicate_reader.with_make_deletions_null();
        }
        let plan = match predicate_reader
            .primitive_predicate_plan(predicate_schema.clone(), predicate.clone())
            .await
        {
            Ok(plan) => plan,
            Err(Error::NotSupported { .. }) => None,
            Err(error) => return Err(error),
        };
        if let Some(PrimitivePredicatePlan { outcome, .. }) = plan {
            match outcome {
                PrimitivePredicateOutcome::AlwaysFalse => {
                    global_metrics.encoded_predicate_applied.add(1);
                    return Ok(Some(EncodedPredicateSelection {
                        ranges: Vec::new(),
                        selected_rows: 0,
                    }));
                }
                // Deleted rows become null in the predicate projection and therefore do not
                // match, even when every stored value does.
                PrimitivePredicateOutcome::AlwaysTrue if !fragment_read_task.with_deleted_rows => {
                    let selected_rows = total_range_rows(&fragment_read_task.ranges);
                    global_metrics.encoded_predicate_applied.add(1);
                    global_metrics
                        .encoded_predicate_selected_rows
                        .add(selected_rows as usize);
                    global_metrics
                        .encoded_predicate_selected_ranges
                        .add(fragment_read_task.ranges.len());
                    return Ok(Some(EncodedPredicateSelection {
                        ranges: fragment_read_task.ranges.clone(),
                        selected_rows,
                    }));
                }
                PrimitivePredicateOutcome::AlwaysTrue | PrimitivePredicateOutcome::Unknown => {}
            }
        }

        if mode == EncodedPredicateMode::Auto && plan.is_none() {
            global_metrics.encoded_predicate_rejected.add(1);
            global_metrics
                .io_metrics
                .record(&fragment_read_task.scan_scheduler);
            return Ok(None);
        }

        let selected_ranges = if mode == EncodedPredicateMode::Auto {
            let plan = plan.expect("auto mode requires a metadata plan");
            if plan.encoding == PrimitivePredicateEncoding::Constant {
                global_metrics.encoded_predicate_rejected.add(1);
                global_metrics
                    .io_metrics
                    .record(&fragment_read_task.scan_scheduler);
                return Ok(None);
            }

            let sample_ranges = encoded_predicate_sample_ranges(&fragment_read_task.ranges);
            let sampled_rows = total_range_rows(&sample_ranges);
            global_metrics
                .encoded_predicate_sample_rows
                .add(sampled_rows as usize);
            let Some(sample_evaluation) = Self::evaluate_encoded_predicate_ranges(
                &predicate_reader,
                sample_ranges,
                fragment_read_task.batch_size,
                predicate_schema.clone(),
                predicate,
                None,
                global_metrics,
            )
            .await?
            else {
                global_metrics.encoded_predicate_rejected.add(1);
                global_metrics
                    .io_metrics
                    .record(&fragment_read_task.scan_scheduler);
                return Ok(None);
            };
            let EncodedPredicateRangeEvaluation::Selection(sample_selection) = sample_evaluation
            else {
                unreachable!("sample evaluation does not have an amplification limit");
            };
            let sample_selected_rows = total_range_rows(&sample_selection);
            global_metrics
                .encoded_predicate_sample_selected_rows
                .add(sample_selected_rows as usize);
            global_metrics
                .encoded_predicate_sample_selected_ranges
                .add(sample_selection.len());
            global_metrics.encoded_predicate_predicted_read_rows.add(
                encoded_predicate_predicted_read_rows(sample_selected_rows, sample_selection.len())
                    as usize,
            );
            if !encoded_predicate_auto_accepts(
                plan.encoding,
                sampled_rows,
                sample_selected_rows,
                sample_selection.len(),
            ) {
                global_metrics.encoded_predicate_rejected.add(1);
                global_metrics
                    .io_metrics
                    .record(&fragment_read_task.scan_scheduler);
                return Ok(None);
            }

            let fragment_rows = total_range_rows(&fragment_read_task.ranges);
            let full_evaluation_limit = EncodedPredicateRangeLimit {
                maximum_read_rows: encoded_predicate_maximum_read_rows(
                    plan.encoding,
                    fragment_rows,
                )
                .expect("constant encoding was rejected before full evaluation"),
                maximum_ranges: ENCODED_PREDICATE_MAX_SELECTED_RANGES,
            };
            let Some(full_evaluation) = Self::evaluate_encoded_predicate_ranges(
                &predicate_reader,
                fragment_read_task.ranges.clone(),
                fragment_read_task.batch_size,
                predicate_schema,
                predicate,
                Some(full_evaluation_limit),
                global_metrics,
            )
            .await?
            else {
                global_metrics.encoded_predicate_rejected.add(1);
                global_metrics
                    .io_metrics
                    .record(&fragment_read_task.scan_scheduler);
                return Ok(None);
            };
            let EncodedPredicateRangeEvaluation::Selection(full_selection) = full_evaluation else {
                global_metrics.encoded_predicate_rejected.add(1);
                global_metrics
                    .io_metrics
                    .record(&fragment_read_task.scan_scheduler);
                return Ok(None);
            };
            full_selection
        } else {
            let Some(evaluation) = Self::evaluate_encoded_predicate_ranges(
                &predicate_reader,
                fragment_read_task.ranges.clone(),
                fragment_read_task.batch_size,
                predicate_schema,
                predicate,
                None,
                global_metrics,
            )
            .await?
            else {
                global_metrics.encoded_predicate_rejected.add(1);
                global_metrics
                    .io_metrics
                    .record(&fragment_read_task.scan_scheduler);
                return Ok(None);
            };
            let EncodedPredicateRangeEvaluation::Selection(selection) = evaluation else {
                unreachable!("forced evaluation does not have an amplification limit");
            };
            selection
        };

        let selected_rows = total_range_rows(&selected_ranges);
        global_metrics.encoded_predicate_applied.add(1);
        global_metrics
            .encoded_predicate_selected_rows
            .add(selected_rows as usize);
        global_metrics
            .encoded_predicate_selected_ranges
            .add(selected_ranges.len());
        global_metrics
            .io_metrics
            .record(&fragment_read_task.scan_scheduler);
        Ok(Some(EncodedPredicateSelection {
            ranges: selected_ranges,
            selected_rows,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{FilteredReadExec, FilteredReadOptions};
    use super::*;
    use crate::Dataset;
    use crate::dataset::WriteParams;
    use arrow_array::cast::AsArray;
    use arrow_array::{
        Int32Array, RecordBatch, RecordBatchIterator, UInt32Array, UInt64Array, types::UInt64Type,
    };
    use arrow_schema::Schema as ArrowSchema;
    use arrow_select::concat::concat_batches;
    use datafusion::execution::TaskContext;
    use datafusion::physical_plan::metrics::MetricsSet;
    use datafusion::physical_plan::{ExecutionPlan, metrics::ExecutionPlanMetricsSet};
    use futures::TryStreamExt;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datafusion::planner::Planner;
    use rstest::rstest;
    #[rstest]
    #[case::eq(Operator::Eq, ComparisonOperator::Equal)]
    #[case::ne(Operator::NotEq, ComparisonOperator::NotEqual)]
    #[case::lt(Operator::Lt, ComparisonOperator::GreaterThan)]
    #[case::le(Operator::LtEq, ComparisonOperator::GreaterThanOrEqual)]
    #[case::gt(Operator::Gt, ComparisonOperator::LessThan)]
    #[case::ge(Operator::GtEq, ComparisonOperator::LessThanOrEqual)]
    fn test_encoded_predicate_literal_on_left(
        #[case] operator: Operator,
        #[case] reversed: ComparisonOperator,
    ) {
        let expr = datafusion_expr::binary_expr(
            datafusion_expr::lit(-2_i32),
            operator,
            datafusion_expr::col("predicate"),
        );
        assert_eq!(
            encoded_primitive_predicate(&expr),
            Some(PrimitivePredicate {
                column: "predicate".into(),
                operator: reversed,
                literal: PrimitiveLiteral::Int32(-2),
            })
        );
    }

    #[test]
    fn test_encoded_predicate_modes() {
        assert_eq!(
            encoded_predicate_mode_from_value(None),
            EncodedPredicateMode::Auto
        );
        assert_eq!(
            encoded_predicate_mode_from_value(Some("auto")),
            EncodedPredicateMode::Auto
        );
        assert_eq!(
            encoded_predicate_mode_from_value(Some("0")),
            EncodedPredicateMode::Disabled
        );
        assert_eq!(
            encoded_predicate_mode_from_value(Some("false")),
            EncodedPredicateMode::Disabled
        );
        assert_eq!(
            encoded_predicate_mode_from_value(Some("1")),
            EncodedPredicateMode::Forced
        );
        assert_eq!(
            encoded_predicate_mode_from_value(Some("forced")),
            EncodedPredicateMode::Forced
        );
    }

    #[test]
    fn test_encoded_predicate_auto_gate() {
        assert!(encoded_predicate_auto_accepts(
            PrimitivePredicateEncoding::Rle,
            65_536,
            32_768,
            8,
        ));
        assert!(!encoded_predicate_auto_accepts(
            PrimitivePredicateEncoding::Rle,
            65_536,
            60_000,
            8,
        ));
        assert!(encoded_predicate_auto_accepts(
            PrimitivePredicateEncoding::Bitpacked,
            65_536,
            8_192,
            8,
        ));
        assert!(!encoded_predicate_auto_accepts(
            PrimitivePredicateEncoding::Bitpacked,
            65_536,
            64,
            64,
        ));
        assert!(encoded_predicate_auto_accepts(
            PrimitivePredicateEncoding::Bitpacked,
            65_536,
            0,
            0,
        ));
        assert!(!encoded_predicate_auto_accepts(
            PrimitivePredicateEncoding::Constant,
            65_536,
            1,
            1,
        ));

        let limit = EncodedPredicateRangeLimit {
            maximum_read_rows: 10_000,
            maximum_ranges: 8,
        };
        assert!(!encoded_predicate_selection_exceeds_limit(1_808, 8, limit));
        assert!(encoded_predicate_selection_exceeds_limit(1_809, 8, limit));
        assert!(encoded_predicate_selection_exceeds_limit(1, 9, limit));
    }

    #[test]
    fn test_encoded_predicate_stratified_sample() {
        let ranges = vec![0..50_000, 60_000..210_000];
        let sampled = encoded_predicate_sample_ranges(&ranges);
        assert_eq!(total_range_rows(&sampled), 32_768);
        assert_eq!(sampled.first().unwrap().start, 0);
        assert_eq!(sampled.last().unwrap().end, 210_000);
        assert!(sampled.windows(2).all(|pair| pair[0].end < pair[1].start));
    }

    async fn execute_encoded_predicate_test_scan(
        dataset: Arc<Dataset>,
        filter: &str,
    ) -> (usize, MetricsSet) {
        let projection = dataset
            .empty_projection()
            .union_column("payload", OnMissing::Error)
            .unwrap();
        let planner = Planner::new(Arc::new(ArrowSchema::from(dataset.schema())));
        let filter = planner.parse_filter(filter).unwrap();
        let options = FilteredReadOptions::new(projection)
            .with_filter(Some(filter.clone()), Some(filter))
            .unwrap();
        let plan = Arc::new(FilteredReadExec::try_new(dataset, options, None).unwrap());
        let batches = plan
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let output_rows = batches.iter().map(RecordBatch::num_rows).sum();
        (output_rows, plan.metrics().unwrap())
    }

    fn metric_value(metrics: &MetricsSet, name: &str) -> usize {
        metrics
            .sum_by_name(name)
            .map(|value| value.as_usize())
            .unwrap_or(0)
    }

    async fn execute_mode_scan(
        dataset: Arc<Dataset>,
        filter: &str,
        mode: EncodedPredicateMode,
        with_deleted_rows: bool,
        expected_encoding: PrimitivePredicateEncoding,
    ) -> (Vec<u64>, Arc<FilteredReadGlobalMetrics>) {
        let projection = dataset
            .empty_projection()
            .union_column("payload", OnMissing::Error)
            .unwrap();
        let planner = Planner::new(Arc::new(ArrowSchema::from(dataset.schema())));
        let filter = planner
            .optimize_expr(planner.parse_filter(filter).unwrap())
            .unwrap();
        let mut options = FilteredReadOptions::new(projection)
            .with_filter(Some(filter.clone()), Some(filter))
            .unwrap();
        options.batch_size = Some(333);
        if with_deleted_rows {
            options = options.with_deleted_rows().unwrap();
        }
        let read = FilteredReadExec::try_new(dataset.clone(), options.clone(), None).unwrap();
        let plan = read
            .get_or_create_plan(Arc::new(TaskContext::default()))
            .await
            .unwrap();
        let scoped = FilteredReadStream::plan_to_scoped_fragments(
            &plan,
            dataset.fragments(),
            &dataset,
            &options,
            FilteredReadStream::make_scan_scheduler(&dataset, &options),
            FilteredReadStream::make_base_schedulers(&options),
        )
        .unwrap();
        assert_eq!(scoped.len(), 2);
        let metrics = Arc::new(FilteredReadGlobalMetrics::new(
            &ExecutionPlanMetricsSet::new(),
        ));
        let mut payload = Vec::new();
        for fragment in scoped {
            // Prove the real file uses the requested direct encoding; a fallback-only
            // scan would not exercise Boolean results together with rep/def levels.
            let predicate = fragment.encoded_predicate.as_ref().unwrap();
            let schema = Arc::new(dataset.schema().project(&["predicate"]).unwrap());
            let reader = fragment
                .fragment
                .open(&schema, fragment.frag_read_config())
                .await
                .unwrap();
            let plan = reader
                .primitive_predicate_plan(schema, predicate.clone())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(plan.encoding, expected_encoding);
            assert_eq!(plan.outcome, PrimitivePredicateOutcome::Unknown);
            let mut stream = FilteredReadStream::read_fragment(
                dataset.clone(),
                fragment,
                metrics.clone(),
                None,
                read.materialization_context.clone(),
                true,
                mode,
            )
            .await
            .unwrap();
            while let Some(task) = stream.next().await {
                for batch in task.unwrap().await.unwrap() {
                    payload.extend_from_slice(
                        batch.batch()["payload"]
                            .as_primitive::<UInt64Type>()
                            .values(),
                    );
                }
            }
        }
        payload.sort_unstable();
        (payload, metrics)
    }

    async fn direct_predicate_dataset(signed_rle: bool, nullable: bool) -> Dataset {
        let num_rows = 100_000_u64;
        let predicate: Arc<dyn arrow_array::Array> = if signed_rle {
            Arc::new(Int32Array::from_iter((0..num_rows).map(|row| {
                (!nullable || row % 7 != 0).then_some(((row % 50_000) / 4096) as i32 - 4)
            })))
        } else {
            Arc::new(UInt32Array::from_iter((0..num_rows).map(|row| {
                (!nullable || row % 7 != 0).then_some((row % 1024) as u32)
            })))
        };
        let batch = RecordBatch::try_from_iter(vec![
            ("predicate", predicate),
            (
                "payload",
                Arc::new(UInt64Array::from_iter_values(0..num_rows)) as Arc<dyn arrow_array::Array>,
            ),
        ])
        .unwrap();
        let schema = batch.schema();
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 50_000,
                data_storage_version: Some(lance_file::version::LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    #[rstest]
    #[case::eq("=", false)]
    #[case::ne("!=", false)]
    #[case::lt("<", false)]
    #[case::le("<=", false)]
    #[case::gt(">", false)]
    #[case::ge(">=", false)]
    #[case::literal_left("<", true)]
    #[tokio::test]
    async fn test_encoded_predicate_direct_scan_parity(
        #[case] operator: &str,
        #[case] literal_left: bool,
        #[values(false, true)] signed_rle: bool,
        #[values(false, true)] nullable: bool,
    ) {
        let dataset = Arc::new(direct_predicate_dataset(signed_rle, nullable).await);
        let literal = if signed_rle { "-2" } else { "512" };
        let filter = if literal_left {
            format!("{literal} {operator} predicate")
        } else {
            format!("predicate {operator} {literal}")
        };
        let encoding = if signed_rle {
            PrimitivePredicateEncoding::Rle
        } else {
            PrimitivePredicateEncoding::Bitpacked
        };
        let (expected, _) = execute_mode_scan(
            dataset.clone(),
            &filter,
            EncodedPredicateMode::Disabled,
            false,
            encoding,
        )
        .await;
        let (actual, metrics) = execute_mode_scan(
            dataset,
            &filter,
            EncodedPredicateMode::Forced,
            false,
            encoding,
        )
        .await;
        assert_eq!(actual, expected);
        assert_eq!(metrics.encoded_predicate_applied.value(), 2);
        assert_eq!(metrics.encoded_predicate_rows_scanned.value(), 100_000);
        assert_eq!(metrics.rows_scanned.value(), actual.len());
    }

    #[rstest]
    #[tokio::test]
    async fn test_encoded_predicate_with_deleted_rows(
        #[values(false, true)] signed_rle: bool,
        #[values(EncodedPredicateMode::Auto, EncodedPredicateMode::Forced)]
        mode: EncodedPredicateMode,
    ) {
        let mut dataset = direct_predicate_dataset(signed_rle, true).await;
        dataset.delete("payload % 3 = 0").await.unwrap();
        let dataset = Arc::new(dataset);
        let (filter, encoding) = if signed_rle {
            ("predicate = -2", PrimitivePredicateEncoding::Rle)
        } else {
            ("predicate = 512", PrimitivePredicateEncoding::Bitpacked)
        };
        let (expected, _) = execute_mode_scan(
            dataset.clone(),
            filter,
            EncodedPredicateMode::Disabled,
            true,
            encoding,
        )
        .await;
        let (actual, metrics) = execute_mode_scan(dataset, filter, mode, true, encoding).await;
        assert_eq!(actual, expected);
        assert!(!actual.is_empty());
        assert!(actual.iter().all(|row| row % 3 != 0));
        assert_eq!(metrics.encoded_predicate_applied.value(), 0);
    }

    #[tokio::test]
    async fn test_encoded_predicate_filter_correctness() {
        use arrow_array::{Int32Array, UInt64Array};
        use arrow_schema::{DataType, Field as ArrowField};

        let tmp_path = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("predicate", DataType::Int32, true),
            ArrowField::new("payload", DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![
                    Some(-2),
                    None,
                    Some(0),
                    Some(4),
                    Some(9),
                ])),
                Arc::new(UInt64Array::from(vec![10, 11, 12, 13, 14])),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let dataset = Arc::new(
            Dataset::write(reader, tmp_path.as_str(), None)
                .await
                .unwrap(),
        );
        let projection = dataset
            .empty_projection()
            .union_column("payload", OnMissing::Error)
            .unwrap();
        let planner = Planner::new(Arc::new(ArrowSchema::from(dataset.schema())));
        let filter = planner.parse_filter("predicate >= 0").unwrap();
        let options = FilteredReadOptions::new(projection)
            .with_filter(Some(filter.clone()), Some(filter))
            .unwrap();
        let plan = FilteredReadExec::try_new(dataset, options, None).unwrap();
        let batches = plan
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let batch = concat_batches(&plan.schema(), &batches).unwrap();
        assert_eq!(batch.schema().fields().len(), 1);
        let payload = batch
            .column(0)
            .as_primitive::<arrow_array::types::UInt64Type>();
        assert_eq!(payload.values().as_ref(), &[12, 13, 14]);
    }

    #[tokio::test]
    async fn test_encoded_predicate_auto_plans_current_encodings() {
        use arrow_array::UInt64Array;
        use arrow_schema::{DataType, Field as ArrowField};

        let tmp_path = TempStrDir::default();
        let num_rows = 200_000_u64;
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("bitpack", DataType::UInt32, false),
            ArrowField::new("rle", DataType::UInt32, false),
            ArrowField::new("constant", DataType::UInt32, false),
            ArrowField::new("payload", DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from_iter_values(
                    (0..num_rows).map(|row| (row % 1024) as u32),
                )),
                Arc::new(UInt32Array::from_iter_values(
                    (0..num_rows).map(|row| (row / 4096) as u32),
                )),
                Arc::new(UInt32Array::from_iter_values((0..num_rows).map(|_| 7))),
                Arc::new(UInt64Array::from_iter_values(0..num_rows)),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let dataset = Arc::new(
            Dataset::write(
                reader,
                tmp_path.as_str(),
                Some(WriteParams {
                    max_rows_per_file: 50_000,
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let (output_rows, metrics) =
            execute_encoded_predicate_test_scan(dataset.clone(), "constant = 8").await;
        assert_eq!(output_rows, 0);
        assert_eq!(metric_value(&metrics, "encoded_predicate_applied"), 4);
        assert_eq!(metric_value(&metrics, "encoded_predicate_rejected"), 0);
        assert_eq!(metric_value(&metrics, "encoded_predicate_selected_rows"), 0);

        let (output_rows, metrics) =
            execute_encoded_predicate_test_scan(dataset.clone(), "constant = 7").await;
        assert_eq!(output_rows, num_rows as usize);
        assert_eq!(metric_value(&metrics, "encoded_predicate_applied"), 0);
        assert_eq!(metric_value(&metrics, "encoded_predicate_rejected"), 4);
        assert_eq!(
            metric_value(&metrics, "encoded_predicate_sample_rows"),
            4 * ENCODED_PREDICATE_AUTO_SAMPLE_ROWS as usize
        );

        let (output_rows, metrics) =
            execute_encoded_predicate_test_scan(dataset.clone(), "rle = 1").await;
        assert_eq!(output_rows, 4096);
        assert_eq!(metric_value(&metrics, "encoded_predicate_applied"), 4);
        assert_eq!(metric_value(&metrics, "encoded_predicate_rejected"), 0);

        let (output_rows, metrics) =
            execute_encoded_predicate_test_scan(dataset.clone(), "bitpack = 1").await;
        assert_eq!(output_rows, 196);
        assert_eq!(metric_value(&metrics, "encoded_predicate_applied"), 0);
        assert_eq!(metric_value(&metrics, "encoded_predicate_rejected"), 4);
        assert_eq!(
            metric_value(&metrics, "encoded_predicate_sample_rows"),
            4 * ENCODED_PREDICATE_AUTO_SAMPLE_ROWS as usize
        );

        let (output_rows, metrics) =
            execute_encoded_predicate_test_scan(dataset, "bitpack = 2048").await;
        assert_eq!(output_rows, 0);
        assert_eq!(metric_value(&metrics, "encoded_predicate_applied"), 4);
        assert_eq!(metric_value(&metrics, "encoded_predicate_rejected"), 0);
        assert_eq!(
            metric_value(&metrics, "encoded_predicate_sample_rows"),
            4 * ENCODED_PREDICATE_AUTO_SAMPLE_ROWS as usize
        );
    }

    #[tokio::test]
    async fn test_encoded_predicate_auto_short_circuits_constant_layout() {
        use arrow_array::UInt64Array;
        use arrow_schema::{DataType, Field as ArrowField};
        use lance_file::version::LanceFileVersion;

        let tmp_path = TempStrDir::default();
        let num_rows = 10_000_u64;
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("constant", DataType::UInt32, false),
            ArrowField::new("payload", DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from_iter_values((0..num_rows).map(|_| 7))),
                Arc::new(UInt64Array::from_iter_values(0..num_rows)),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let dataset = Arc::new(
            Dataset::write(
                reader,
                tmp_path.as_str(),
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let (output_rows, metrics) =
            execute_encoded_predicate_test_scan(dataset.clone(), "constant = 8").await;
        assert_eq!(output_rows, 0);
        assert_eq!(metric_value(&metrics, "encoded_predicate_applied"), 1);
        assert_eq!(metric_value(&metrics, "encoded_predicate_sample_rows"), 0);

        let (output_rows, metrics) =
            execute_encoded_predicate_test_scan(dataset, "constant = 7").await;
        assert_eq!(output_rows, num_rows as usize);
        assert_eq!(metric_value(&metrics, "encoded_predicate_applied"), 1);
        assert_eq!(metric_value(&metrics, "encoded_predicate_sample_rows"), 0);
        assert_eq!(
            metric_value(&metrics, "encoded_predicate_selected_rows"),
            num_rows as usize
        );
    }

    #[tokio::test]
    async fn test_encoded_predicate_auto_preserves_nullable_constant_layout_filter() {
        use arrow_array::UInt64Array;
        use arrow_schema::{DataType, Field as ArrowField};
        use lance_file::version::LanceFileVersion;

        let tmp_path = TempStrDir::default();
        let num_rows = 10_000_u64;
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("predicate", DataType::Int32, true),
            ArrowField::new("payload", DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter(
                    (0..num_rows).map(|row| row.is_multiple_of(2).then_some(7)),
                )),
                Arc::new(UInt64Array::from_iter_values(0..num_rows)),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let dataset = Arc::new(
            Dataset::write(
                reader,
                tmp_path.as_str(),
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let (output_rows, metrics) =
            execute_encoded_predicate_test_scan(dataset.clone(), "predicate = 7").await;
        assert_eq!(output_rows, num_rows as usize / 2);
        assert_eq!(metric_value(&metrics, "encoded_predicate_applied"), 0);
        assert_eq!(metric_value(&metrics, "encoded_predicate_rejected"), 1);
        assert_eq!(metric_value(&metrics, "encoded_predicate_sample_rows"), 0);

        let (output_rows, metrics) =
            execute_encoded_predicate_test_scan(dataset, "predicate = 8").await;
        assert_eq!(output_rows, 0);
        assert_eq!(metric_value(&metrics, "encoded_predicate_applied"), 1);
        assert_eq!(metric_value(&metrics, "encoded_predicate_rejected"), 0);
    }

    #[rstest]
    #[case::fragmented_first(true)]
    #[case::fragmented_last(false)]
    #[tokio::test]
    async fn test_encoded_predicate_auto_decides_per_fragment(#[case] fragmented_first: bool) {
        use arrow_array::UInt64Array;
        use arrow_schema::{DataType, Field as ArrowField};
        use lance_file::version::LanceFileVersion;

        let tmp_path = TempStrDir::default();
        let rows_per_fragment = 50_000_u64;
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("predicate", DataType::UInt32, false),
            ArrowField::new("payload", DataType::UInt64, false),
        ]));
        let no_match = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from_iter_values(
                    (0..rows_per_fragment).map(|row| (row % 1024) as u32),
                )),
                Arc::new(UInt64Array::from_iter_values(0..rows_per_fragment)),
            ],
        )
        .unwrap();
        let fragmented = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from_iter_values(
                    (0..rows_per_fragment).map(|row| if row.is_multiple_of(2) { 2048 } else { 0 }),
                )),
                Arc::new(UInt64Array::from_iter_values(
                    rows_per_fragment..2 * rows_per_fragment,
                )),
            ],
        )
        .unwrap();
        let batches = if fragmented_first {
            vec![Ok(fragmented), Ok(no_match)]
        } else {
            vec![Ok(no_match), Ok(fragmented)]
        };
        let reader = RecordBatchIterator::new(batches, schema);
        let dataset = Arc::new(
            Dataset::write(
                reader,
                tmp_path.as_str(),
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    max_rows_per_file: rows_per_fragment as usize,
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let (output_rows, metrics) =
            execute_encoded_predicate_test_scan(dataset, "predicate = 2048").await;
        assert_eq!(output_rows, rows_per_fragment as usize / 2);
        assert_eq!(metric_value(&metrics, "encoded_predicate_applied"), 1);
        assert_eq!(metric_value(&metrics, "encoded_predicate_rejected"), 1);
        assert_eq!(
            metric_value(&metrics, "encoded_predicate_sample_rows"),
            2 * ENCODED_PREDICATE_AUTO_SAMPLE_ROWS as usize
        );
    }

    #[tokio::test]
    async fn test_encoded_predicate_auto_caps_full_fragment_amplification() {
        use arrow_array::UInt64Array;
        use arrow_schema::{DataType, Field as ArrowField};
        use lance_file::version::LanceFileVersion;

        let tmp_path = TempStrDir::default();
        let num_rows = 50_000_u64;
        let sample_window_rows = ENCODED_PREDICATE_AUTO_SAMPLE_ROWS / 2;
        let fragmented_end = num_rows - sample_window_rows;
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("predicate", DataType::UInt32, false),
            ArrowField::new("payload", DataType::UInt64, false),
        ]));
        let predicate = UInt32Array::from_iter_values((0..num_rows).map(|row| {
            if (sample_window_rows..fragmented_end).contains(&row) && row.is_multiple_of(2) {
                2048
            } else {
                (row % 1024) as u32
            }
        }));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(predicate),
                Arc::new(UInt64Array::from_iter_values(0..num_rows)),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let dataset = Arc::new(
            Dataset::write(
                reader,
                tmp_path.as_str(),
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let (output_rows, metrics) =
            execute_encoded_predicate_test_scan(dataset, "predicate = 2048").await;
        assert_eq!(
            output_rows,
            ((fragmented_end - sample_window_rows) / 2) as usize
        );
        assert_eq!(metric_value(&metrics, "encoded_predicate_applied"), 0);
        assert_eq!(metric_value(&metrics, "encoded_predicate_rejected"), 1);
        assert_eq!(
            metric_value(&metrics, "encoded_predicate_sample_rows"),
            ENCODED_PREDICATE_AUTO_SAMPLE_ROWS as usize
        );
        assert_eq!(
            metric_value(&metrics, "encoded_predicate_sample_selected_rows"),
            0
        );
        assert_eq!(
            metric_value(&metrics, "encoded_predicate_selected_ranges"),
            0
        );
    }
}
