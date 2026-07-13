// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Duration;

use super::cleanup_data_fragments;
use super::retry::{RetryConfig, RetryExecutor, execute_with_retry};
use super::{CommitBuilder, WriteParams, write_fragments_internal};
use crate::dataset::ManifestWriteConfig;
use crate::dataset::rowids::get_row_id_index;
use crate::dataset::transaction::UpdateMode::RewriteRows;
use crate::dataset::transaction::{Operation, RowAddressManifestApplyContext, TransactionBuilder};
use crate::dataset::utils::make_rowid_capture_stream;
use crate::index::DatasetIndexExt;
use crate::io::exec::filtered_read::{EvaluatedIndex, FilteredReadExec, FilteredReadOptions};
use crate::{Dataset, io::exec::Planner};
use crate::{Error, Result};
use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, DataType, Schema as ArrowSchema};
use datafusion::common::DFSchema;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::ExprSchemable;
use datafusion::physical_expr::PhysicalSortExpr;
use datafusion::physical_plan::expressions;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{
    ExecutionPlan, ExecutionPlanProperties, PhysicalExpr, RecordBatchStream,
};
use datafusion::prelude::Expr;
use datafusion::scalar::ScalarValue;
use datafusion_physical_expr::LexOrdering;
use futures::StreamExt;
use lance_arrow::RecordBatchExt;
use lance_core::datatypes::BlobHandling;
use lance_core::error::{InvalidInputSnafu, box_error};
use lance_core::utils::address::LogicalRowAddress;
use lance_core::utils::tempfile::TempDir;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{ROW_ADDR_FIELD, ROW_ID_FIELD, ROW_OFFSET_FIELD};
use lance_datafusion::exec::{execute_plan, get_session_context};
use lance_datafusion::expr::safe_coerce_scalar;
use lance_datafusion::spill::{SpillSender, create_replay_spill};
use lance_select::RowAddrTreeMap;
use lance_table::format::{
    DeletionFile, DeletionFileType, Fragment, LogicalRowAddressSelection, RowAddressFieldChange,
    RowAddressLayout, RowAddressLayoutDelta, RowAddressPlacement, RowAddressPlacementDelta,
    RowAddressPlacementKind, RowAddressSourceFloor, RowAddressTargetFragment,
    RowAddressTargetRange, RowIdMeta, fingerprint_row_sequence,
};
use roaring::{RoaringBitmap, RoaringTreemap};
use snafu::ResultExt;

#[derive(Debug, Clone, PartialEq, Eq)]
struct V2_3UpdateLogicalOrderPlan {
    /// Exclusive physical-fragment boundaries for monotonic input runs.
    /// Empty means the manifest's physical order is already logical order.
    logical_run_ends: Vec<u32>,
    /// ExplicitMap and an internally non-monotonic physical fragment cannot be
    /// split into independently ordered scanner inputs.
    requires_full_logical_sort: bool,
}

#[derive(Debug, Clone, Copy)]
struct LogicalOrderSegment {
    destination_start: u32,
    first_logical_address: u64,
    last_logical_address: u64,
}

fn selection_logical_bounds(selection: &LogicalRowAddressSelection) -> Result<Option<(u64, u64)>> {
    let cardinality = selection.cardinality();
    if cardinality == 0 {
        return Ok(None);
    }
    let first = selection.select(0)?.ok_or_else(|| {
        Error::invalid_input("logical selection cardinality exceeds encoded values")
    })?;
    let last = selection.select(cardinality - 1)?.ok_or_else(|| {
        Error::invalid_input("logical selection cardinality exceeds encoded values")
    })?;
    Ok(Some((first.raw(), last.raw())))
}

/// Plan logical-order UPDATE input from manifest routing metadata only.
///
/// Each fast placement codec is logically sorted inside a source segment. We
/// inspect only the segment destination offset and its first/last logical
/// address. Exclusions and deletion vectors only remove rows from those sorted
/// sequences, so the untrimmed bounds are conservative without decoding rows.
fn plan_v2_3_update_logical_order(
    fragments: &[Fragment],
    layout: &RowAddressLayout,
) -> Result<V2_3UpdateLogicalOrderPlan> {
    let fragment_ids = fragments
        .iter()
        .map(|fragment| {
            u32::try_from(fragment.id).map_err(|_| {
                Error::invalid_input(format!(
                    "physical fragment id {} exceeds row-address capacity",
                    fragment.id
                ))
            })
        })
        .collect::<Result<BTreeSet<_>>>()?;
    let mut segments = BTreeMap::<u32, Vec<LogicalOrderSegment>>::new();
    let mut push_segment = |physical_fragment_id: u32, segment: LogicalOrderSegment| {
        if fragment_ids.contains(&physical_fragment_id) {
            segments
                .entry(physical_fragment_id)
                .or_default()
                .push(segment);
        }
    };

    for placement in &layout.placements {
        match placement {
            RowAddressPlacement::Direct(value) => {
                let first =
                    LogicalRowAddress::try_new_from_parts(value.source.logical_fragment_id, 0)?;
                let last = LogicalRowAddress::try_new_from_parts(
                    value.source.logical_fragment_id,
                    value.source.slot_count.checked_sub(1).ok_or_else(|| {
                        Error::invalid_input("Direct placement has an empty logical domain")
                    })?,
                )?;
                push_segment(
                    value.destination_fragment_id,
                    LogicalOrderSegment {
                        destination_start: value.destination_start,
                        first_logical_address: first.raw(),
                        last_logical_address: last.raw(),
                    },
                );
            }
            RowAddressPlacement::PackedRun(value) => {
                let first_domain = value.domains.domain_at(0)?;
                let last_domain_ordinal =
                    value.domains.domain_count().checked_sub(1).ok_or_else(|| {
                        Error::invalid_input("PackedRun placement has no logical domains")
                    })?;
                let last_domain = value.domains.domain_at(last_domain_ordinal)?;
                let first =
                    LogicalRowAddress::try_new_from_parts(first_domain.logical_fragment_id, 0)?;
                let last = LogicalRowAddress::try_new_from_parts(
                    last_domain.logical_fragment_id,
                    last_domain.slot_count.checked_sub(1).ok_or_else(|| {
                        Error::invalid_input("PackedRun placement has an empty logical domain")
                    })?,
                )?;
                push_segment(
                    value.destination_fragment_id,
                    LogicalOrderSegment {
                        destination_start: value.destination_start,
                        first_logical_address: first.raw(),
                        last_logical_address: last.raw(),
                    },
                );
            }
            RowAddressPlacement::Selected(value) => {
                if let Some((first, last)) = selection_logical_bounds(&value.selection)? {
                    push_segment(
                        value.destination_fragment_id,
                        LogicalOrderSegment {
                            destination_start: value.destination_start,
                            first_logical_address: first,
                            last_logical_address: last,
                        },
                    );
                }
            }
            RowAddressPlacement::ExtentList(value) => {
                for extent in &value.extents {
                    let last_slot = extent
                        .source_start
                        .checked_add(extent.length)
                        .and_then(|end| end.checked_sub(1))
                        .ok_or_else(|| {
                            Error::invalid_input(format!(
                                "ExtentList source range overflows: source_start={}, length={}",
                                extent.source_start, extent.length
                            ))
                        })?;
                    let first = LogicalRowAddress::try_new_from_parts(
                        value.source.logical_fragment_id,
                        extent.source_start,
                    )?;
                    let last = LogicalRowAddress::try_new_from_parts(
                        value.source.logical_fragment_id,
                        last_slot,
                    )?;
                    push_segment(
                        extent.destination_fragment_id,
                        LogicalOrderSegment {
                            destination_start: extent.destination_start,
                            first_logical_address: first.raw(),
                            last_logical_address: last.raw(),
                        },
                    );
                }
            }
            RowAddressPlacement::SparseSelection(value) => {
                let mut destination_start = u64::from(value.destination_start);
                for source in &value.sources {
                    if let Some((first, last)) = selection_logical_bounds(&source.selection)? {
                        push_segment(
                            value.destination_fragment_id,
                            LogicalOrderSegment {
                                destination_start: u32::try_from(destination_start).map_err(
                                    |_| {
                                        Error::invalid_input(
                                            "SparseSelection destination offset exceeds u32",
                                        )
                                    },
                                )?,
                                first_logical_address: first,
                                last_logical_address: last,
                            },
                        );
                    }
                    destination_start = destination_start
                        .checked_add(source.selection.cardinality())
                        .ok_or_else(|| {
                            Error::invalid_input("SparseSelection destination offset overflow")
                        })?;
                }
            }
            RowAddressPlacement::ExplicitMap(value) => {
                if value
                    .destinations
                    .iter()
                    .any(|destination| fragment_ids.contains(&destination.physical_fragment_id))
                {
                    return Ok(V2_3UpdateLogicalOrderPlan {
                        logical_run_ends: Vec::new(),
                        requires_full_logical_sort: true,
                    });
                }
            }
        }
    }

    let mut logical_run_ends = Vec::new();
    let mut current_run_last = None;
    for (fragment_index, fragment) in fragments.iter().enumerate() {
        let fragment_id = u32::try_from(fragment.id).map_err(|_| {
            Error::invalid_input(format!(
                "physical fragment id {} exceeds row-address capacity",
                fragment.id
            ))
        })?;
        let mut fragment_segments = if let Some(native) = fragment.native_logical_domain {
            let physical_rows = u32::try_from(fragment.physical_rows.ok_or_else(|| {
                Error::invalid_input(format!(
                    "native logical fragment {} is missing physical_rows",
                    fragment.id
                ))
            })?)
            .map_err(|_| Error::invalid_input("native logical fragment exceeds u32 rows"))?;
            let first = LogicalRowAddress::try_new_from_parts(native.logical_fragment_id, 0)?;
            let last_slot = physical_rows.checked_sub(1).ok_or_else(|| {
                Error::invalid_input(format!(
                    "native logical fragment {} has zero physical rows",
                    fragment.id
                ))
            })?;
            let last =
                LogicalRowAddress::try_new_from_parts(native.logical_fragment_id, last_slot)?;
            vec![LogicalOrderSegment {
                destination_start: 0,
                first_logical_address: first.raw(),
                last_logical_address: last.raw(),
            }]
        } else {
            segments.remove(&fragment_id).unwrap_or_default()
        };
        if fragment_segments.is_empty() {
            continue;
        }
        fragment_segments.sort_unstable_by_key(|segment| segment.destination_start);
        if fragment_segments
            .windows(2)
            .any(|pair| pair[0].last_logical_address >= pair[1].first_logical_address)
        {
            return Ok(V2_3UpdateLogicalOrderPlan {
                logical_run_ends: Vec::new(),
                requires_full_logical_sort: true,
            });
        }
        let first = fragment_segments[0].first_logical_address;
        let last = fragment_segments
            .last()
            .map(|segment| segment.last_logical_address)
            .ok_or_else(|| Error::internal("logical-order fragment segments disappeared"))?;
        if current_run_last.is_some_and(|previous| previous >= first) {
            logical_run_ends.push(
                u32::try_from(fragment_index).map_err(|_| {
                    Error::invalid_input("update source fragment count exceeds u32")
                })?,
            );
        }
        current_run_last = Some(last);
    }
    if !logical_run_ends.is_empty() {
        logical_run_ends.push(
            u32::try_from(fragments.len())
                .map_err(|_| Error::invalid_input("update source fragment count exceeds u32"))?,
        );
    }
    Ok(V2_3UpdateLogicalOrderPlan {
        logical_run_ends,
        requires_full_logical_sort: false,
    })
}

fn logical_row_ordering(plan: &Arc<dyn ExecutionPlan>) -> Result<LexOrdering> {
    let row_id_sort = PhysicalSortExpr {
        expr: expressions::col(ROW_ID_FIELD.name(), plan.schema().as_ref())?,
        options: arrow::compute::SortOptions {
            descending: false,
            nulls_first: false,
        },
    };
    LexOrdering::new([row_id_sort])
        .ok_or_else(|| Error::internal("logical row ordering cannot be empty"))
}

async fn create_v2_3_update_scan_plan(
    scanner: &crate::dataset::scanner::Scanner,
    fragments: &[Fragment],
    order_plan: &V2_3UpdateLogicalOrderPlan,
) -> Result<Arc<dyn ExecutionPlan>> {
    create_v2_3_update_scan_plan_impl(scanner, fragments, order_plan, None).await
}

async fn create_v2_3_full_sort_scan_plan(
    scanner: &crate::dataset::scanner::Scanner,
    fragments: &[Fragment],
) -> Result<Arc<dyn ExecutionPlan>> {
    let mut full_scanner = scanner.clone();
    full_scanner
        .with_fragments(fragments.to_vec())
        .scan_in_order(false);
    let input = full_scanner.create_plan().await?;
    let ordering = logical_row_ordering(&input)?;
    Ok(Arc::new(SortExec::new(ordering, input)))
}

fn collect_filtered_read_execs<'a>(
    plan: &'a Arc<dyn ExecutionPlan>,
    output: &mut Vec<&'a FilteredReadExec>,
) {
    if let Some(filtered_read) = plan.as_any().downcast_ref::<FilteredReadExec>() {
        output.push(filtered_read);
    }
    for child in plan.children() {
        collect_filtered_read_execs(child, output);
    }
}

fn replace_filtered_read(
    template: Arc<dyn ExecutionPlan>,
    dataset: Arc<Dataset>,
    mut options: FilteredReadOptions,
    fragments: &[Fragment],
    evaluated_index: Option<Arc<EvaluatedIndex>>,
) -> Result<Arc<dyn ExecutionPlan>> {
    options.fragments = Some(Arc::new(fragments.to_vec()));
    let replacement = FilteredReadExec::try_new(dataset, options, None)?;
    let replacement = if let Some(evaluated_index) = evaluated_index {
        replacement.with_evaluated_index(evaluated_index)
    } else {
        replacement
    };
    let replacement: Arc<dyn ExecutionPlan> = Arc::new(replacement);
    let mut replacement_count = 0_usize;
    let transformed = template.transform_up(|node| {
        if node.as_any().is::<FilteredReadExec>() {
            replacement_count += 1;
            Ok(Transformed::yes(replacement.clone()))
        } else {
            Ok(Transformed::no(node))
        }
    })?;
    if replacement_count != 1 {
        return Err(Error::internal(format!(
            "ordered UPDATE run replaced {replacement_count} FilteredReadExec nodes instead of one"
        )));
    }
    Ok(transformed.data)
}

async fn create_v2_3_update_scan_plan_impl(
    scanner: &crate::dataset::scanner::Scanner,
    fragments: &[Fragment],
    order_plan: &V2_3UpdateLogicalOrderPlan,
    index_evaluations: Option<&std::sync::atomic::AtomicUsize>,
) -> Result<Arc<dyn ExecutionPlan>> {
    if order_plan.requires_full_logical_sort {
        if !order_plan.logical_run_ends.is_empty() {
            return Err(Error::internal(
                "full logical sort cannot also declare sort-preserving input runs",
            ));
        }
        return create_v2_3_full_sort_scan_plan(scanner, fragments).await;
    }

    if order_plan.logical_run_ends.is_empty() {
        let mut ordered_scanner = scanner.clone();
        ordered_scanner
            .with_fragments(fragments.to_vec())
            .scan_in_order(true);
        return ordered_scanner.create_plan().await;
    }

    let mut previous_end = 0_usize;
    let mut logical_runs = Vec::with_capacity(order_plan.logical_run_ends.len());
    for end in &order_plan.logical_run_ends {
        let end = usize::try_from(*end)
            .map_err(|_| Error::internal("logical run boundary does not fit in usize"))?;
        if end <= previous_end || end > fragments.len() {
            return Err(Error::internal(format!(
                "invalid logical run boundary {end} after {previous_end} for {} source fragments",
                fragments.len()
            )));
        }
        logical_runs.push(previous_end..end);
        previous_end = end;
    }
    if previous_end != fragments.len() {
        return Err(Error::internal(format!(
            "logical run boundaries cover {previous_end} of {} source fragments",
            fragments.len()
        )));
    }

    let mut ordered_scanner = scanner.clone();
    ordered_scanner
        .with_fragments(fragments.to_vec())
        .scan_in_order(true);
    let template = ordered_scanner.create_plan().await?;
    let mut filtered_reads = Vec::new();
    collect_filtered_read_execs(&template, &mut filtered_reads);
    if filtered_reads.iter().any(|read| {
        read.options().scan_range_before_filter.is_some()
            || read.options().scan_range_after_filter.is_some()
    }) {
        return Err(Error::internal(
            "storage-version-2.3 UPDATE cannot split a scan with a before-filter or after-filter range",
        ));
    }
    if filtered_reads.len() != 1 {
        return create_v2_3_full_sort_scan_plan(scanner, fragments).await;
    }

    let all_fragment_ids = fragments
        .iter()
        .map(|fragment| {
            u32::try_from(fragment.id).map_err(|_| {
                Error::internal(format!(
                    "physical fragment id {} exceeds row-address capacity",
                    fragment.id
                ))
            })
        })
        .collect::<Result<BTreeSet<_>>>()?;
    if all_fragment_ids.len() != fragments.len() {
        return Err(Error::internal(
            "storage-version-2.3 UPDATE source contains duplicate physical fragment ids",
        ));
    }

    let filtered_read = filtered_reads[0];
    let dataset = filtered_read.dataset().clone();
    let options = filtered_read.options().clone();
    if filtered_read.index_input().is_none() {
        let mut inputs = Vec::<Arc<dyn ExecutionPlan>>::with_capacity(logical_runs.len());
        for run in &logical_runs {
            let input = replace_filtered_read(
                template.clone(),
                dataset.clone(),
                options.clone(),
                &fragments[run.clone()],
                None,
            )?;
            if input.output_partitioning().partition_count() != 1 {
                return Err(Error::internal(format!(
                    "ordered update run produced {} partitions instead of one",
                    input.output_partitioning().partition_count()
                )));
            }
            inputs.push(input);
        }
        let union = UnionExec::try_new(inputs)?;
        let ordering = logical_row_ordering(&union)?;
        return Ok(Arc::new(SortPreservingMergeExec::new(ordering, union)));
    }

    let execution_options = scanner.execution_options();
    let task_ctx = get_session_context(&execution_options).task_ctx();
    if let Some(index_evaluations) = index_evaluations {
        index_evaluations.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let evaluated_index = filtered_read
        .evaluate_index(task_ctx)
        .await?
        .ok_or_else(|| {
            Error::internal("indexed UPDATE scan did not produce an evaluated scalar index")
        })?;

    let mut fragment_coverage = BTreeMap::<u32, usize>::new();
    let mut inputs = Vec::<Arc<dyn ExecutionPlan>>::with_capacity(logical_runs.len());
    for run in logical_runs {
        let run_fragments = &fragments[run];
        let run_fragment_ids = run_fragments
            .iter()
            .map(|fragment| {
                u32::try_from(fragment.id).map_err(|_| {
                    Error::internal(format!(
                        "physical fragment id {} exceeds row-address capacity",
                        fragment.id
                    ))
                })
            })
            .collect::<Result<BTreeSet<_>>>()?;
        for fragment_id in run_fragment_ids {
            *fragment_coverage.entry(fragment_id).or_default() += 1;
        }

        let input = replace_filtered_read(
            template.clone(),
            dataset.clone(),
            options.clone(),
            run_fragments,
            Some(evaluated_index.clone()),
        )?;
        if input.output_partitioning().partition_count() != 1 {
            return Err(Error::internal(format!(
                "ordered update run produced {} partitions instead of one",
                input.output_partitioning().partition_count()
            )));
        }
        inputs.push(input);
    }
    if fragment_coverage.len() != all_fragment_ids.len()
        || fragment_coverage.values().any(|count| *count != 1)
    {
        return Err(Error::internal(
            "ordered UPDATE runs do not cover every physical fragment exactly once",
        ));
    }
    let union = UnionExec::try_new(inputs)?;
    let ordering = logical_row_ordering(&union)?;
    Ok(Arc::new(SortPreservingMergeExec::new(ordering, union)))
}

/// Collect a field id and all of its descendant field ids (pre-order). A struct
/// column update rewrites the whole subtree, so an index on any descendant must be
/// treated as modified.
fn collect_subtree_field_ids(field: &lance_core::datatypes::Field, out: &mut Vec<u32>) {
    out.push(field.id as u32);
    for child in &field.children {
        collect_subtree_field_ids(child, out);
    }
}

fn build_v2_3_update_delta(
    dataset: &Dataset,
    logical_row_ids: &lance_table::rowids::RowIdSequence,
    new_fragments: &[Fragment],
    field_ids: &[i32],
) -> Result<RowAddressLayoutDelta> {
    let layout = dataset
        .manifest
        .row_address_layout
        .as_ref()
        .ok_or_else(|| Error::internal("storage-version-2.3 update is missing RowAddressLayout"))?;
    let addresses = logical_row_ids
        .iter()
        .map(LogicalRowAddress::try_from)
        .collect::<Result<Vec<_>>>()?;
    if addresses.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(Error::not_supported(
            "storage-version-2.3 default update output must be strictly ordered by logical row address",
        ));
    }
    let expected_rows = new_fragments.iter().try_fold(0_usize, |count, fragment| {
        count
            .checked_add(fragment.physical_rows.ok_or_else(|| {
                Error::internal("storage-version-2.3 update output is missing physical_rows")
            })?)
            .ok_or_else(|| Error::invalid_input("update output row count overflow"))
    })?;
    if addresses.len() != expected_rows {
        return Err(Error::internal(format!(
            "captured {} logical row IDs for {} update output rows",
            addresses.len(),
            expected_rows
        )));
    }

    let router = dataset.row_address_router()?;
    let mut source_domains = addresses
        .iter()
        .map(|address| address.logical_fragment_id())
        .collect::<Vec<_>>();
    source_domains.sort_unstable();
    source_domains.dedup();
    let source_domains = source_domains
        .into_iter()
        .map(|logical_fragment_id| {
            router.logical_domain(logical_fragment_id)?.ok_or_else(|| {
                Error::internal(format!(
                    "logical domain {logical_fragment_id} is missing from the source layout"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut placements = Vec::with_capacity(new_fragments.len());
    let mut offset = 0_usize;
    for (ordinal, fragment) in new_fragments.iter().enumerate() {
        let row_count = fragment.physical_rows.ok_or_else(|| {
            Error::internal("storage-version-2.3 update output is missing physical_rows")
        })?;
        let end = offset + row_count;
        let output_addresses = &addresses[offset..end];
        let target = RowAddressTargetRange {
            fragment: RowAddressTargetFragment::NewFragmentOrdinal(
                u32::try_from(ordinal)
                    .map_err(|_| Error::invalid_input("too many update output fragments"))?,
            ),
            start_offset: 0,
            end_offset: u32::try_from(row_count)
                .map_err(|_| Error::invalid_input("update output exceeds u32 row capacity"))?,
        };

        let mut source_selections = Vec::new();
        let mut domain_start = 0_usize;
        while domain_start < output_addresses.len() {
            let logical_fragment_id = output_addresses[domain_start].logical_fragment_id();
            let domain_end = output_addresses[domain_start..]
                .partition_point(|address| address.logical_fragment_id() == logical_fragment_id)
                + domain_start;
            let bitmap = output_addresses[domain_start..domain_end]
                .iter()
                .map(|address| address.raw())
                .collect::<RoaringTreemap>();
            source_selections.push(LogicalRowAddressSelection::from_bitmap(bitmap)?);
            domain_start = domain_end;
        }
        let placement_kind = if source_selections.len() == 1 {
            RowAddressPlacementKind::Selected
        } else {
            RowAddressPlacementKind::SparseSelection
        };
        placements.push(RowAddressPlacementDelta {
            source_selections,
            target,
            placement_kind,
            output_cardinality: row_count as u64,
            output_row_sequence_fingerprint: fingerprint_row_sequence(
                target,
                output_addresses.iter().copied(),
            )?,
        });
        offset = end;
    }

    let all_selection = LogicalRowAddressSelection::from_bitmap(
        addresses
            .iter()
            .map(|address| address.raw())
            .collect::<RoaringTreemap>(),
    )?;
    let source_floors = field_ids
        .iter()
        .map(|field_id| {
            let generation = layout
                .index_commit_floors
                .iter()
                .find(|floor| floor.field_id == *field_id)
                .or_else(|| {
                    layout
                        .field_default_generations
                        .iter()
                        .find(|generation| generation.field_id == *field_id)
                })
                .map(|generation| generation.generation)
                .ok_or_else(|| {
                    Error::internal(format!(
                        "field {field_id} is missing a source generation floor"
                    ))
                })?;
            Ok(RowAddressSourceFloor {
                field_id: *field_id,
                generation,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(RowAddressLayoutDelta {
        source_domains,
        placements,
        retired_selections: Vec::new(),
        field_changes: (!field_ids.is_empty())
            .then(|| RowAddressFieldChange {
                selection: all_selection,
                field_ids: field_ids.to_vec(),
            })
            .into_iter()
            .collect(),
        source_floors,
        expected_layout_fingerprint: layout.fingerprint.clone(),
        replaced_generations: Vec::new(),
        row_aligned_rewrite_proofs: Vec::new(),
        create_namespace_uuid: None,
        explicit_map_placements: BTreeMap::new(),
    })
}

fn v2_3_update_output_fragments(row_count: usize, max_rows_per_file: usize) -> Vec<Fragment> {
    let mut remaining = row_count;
    let mut fragments = Vec::new();
    while remaining != 0 {
        let rows = remaining.min(max_rows_per_file);
        fragments.push(Fragment {
            id: 0,
            files: Vec::new(),
            deletion_file: None,
            row_id_meta: None,
            physical_rows: Some(rows),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            native_logical_domain: None,
        });
        remaining -= rows;
    }
    fragments
}

async fn preflight_v2_3_update(
    dataset: &Dataset,
    logical_row_ids: &lance_table::rowids::RowIdSequence,
    field_ids: &[u32],
    max_rows_per_file: usize,
) -> Result<RowAddressLayoutDelta> {
    if logical_row_ids.is_empty() {
        let layout = dataset
            .manifest
            .row_address_layout
            .as_ref()
            .ok_or_else(|| {
                Error::internal("storage-version-2.3 update is missing RowAddressLayout")
            })?;
        return Ok(RowAddressLayoutDelta {
            expected_layout_fingerprint: layout.fingerprint.clone(),
            ..RowAddressLayoutDelta::default()
        });
    }
    let row_count = usize::try_from(logical_row_ids.len())
        .map_err(|_| Error::invalid_input("update row count exceeds platform capacity"))?;
    let new_fragments = v2_3_update_output_fragments(row_count, max_rows_per_file);
    let signed_field_ids = field_ids
        .iter()
        .map(|field_id| {
            i32::try_from(*field_id).map_err(|_| {
                Error::invalid_input(format!("updated field id {field_id} exceeds i32"))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut row_address_layout_delta =
        build_v2_3_update_delta(dataset, logical_row_ids, &new_fragments, &signed_field_ids)?;

    let raw_ids = logical_row_ids.iter().collect::<Vec<_>>();
    let resolved = dataset.resolve_logical_row_ids_async(&raw_ids).await?;
    let mut new_deletions = BTreeMap::<u32, RoaringBitmap>::new();
    for (row_id, address) in raw_ids.into_iter().zip(resolved) {
        let address = address.ok_or_else(|| {
            Error::internal(format!(
                "updated logical row id {row_id} has no current physical placement"
            ))
        })?;
        new_deletions
            .entry(address.fragment_id())
            .or_default()
            .insert(address.row_offset());
    }

    let mut updated_fragments = Vec::new();
    let mut removed_fragment_ids = Vec::new();
    let mut context = RowAddressManifestApplyContext::default();
    for (fragment_id, additions) in new_deletions {
        let fragment = dataset.get_fragment(fragment_id as usize).ok_or_else(|| {
            Error::internal(format!("missing update source fragment {fragment_id}"))
        })?;
        let current = fragment
            .get_deletion_vector()
            .await?
            .map(|deletions| deletions.to_sorted_iter().collect::<RoaringBitmap>())
            .unwrap_or_default();
        context
            .current_deletion_vectors
            .insert(fragment_id, current.clone());
        let mut successor = current;
        successor |= additions;
        let physical_rows = u32::try_from(fragment.metadata.physical_rows.ok_or_else(|| {
            Error::internal(format!(
                "source fragment {fragment_id} is missing physical_rows"
            ))
        })?)
        .map_err(|_| Error::invalid_input("update source fragment exceeds u32 rows"))?;
        if successor.len() == physical_rows as u64 && successor.contains_range(0..physical_rows) {
            removed_fragment_ids.push(fragment_id as u64);
            context
                .newly_fully_deleted_source_fragments
                .insert(fragment_id);
        } else {
            let mut metadata = fragment.metadata.clone();
            metadata.deletion_file = Some(DeletionFile {
                read_version: dataset.manifest.version + 1,
                id: 0,
                file_type: DeletionFileType::Bitmap,
                num_deleted_rows: Some(successor.len() as usize),
                base_id: None,
            });
            updated_fragments.push(metadata);
            context
                .successor_deletion_vectors
                .insert(fragment_id, successor);
        }
    }

    super::include_previous_deletions_in_v2_3_retirement(
        dataset,
        &removed_fragment_ids,
        &context.current_deletion_vectors,
        &mut row_address_layout_delta,
    )
    .await?;

    let operation = Operation::Update {
        removed_fragment_ids,
        updated_fragments,
        new_fragments,
        fields_modified: field_ids.to_vec(),
        merged_generations: Vec::new(),
        fields_for_preserving_frag_bitmap: field_ids.to_vec(),
        update_mode: Some(RewriteRows),
        inserted_rows_filter: None,
        updated_fragment_offsets: None,
    };
    let transaction = TransactionBuilder::new(dataset.manifest.version, operation)
        .row_address_layout_delta(Some(row_address_layout_delta.clone()))
        .build();
    let indices = dataset.load_indices().await?;
    transaction.build_manifest_with_row_address_context(
        Some(dataset.manifest.as_ref()),
        indices.as_ref().clone(),
        "row-address-update-preflight.txn",
        &ManifestWriteConfig::default(),
        Some(&context),
    )?;
    Ok(row_address_layout_delta)
}

/// Build an update operation.
///
/// This operation is similar to SQL's UPDATE statement. It allows you to change
/// the values of all or a subset of columns with SQL expressions.
///
/// Use the [UpdateBuilder] to construct an update job. For example:
///
/// ```
/// # use lance::{Dataset, Result};
/// # use lance::dataset::UpdateBuilder;
/// # use std::sync::Arc;
/// # async fn example(dataset: Arc<Dataset>) -> Result<()> {
/// let result = UpdateBuilder::new(dataset)
///     .update_where("region_id = 10")?
///     .set("region_name", "New York")?
///     .build()?
///     .execute()
///     .await?;
/// # Ok(())
/// # }
/// ```
///
#[derive(Debug, Clone)]
pub struct UpdateBuilder {
    /// The dataset snapshot to update.
    dataset: Arc<Dataset>,
    /// The condition to apply to find matching rows to update. If None, all rows are updated.
    condition: Option<Expr>,
    /// The updates to apply to matching rows.
    updates: HashMap<String, Expr>,
    /// Number of times to retry on commit conflicts.
    conflict_retries: u32,
    /// Total timeout for retries.
    retry_timeout: Duration,
}

impl UpdateBuilder {
    pub fn new(dataset: Arc<Dataset>) -> Self {
        Self {
            dataset,
            condition: None,
            updates: HashMap::new(),
            conflict_retries: 10,
            retry_timeout: Duration::from_secs(30),
        }
    }

    fn filterable_schema(dataset_schema: &lance_core::datatypes::Schema) -> ArrowSchema {
        let extra_columns = ArrowSchema::new(vec![
            ROW_ID_FIELD.clone(),
            ROW_ADDR_FIELD.clone(),
            ROW_OFFSET_FIELD.clone(),
        ]);
        let merged = dataset_schema
            .merge(&extra_columns)
            .expect("Failed to merge system columns into filterable schema");
        (&merged).into()
    }

    pub fn update_where(mut self, filter: &str) -> Result<Self> {
        let filter_schema = Self::filterable_schema(self.dataset.schema());
        let planner = Planner::new(Arc::new(filter_schema));
        let expr = planner
            .parse_filter(filter)
            .map_err(box_error)
            .context(InvalidInputSnafu {})?;
        self.condition = Some(
            planner
                .optimize_expr(expr)
                .map_err(box_error)
                .context(InvalidInputSnafu {})?,
        );
        Ok(self)
    }

    pub fn set(mut self, column: impl AsRef<str>, value: &str) -> Result<Self> {
        let field = self
            .dataset
            .schema()
            .field(column.as_ref())
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Column '{}' does not exist in dataset schema: {:?}",
                    column.as_ref(),
                    self.dataset.schema()
                ))
            })?;

        // TODO: support nested column references. This is mostly blocked on the
        // ability to insert them into the RecordBatch properly.
        if column.as_ref().contains('.') {
            return Err(Error::not_supported_source(
                format!(
                    "Nested column references are not yet supported. Referenced: {}",
                    column.as_ref(),
                )
                .into(),
            ));
        }

        let schema: Arc<ArrowSchema> = Arc::new(self.dataset.schema().into());
        let planner = Planner::new(schema.clone());
        let mut expr = planner
            .parse_expr(value)
            .map_err(box_error)
            .context(InvalidInputSnafu {})?;

        // Cast expression to the column's data type if necessary.
        let dest_type = field.data_type();
        let df_schema = DFSchema::try_from(schema.as_ref().clone())?;
        let src_type = expr
            .get_type(&df_schema)
            .map_err(box_error)
            .context(InvalidInputSnafu {})?;
        if dest_type != src_type {
            expr = match expr {
                // TODO: remove this branch once DataFusion supports casting List to FSL
                // This should happen in Arrow 51.0.0
                Expr::Literal(value @ ScalarValue::List(_), metadata)
                    if matches!(dest_type, DataType::FixedSizeList(_, _)) =>
                {
                    Expr::Literal(
                        safe_coerce_scalar(&value, &dest_type).ok_or_else(|| {
                            ArrowError::CastError(format!(
                                "Failed to cast {} to {} during planning",
                                value.data_type(),
                                dest_type
                            ))
                        })?,
                        metadata,
                    )
                }
                _ => expr
                    .cast_to(&dest_type, &df_schema)
                    .map_err(box_error)
                    .context(InvalidInputSnafu {})?,
            };
        }

        // Optimize the expression. For example, this might apply the cast on
        // literals. (Expr.cast_to() only wraps the expression in a Cast node,
        // it doesn't actually apply the cast to the literals.)
        let expr = planner
            .optimize_expr(expr)
            .map_err(box_error)
            .context(InvalidInputSnafu {})?;

        self.updates.insert(column.as_ref().to_string(), expr);
        Ok(self)
    }

    /// Set the number of times to retry on commit conflicts.
    ///
    /// Default is 10.
    pub fn conflict_retries(mut self, retries: u32) -> Self {
        self.conflict_retries = retries;
        self
    }

    /// Set the total timeout for all retries.
    ///
    /// Default is 30 seconds.
    pub fn retry_timeout(mut self, timeout: Duration) -> Self {
        self.retry_timeout = timeout;
        self
    }

    // TODO: set write params
    // pub fn with_write_params(mut self, params: WriteParams) -> Self { ... }

    pub fn build(self) -> Result<UpdateJob> {
        let mut updates = HashMap::new();

        let planner = Planner::new(Arc::new(self.dataset.schema().into()));

        for (column, expr) in self.updates {
            let physical_expr = planner.create_physical_expr(&expr)?;
            updates.insert(column, physical_expr);
        }

        if updates.is_empty() {
            return Err(Error::invalid_input("No updates provided"));
        }

        let updates = Arc::new(updates);

        Ok(UpdateJob {
            dataset: self.dataset,
            condition: self.condition,
            updates,
            conflict_retries: self.conflict_retries,
            retry_timeout: self.retry_timeout,
        })
    }
}

// TODO: support distributed operation.

#[derive(Debug, Clone)]
pub struct UpdateResult {
    pub new_dataset: Arc<Dataset>,
    pub rows_updated: u64,
}

#[derive(Debug)]
pub struct UpdateData {
    removed_fragment_ids: Vec<u64>,
    old_fragments: Vec<Fragment>,
    new_fragments: Vec<Fragment>,
    affected_rows: RowAddrTreeMap,
    row_address_layout_delta: Option<RowAddressLayoutDelta>,
    num_updated_rows: u64,
}

#[derive(Debug, Clone)]
pub struct UpdateJob {
    dataset: Arc<Dataset>,
    condition: Option<Expr>,
    updates: Arc<HashMap<String, Arc<dyn PhysicalExpr>>>,
    conflict_retries: u32,
    retry_timeout: Duration,
}

impl UpdateJob {
    fn modified_field_ids(&self, dataset: &Dataset) -> Vec<u32> {
        let mut field_ids = Vec::new();
        for column_name in self.updates.keys() {
            if let Some(field) = dataset.schema().field(column_name) {
                collect_subtree_field_ids(field, &mut field_ids);
            }
        }
        field_ids.sort_unstable();
        field_ids.dedup();
        field_ids
    }

    pub async fn execute(self) -> Result<UpdateResult> {
        let dataset = self.dataset.clone();
        let config = RetryConfig {
            max_retries: self.conflict_retries,
            retry_timeout: self.retry_timeout,
        };

        Box::pin(execute_with_retry(self, dataset, config)).await
    }

    async fn execute_impl(self) -> Result<UpdateData> {
        let mut scanner = self.dataset.scan();
        scanner.with_row_id();
        scanner.blob_handling(BlobHandling::AllBinary);

        if let Some(expr) = &self.condition {
            scanner.filter_expr(expr.clone());
        }

        let uses_logical_row_addresses = self.dataset.manifest.uses_stable_logical_row_addresses();
        let plan = if uses_logical_row_addresses {
            let layout = self
                .dataset
                .manifest
                .row_address_layout
                .as_ref()
                .ok_or_else(|| {
                    Error::internal("storage-version-2.3 update is missing RowAddressLayout")
                })?;
            let fragments = self.dataset.manifest.fragments.as_ref();
            let order_plan = plan_v2_3_update_logical_order(fragments, layout)?;
            create_v2_3_update_scan_plan(&scanner, fragments, &order_plan).await?
        } else {
            scanner.create_plan().await?
        };
        let stream = execute_plan(plan, scanner.execution_options())?;

        // We keep track of seen row ids so we can delete them from the existing
        // fragments and then set the row id segments in the new fragments.
        let (stream, row_id_rx) = make_rowid_capture_stream(
            stream,
            self.dataset.manifest.uses_legacy_stable_row_ids() || uses_logical_row_addresses,
        )?;

        let schema = stream.schema();

        let expected_schema = self.dataset.schema().into();
        if schema.as_ref() != &expected_schema {
            return Err(Error::internal(format!(
                "Expected schema {:?} but got {:?}",
                expected_schema, schema
            )));
        }

        let updates_ref = self.updates.clone();
        let stream = stream
            .map(move |batch| {
                let updates = updates_ref.clone();
                tokio::task::spawn_blocking(move || Self::apply_updates(batch?, updates))
            })
            .buffered(get_num_compute_intensive_cpus())
            .map(|res| match res {
                Ok(Ok(batch)) => Ok(batch),
                Ok(Err(err)) => Err(err),
                Err(e) => Err(DataFusionError::ExecutionJoin(Box::new(e))),
            });
        let stream = RecordBatchStreamAdapter::new(schema, stream);

        let mut write_params = WriteParams::with_storage_version(
            self.dataset
                .manifest()
                .data_storage_format
                .lance_file_version()?,
        );
        if uses_logical_row_addresses {
            // The preflight freezes row-address target boundaries by row
            // count. A byte-triggered early rollover would change the physical
            // fragment plan after remote writes had started.
            write_params.max_bytes_per_file = usize::MAX;
        }
        let mut captured_before_write = None;
        let mut row_address_layout_delta = None;
        let mut spill_guard: Option<(TempDir, SpillSender)> = None;
        let stream: datafusion::physical_plan::SendableRecordBatchStream =
            if uses_logical_row_addresses {
                let temp_dir = tokio::task::spawn_blocking(TempDir::try_new)
                    .await
                    .map_err(|error| {
                        Error::internal(format!("update preflight spill task failed: {error}"))
                    })??;
                let path = temp_dir.std_path().join("update-preflight.arrow");
                let (mut sender, receiver) =
                    create_replay_spill(path, stream.schema(), 100 * 1024 * 1024);
                let mut stream = Box::pin(stream);
                while let Some(batch) = stream.next().await {
                    sender.write(batch?).await?;
                }
                sender.finish().await?;
                let captured = row_id_rx.try_recv().map_err(|error| {
                    Error::internal(format!("Failed to receive row ids: {error}"))
                })?;
                let logical_row_ids = captured.row_id_sequence().ok_or_else(|| {
                    Error::internal("storage-version-2.3 update captured physical row addresses")
                })?;
                row_address_layout_delta = Some(
                    preflight_v2_3_update(
                        self.dataset.as_ref(),
                        logical_row_ids,
                        &self.modified_field_ids(self.dataset.as_ref()),
                        write_params.max_rows_per_file,
                    )
                    .await?,
                );
                captured_before_write = Some(captured);
                spill_guard = Some((temp_dir, sender));
                receiver.read()
            } else {
                Box::pin(stream)
            };

        let (mut new_fragments, _) = write_fragments_internal(
            Some(&self.dataset),
            self.dataset.object_store.clone(),
            &self.dataset.base,
            self.dataset.schema().clone(),
            stream,
            write_params,
            None, // TODO: support multiple bases for update
        )
        .await?;

        let removed_row_ids = if let Some(captured) = captured_before_write {
            captured
        } else {
            row_id_rx
                .try_recv()
                .map_err(|err| Error::internal(format!("Failed to receive row ids: {}", err)))?
        };
        drop(spill_guard);

        if !uses_logical_row_addresses
            && let Some(row_id_sequence) = removed_row_ids.row_id_sequence()
        {
            let fragment_sizes = new_fragments
                .iter()
                .map(|f| f.physical_rows.unwrap() as u64);
            let sequences = lance_table::rowids::rechunk_sequences(
                [row_id_sequence.clone()],
                fragment_sizes,
                false,
            )
            .map_err(|e| {
                Error::internal(format!(
                    "Captured row ids not equal to number of rows written: {}",
                    e
                ))
            })?;
            for (fragment, sequence) in new_fragments.iter_mut().zip(sequences) {
                let serialized = lance_table::rowids::write_row_ids(&sequence);
                fragment.row_id_meta = Some(RowIdMeta::Inline(serialized));
            }
        }

        // Apply deletions
        let logical_row_ids = uses_logical_row_addresses
            .then(|| removed_row_ids.row_id_sequence().cloned())
            .flatten();
        let row_addrs = if let Some(logical_row_ids) = &logical_row_ids {
            let raw_ids = logical_row_ids.iter().collect::<Vec<_>>();
            let resolved = self.dataset.resolve_logical_row_ids_async(&raw_ids).await?;
            let mut addresses = RoaringTreemap::new();
            for (row_id, address) in raw_ids.into_iter().zip(resolved) {
                let address = address.ok_or_else(|| {
                    Error::internal(format!(
                        "updated logical row id {row_id} has no current physical placement"
                    ))
                })?;
                addresses.insert(u64::from(address));
            }
            std::borrow::Cow::Owned(addresses)
        } else {
            let row_id_index = get_row_id_index(&self.dataset).await?;
            removed_row_ids.row_addrs(row_id_index.as_deref())
        };
        let deletions_result = self.apply_deletions(&row_addrs).await;
        let (old_fragments, removed_fragment_ids) = match deletions_result {
            Ok(v) => v,
            Err(e) => {
                cleanup_data_fragments(
                    &self.dataset.object_store,
                    &self.dataset.base,
                    None,
                    &new_fragments,
                )
                .await;
                return Err(e);
            }
        };
        let affected_rows = RowAddrTreeMap::from(row_addrs.as_ref().clone());

        let num_updated_rows = new_fragments
            .iter()
            .map(|f| f.physical_rows.unwrap() as u64)
            .sum::<u64>();

        Ok(UpdateData {
            removed_fragment_ids,
            old_fragments,
            new_fragments,
            affected_rows,
            row_address_layout_delta,
            num_updated_rows,
        })
    }

    async fn commit_impl(
        &self,
        dataset: Arc<Dataset>,
        update_data: UpdateData,
    ) -> Result<UpdateResult> {
        // Updated columns are top-level (nested references are rejected by `set`), but a
        // struct-column update rewrites all of its descendants. Collect the full field
        // subtree so an index on a nested child field is recognized as modified and not
        // wrongly extended over the rewritten fragment.
        let fields_for_preserving_frag_bitmap = self.modified_field_ids(dataset.as_ref());

        let row_address_layout_delta = update_data.row_address_layout_delta;
        let fields_modified = row_address_layout_delta
            .as_ref()
            .map(|_| fields_for_preserving_frag_bitmap.clone())
            .unwrap_or_default();

        // Commit updated and new fragments
        let operation = Operation::Update {
            removed_fragment_ids: update_data.removed_fragment_ids,
            updated_fragments: update_data.old_fragments,
            new_fragments: update_data.new_fragments,
            // In "rewrite rows" mode, the rows that are updated in the fragment
            // are moved(deleted and appended).
            // so we do not need to handle the frag bitmap of the index about it.
            fields_modified,
            merged_generations: Vec::new(),
            fields_for_preserving_frag_bitmap,
            update_mode: Some(RewriteRows),
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        };

        let transaction = TransactionBuilder::new(dataset.manifest.version, operation)
            .row_address_layout_delta(row_address_layout_delta)
            .build();

        let new_dataset = CommitBuilder::new(dataset)
            .with_affected_rows(update_data.affected_rows)
            .execute(transaction)
            .await?;

        Ok(UpdateResult {
            new_dataset: Arc::new(new_dataset),
            rows_updated: update_data.num_updated_rows,
        })
    }

    fn apply_updates(
        mut batch: RecordBatch,
        updates: Arc<HashMap<String, Arc<dyn PhysicalExpr>>>,
    ) -> DFResult<RecordBatch> {
        for (column, expr) in updates.iter() {
            let new_values = expr.evaluate(&batch)?.into_array(batch.num_rows())?;
            batch = batch.replace_column_by_name(column.as_str(), new_values)?;
        }
        Ok(batch)
    }

    /// Use previous found rows ids to delete rows from existing fragments.
    ///
    /// Returns the set of modified fragments and removed fragments, if any.
    async fn apply_deletions(
        &self,
        removed_row_addrs: &RoaringTreemap,
    ) -> Result<(Vec<Fragment>, Vec<u64>)> {
        let bitmaps = Arc::new(removed_row_addrs.bitmaps().collect::<BTreeMap<_, _>>());

        enum FragmentChange {
            Unchanged,
            Modified(Box<Fragment>),
            Removed(u64),
        }

        let mut updated_fragments = Vec::new();
        let mut removed_fragments = Vec::new();

        let mut stream = futures::stream::iter(self.dataset.get_fragments())
            .map(move |fragment| {
                let bitmaps_ref = bitmaps.clone();
                async move {
                    let fragment_id = fragment.id();
                    if let Some(bitmap) = bitmaps_ref.get(&(fragment_id as u32)) {
                        match fragment.extend_deletions(*bitmap).await {
                            Ok(Some(new_fragment)) => {
                                Ok(FragmentChange::Modified(Box::new(new_fragment.metadata)))
                            }
                            Ok(None) => Ok(FragmentChange::Removed(fragment_id as u64)),
                            Err(e) => Err(e),
                        }
                    } else {
                        Ok(FragmentChange::Unchanged)
                    }
                }
            })
            .buffer_unordered(self.dataset.object_store.io_parallelism());

        while let Some(res) = stream.next().await.transpose()? {
            match res {
                FragmentChange::Unchanged => {}
                FragmentChange::Modified(fragment) => updated_fragments.push(*fragment),
                FragmentChange::Removed(fragment_id) => removed_fragments.push(fragment_id),
            }
        }

        Ok((updated_fragments, removed_fragments))
    }
}

impl RetryExecutor for UpdateJob {
    type Data = UpdateData;
    type Result = UpdateResult;

    async fn execute_impl(&self) -> Result<Self::Data> {
        self.clone().execute_impl().await
    }

    async fn commit(&self, dataset: Arc<Dataset>, data: Self::Data) -> Result<Self::Result> {
        self.commit_impl(dataset, data).await
    }

    fn update_dataset(&mut self, dataset: Arc<Dataset>) {
        self.dataset = dataset;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use crate::{
        dataset::{InsertBuilder, ReadParams, WriteParams, builder::DatasetBuilder},
        session::Session,
        utils::test::ThrottledStoreWrapper,
    };

    use super::*;

    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::dataset::{WriteDestination, WriteMode};
    use crate::index::DatasetIndexExt;
    use crate::index::vector::VectorIndexParams;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow::{
        array::AsArray,
        datatypes::{Int64Type, UInt32Type},
    };
    use arrow_array::types::Float32Type;
    use arrow_array::{Int64Array, RecordBatchIterator, StringArray, UInt32Array, UInt64Array};
    use arrow_schema::{Field, Schema as ArrowSchema};
    use arrow_select::concat::concat_batches;
    use datafusion::physical_plan::displayable;
    use futures::{TryStreamExt, future::try_join_all};
    use lance_arrow::ARROW_EXT_NAME_KEY;
    use lance_arrow::json::{ARROW_JSON_EXT_NAME, is_arrow_json_field, is_json_field};
    use lance_core::ROW_ID;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::{Dimension, RowCount};
    use lance_file::version::LanceFileVersion;
    use lance_index::IndexType;
    use lance_index::scalar::ScalarIndexParams;
    use lance_io::object_store::ObjectStoreParams;
    use lance_linalg::distance::MetricType;
    use object_store::throttle::ThrottleConfig;
    use rstest::rstest;
    use tokio::sync::Barrier;

    /// Returns a dataset with 3 fragments, each with 10 rows.
    ///
    /// Also returns the TempDir, which should be kept alive as long as the
    /// dataset is being accessed. Once that is dropped, the temp directory is
    /// deleted.
    async fn make_test_dataset(
        version: LanceFileVersion,
        enable_stable_row_ids: bool,
    ) -> (Arc<Dataset>, TempStrDir) {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(0..30)),
                Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                    "foo", 30,
                ))),
            ],
        )
        .unwrap();

        let write_params = WriteParams {
            max_rows_per_file: 10,
            data_storage_version: Some(version),
            enable_stable_row_ids,
            ..Default::default()
        };

        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let batches = RecordBatchIterator::new([Ok(batch)], schema.clone());
        let ds = Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        (Arc::new(ds), test_dir)
    }

    #[tokio::test]
    async fn test_update_validation() {
        let (dataset, _test_dir) = make_test_dataset(LanceFileVersion::Legacy, false).await;

        let builder = UpdateBuilder::new(dataset);

        assert!(
            matches!(
                builder.clone().update_where("foo = 10"),
                Err(Error::InvalidInput { .. })
            ),
            "Should return error if condition references non-existent column"
        );

        assert!(
            matches!(
                builder.clone().set("foo", "1"),
                Err(Error::InvalidInput { .. })
            ),
            "Should return error if update key references non-existent column"
        );

        assert!(
            matches!(
                builder.clone().set("id", "id2 + 1"),
                Err(Error::InvalidInput { .. })
            ),
            "Should return error if update expression references non-existent column"
        );

        assert!(
            matches!(builder.build(), Err(Error::InvalidInput { .. })),
            "Should return error if no update expressions are provided"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_update_all(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::V2_0)] version: LanceFileVersion,
        #[values(false, true)] enable_stable_row_ids: bool,
    ) {
        let (dataset, _test_dir) = make_test_dataset(version, enable_stable_row_ids).await;

        let update_result = UpdateBuilder::new(dataset)
            .set("name", "'bar' || cast(id as string)")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        let dataset = update_result.new_dataset;
        let actual_batches = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let actual_batch = concat_batches(&actual_batches[0].schema(), &actual_batches).unwrap();

        let expected = RecordBatch::try_new(
            Arc::new(dataset.schema().into()),
            vec![
                Arc::new(Int64Array::from_iter_values(0..30)),
                Arc::new(StringArray::from_iter_values(
                    (0..30).map(|i| format!("bar{}", i)),
                )),
            ],
        )
        .unwrap();

        assert_eq!(actual_batch, expected);

        assert_eq!(dataset.get_fragments().len(), 1);
    }

    #[rstest]
    #[tokio::test]
    async fn test_update_conditional(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::V2_0)] version: LanceFileVersion,
        #[values(false, true)] enable_stable_row_ids: bool,
    ) {
        let (dataset, _test_dir) = make_test_dataset(version, enable_stable_row_ids).await;

        let original_fragments = dataset.get_fragments();

        let update_result = UpdateBuilder::new(dataset)
            .update_where("id >= 15")
            .unwrap()
            .set("name", "'bar' || cast(id as string)")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        let dataset = update_result.new_dataset;
        let actual_batches = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let actual_batch = concat_batches(&actual_batches[0].schema(), &actual_batches).unwrap();

        let expected = RecordBatch::try_new(
            Arc::new(dataset.schema().into()),
            vec![
                Arc::new(Int64Array::from_iter_values(0..30)),
                Arc::new(StringArray::from_iter_values(
                    (0..15)
                        .map(|_| "foo".to_string())
                        .chain((15..30).map(|i| format!("bar{}", i))),
                )),
            ],
        )
        .unwrap();

        assert_eq!(actual_batch, expected);

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 3);

        // One fragment not touched (id = 0..10)
        assert_eq!(fragments[0].metadata.id, original_fragments[0].metadata.id);
        assert_eq!(
            fragments[0].metadata.files,
            original_fragments[0].metadata.files
        );
        assert_eq!(
            fragments[0].metadata.physical_rows,
            original_fragments[0].metadata.physical_rows
        );
        assert_eq!(
            fragments[0].metadata.row_id_meta,
            original_fragments[0].metadata.row_id_meta
        );
        // One fragment partially modified (id = 10..15)
        assert_eq!(
            fragments[1].metadata.files,
            original_fragments[1].metadata.files,
        );
        assert_eq!(
            fragments[1]
                .metadata
                .deletion_file
                .as_ref()
                .and_then(|f| f.num_deleted_rows),
            Some(5)
        );
        // One fragment fully modified
        assert_eq!(fragments[2].metadata.physical_rows, Some(15));
    }

    #[tokio::test]
    async fn test_update_json_and_regular_columns() {
        let mut metadata = HashMap::new();
        metadata.insert(
            ARROW_EXT_NAME_KEY.to_string(),
            ARROW_JSON_EXT_NAME.to_string(),
        );
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("meta", DataType::Utf8, true).with_metadata(metadata),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values([1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
                Arc::new(StringArray::from(vec![
                    r#"{"before":1}"#,
                    r#"{"before":2}"#,
                    r#"{"before":3}"#,
                ])),
            ],
        )
        .unwrap();

        let test_dir = TempStrDir::default();
        let batches = RecordBatchIterator::new([Ok(batch)], schema);
        let dataset = Arc::new(
            Dataset::write(batches, &test_dir, Some(WriteParams::default()))
                .await
                .unwrap(),
        );

        let physical_schema: ArrowSchema = dataset.schema().into();
        assert!(is_json_field(
            physical_schema.field_with_name("meta").unwrap()
        ));

        let update_result = UpdateBuilder::new(dataset)
            .update_where("id = 2")
            .unwrap()
            .set("name", "'updated'")
            .unwrap()
            .set("meta", r#"jsonb '{"after":true,"n":2}'"#)
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        let updated_dataset = update_result.new_dataset;
        let actual_batches = updated_dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let actual_batch = concat_batches(&actual_batches[0].schema(), &actual_batches).unwrap();
        assert!(is_arrow_json_field(
            actual_batch.schema().field_with_name("meta").unwrap()
        ));

        let ids = actual_batch["id"].as_primitive::<Int64Type>();
        let names = actual_batch["name"].as_string::<i32>();
        let metas = actual_batch["meta"].as_string::<i32>();
        let updated_row_idx = ids.iter().position(|id| id == Some(2)).unwrap();

        assert_eq!(names.value(updated_row_idx), "updated");
        assert_eq!(metas.value(updated_row_idx), r#"{"after":true,"n":2}"#);
    }

    #[rstest]
    #[tokio::test]
    async fn test_update_concurrency(#[values(false, true)] enable_stable_row_ids: bool) {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::UInt32, false),
            Field::new("value", DataType::UInt32, false),
        ]));
        let concurrency = 3;
        let initial_data = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from_iter_values(0..concurrency)),
                Arc::new(UInt32Array::from_iter_values(std::iter::repeat_n(
                    0,
                    concurrency as usize,
                ))),
            ],
        )
        .unwrap();

        // Increase likelihood of contention by throttling the store
        let throttled = Arc::new(ThrottledStoreWrapper {
            config: ThrottleConfig {
                wait_list_per_call: Duration::from_millis(1),
                wait_get_per_call: Duration::from_millis(1),
                ..Default::default()
            },
        });
        let session = Arc::new(Session::default());

        let mut dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                store_params: Some(ObjectStoreParams {
                    object_store_wrapper: Some(throttled.clone()),
                    ..Default::default()
                }),
                session: Some(session.clone()),
                enable_stable_row_ids,
                ..Default::default()
            })
            .execute(vec![initial_data])
            .await
            .unwrap();

        let barrier = Arc::new(Barrier::new(concurrency as usize));
        let mut handles = Vec::new();
        for i in 0..concurrency {
            let session_ref = session.clone();
            let barrier_ref = barrier.clone();
            let throttled_ref = throttled.clone();
            let handle = tokio::task::spawn(async move {
                let dataset = DatasetBuilder::from_uri("memory://")
                    .with_read_params(ReadParams {
                        store_options: Some(ObjectStoreParams {
                            object_store_wrapper: Some(throttled_ref.clone()),
                            ..Default::default()
                        }),
                        session: Some(session_ref.clone()),
                        ..Default::default()
                    })
                    .load()
                    .await
                    .unwrap();

                let job = UpdateBuilder::new(Arc::new(dataset))
                    .update_where(&format!("id = {}", i))
                    .unwrap()
                    .set("value", "1")
                    .unwrap()
                    .build()
                    .unwrap();
                barrier_ref.wait().await;

                job.execute().await.unwrap();
            });
            handles.push(handle);
        }

        try_join_all(handles).await.unwrap();

        dataset.checkout_latest().await.unwrap();

        let data = dataset.scan().try_into_batch().await.unwrap();

        let mut ids = data["id"]
            .as_primitive::<UInt32Type>()
            .values()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        ids.sort();
        assert_eq!(ids, vec![0, 1, 2],);
        let values = data["value"].as_primitive::<UInt32Type>().values();
        assert!(values.iter().all(|&value| value == 1));
    }

    #[rstest]
    #[tokio::test]
    async fn test_update_same_row_concurrency(#[values(false, true)] enable_stable_row_ids: bool) {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::UInt32, false),
            Field::new("value", DataType::UInt32, false),
        ]));
        let concurrency = 3;
        // Create dataset with just one row that all workers will update
        let initial_data = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![0])),
                Arc::new(UInt32Array::from(vec![10])),
            ],
        )
        .unwrap();

        // Increase likelihood of contention by throttling the store
        let throttled = Arc::new(ThrottledStoreWrapper {
            config: ThrottleConfig {
                wait_list_per_call: Duration::from_millis(10),
                wait_get_per_call: Duration::from_millis(10),
                ..Default::default()
            },
        });
        let session = Arc::new(Session::default());

        let mut dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                store_params: Some(ObjectStoreParams {
                    object_store_wrapper: Some(throttled.clone()),
                    ..Default::default()
                }),
                session: Some(session.clone()),
                enable_stable_row_ids,
                ..Default::default()
            })
            .execute(vec![initial_data])
            .await
            .unwrap();

        let barrier = Arc::new(Barrier::new(concurrency as usize));
        let mut handles = Vec::new();
        for _i in 0..concurrency {
            let session_ref = session.clone();
            let barrier_ref = barrier.clone();
            let throttled_ref = throttled.clone();
            let handle = tokio::task::spawn(async move {
                let dataset = DatasetBuilder::from_uri("memory://")
                    .with_read_params(ReadParams {
                        store_options: Some(ObjectStoreParams {
                            object_store_wrapper: Some(throttled_ref.clone()),
                            ..Default::default()
                        }),
                        session: Some(session_ref.clone()),
                        ..Default::default()
                    })
                    .load()
                    .await
                    .unwrap();

                let job = UpdateBuilder::new(Arc::new(dataset))
                    .update_where("id = 0")
                    .unwrap()
                    .set("value", "99")
                    .unwrap()
                    .build()
                    .unwrap();
                barrier_ref.wait().await;

                job.execute().await.unwrap();
            });
            handles.push(handle);
        }

        try_join_all(handles).await.unwrap();

        dataset.checkout_latest().await.unwrap();

        let data = dataset.scan().try_into_batch().await.unwrap();

        // With retry-based conflict resolution, all concurrent updates should succeed
        // Even though they all target the same row, they should not fail with commit conflicts
        // The final result should be exactly one row (not duplicated) because the retries
        // should work from the latest dataset state, preventing duplicate row creation
        let ids = data["id"].as_primitive::<UInt32Type>().values();
        assert_eq!(ids, &[0]);

        let values = data["value"].as_primitive::<UInt32Type>().values();
        assert_eq!(values, &[99]);
    }

    #[tokio::test]
    async fn test_row_ids_stable_after_update() {
        let (dataset, _test_dir) = make_test_dataset(LanceFileVersion::V2_0, true).await;

        let orig_batch = dataset.scan().with_row_id().try_into_batch().await.unwrap();
        let orig_row_ids = orig_batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let orig_ids = orig_batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        let updated_batch = UpdateBuilder::new(dataset)
            .update_where("id >= 15")
            .unwrap()
            .set("name", "'updated'")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset
            .scan()
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();

        let updated_row_ids = updated_batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let updated_ids = updated_batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        assert_eq!(orig_row_ids, updated_row_ids);
        assert_eq!(orig_ids, updated_ids);
    }

    #[tokio::test]
    async fn test_row_ids_stable_after_update_odd_id() {
        use std::collections::HashSet;

        let (dataset, _test_dir) = make_test_dataset(LanceFileVersion::V2_0, true).await;

        let orig_batch = dataset.scan().with_row_id().try_into_batch().await.unwrap();
        let orig_row_ids = orig_batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let orig_ids = orig_batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let orig_names = orig_batch
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        let updated_batch = UpdateBuilder::new(dataset)
            .update_where("id % 2 = 1")
            .unwrap()
            .set("name", "'updated'")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset
            .scan()
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();

        let updated_row_ids = updated_batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let updated_ids = updated_batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let updated_names = updated_batch
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        assert_eq!(
            orig_row_ids
                .values()
                .iter()
                .cloned()
                .collect::<HashSet<_>>(),
            updated_row_ids
                .values()
                .iter()
                .cloned()
                .collect::<HashSet<_>>()
        );
        assert_eq!(
            orig_ids.values().iter().cloned().collect::<HashSet<_>>(),
            updated_ids.values().iter().cloned().collect::<HashSet<_>>()
        );

        for i in 0..orig_row_ids.len() {
            let row_id = orig_row_ids.value(i);
            let updated_idx = updated_row_ids
                .iter()
                .position(|rid| rid == Some(row_id))
                .unwrap();
            let id = orig_ids.value(i);
            let updated_name = updated_names.value(updated_idx);
            if id % 2 == 1 {
                assert_eq!(updated_name, "updated");
            } else {
                assert_eq!(updated_name, orig_names.value(i));
            }
        }
    }

    #[tokio::test]
    async fn test_update_affects_index_fragment_bitmap() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "str",
                lance_datagen::array::cycle_utf8_literals(&["a", "b", "c", "d", "e", "f"]),
            )
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(4)),
            )
            .into_ram_dataset_with_params(
                FragmentCount::from(2),
                FragmentRowCount::from(3),
                Some(WriteParams {
                    max_rows_per_file: 3,
                    enable_stable_row_ids: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

        let scalar_params = ScalarIndexParams::default();
        dataset
            .create_index(
                &["str"],
                IndexType::Scalar,
                Some("str_idx".to_string()),
                &scalar_params,
                true,
            )
            .await
            .unwrap();

        let vector_params = VectorIndexParams::ivf_flat(1, MetricType::L2);
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some("vec_idx".to_string()),
                &vector_params,
                true,
            )
            .await
            .unwrap();

        let indices = dataset.load_indices().await.unwrap();
        let str_index = indices.iter().find(|idx| idx.name == "str_idx").unwrap();
        let vec_index = indices.iter().find(|idx| idx.name == "vec_idx").unwrap();

        assert_eq!(
            str_index
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            vec_index
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![0, 1]
        );

        let updated_dataset = UpdateBuilder::new(Arc::new(dataset))
            .update_where("str = 'e'")
            .unwrap()
            .set("vec", "array[25.0, 26.0, 27.0, 28.0]")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset;

        let updated_indices = updated_dataset.load_indices().await.unwrap();
        let updated_str_index = updated_indices
            .iter()
            .find(|idx| idx.name == "str_idx")
            .unwrap();
        let updated_vec_index = updated_indices
            .iter()
            .find(|idx| idx.name == "vec_idx")
            .unwrap();

        let str_bitmap = updated_str_index.fragment_bitmap.as_ref().unwrap();
        assert_eq!(str_bitmap.len(), 3);
        assert_eq!(str_bitmap.iter().collect::<Vec<_>>(), vec![0, 1, 2]);

        let vec_bitmap = updated_vec_index.fragment_bitmap.as_ref().unwrap();
        assert_eq!(vec_bitmap.len(), 2);
        assert_eq!(vec_bitmap.iter().collect::<Vec<_>>(), vec![0, 1]);

        let fragments = updated_dataset.get_fragments();
        assert!(fragments.len() > 2);

        let second_fragment = &fragments[1];
        assert!(
            second_fragment
                .get_deletion_vector()
                .await
                .unwrap()
                .is_some()
        );
    }

    #[tokio::test]
    async fn test_update_mixed_indexed_unindexed_fragments() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "str",
                lance_datagen::array::cycle_utf8_literals(&["a", "b", "c", "d", "e", "f"]),
            )
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(4)),
            )
            .into_ram_dataset_with_params(
                FragmentCount::from(2),
                FragmentRowCount::from(3),
                Some(WriteParams {
                    max_rows_per_file: 3,
                    enable_stable_row_ids: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

        dataset
            .create_index(
                &["str"],
                IndexType::Scalar,
                Some("str_idx".to_string()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some("vec_idx".to_string()),
                &VectorIndexParams::ivf_flat(1, MetricType::L2),
                true,
            )
            .await
            .unwrap();

        let initial_indices = dataset.load_indices().await.unwrap();
        let str_index = initial_indices
            .iter()
            .find(|idx| idx.name == "str_idx")
            .unwrap();
        let vec_index = initial_indices
            .iter()
            .find(|idx| idx.name == "vec_idx")
            .unwrap();

        assert_eq!(
            str_index
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            vec_index
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![0, 1]
        );

        // insert data to create the third frag
        let new_batch = lance_datagen::gen_batch()
            .col(
                "str",
                lance_datagen::array::cycle_utf8_literals(&["g", "h", "i"]),
            )
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(4)),
            )
            .into_batch_rows(RowCount::from(3))
            .unwrap();

        dataset = InsertBuilder::new(WriteDestination::Dataset(Arc::new(dataset)))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                enable_stable_row_ids: true,
                ..Default::default()
            })
            .execute(vec![new_batch])
            .await
            .unwrap();

        assert_eq!(dataset.get_fragments().len(), 3);

        let indices_after_insert = dataset.load_indices().await.unwrap();
        let str_index_after_insert = indices_after_insert
            .iter()
            .find(|idx| idx.name == "str_idx")
            .unwrap();
        let vec_index_after_insert = indices_after_insert
            .iter()
            .find(|idx| idx.name == "vec_idx")
            .unwrap();

        assert_eq!(
            str_index_after_insert
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .len(),
            2
        );
        assert!(
            !str_index_after_insert
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .contains(2)
        );
        assert_eq!(
            vec_index_after_insert
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .len(),
            2
        );
        assert!(
            !vec_index_after_insert
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .contains(2)
        );

        let updated_dataset = UpdateBuilder::new(Arc::new(dataset))
            // 'a' in fragment 0，'g' in fragment 2, and frag 2 not in frag bitmap
            .update_where("str = 'a' OR str = 'g'")
            .unwrap()
            .set("vec", "array[99.0, 99.0, 99.0, 99.0]")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset;

        // reload indices
        let updated_indices = updated_dataset.load_indices().await.unwrap();
        let updated_str_index = updated_indices
            .iter()
            .find(|idx| idx.name == "str_idx")
            .unwrap();
        let updated_vec_index = updated_indices
            .iter()
            .find(|idx| idx.name == "vec_idx")
            .unwrap();

        let str_bitmap = updated_str_index.fragment_bitmap.as_ref().unwrap();
        let vec_bitmap = updated_vec_index.fragment_bitmap.as_ref().unwrap();

        assert!(updated_dataset.get_fragments().len() > 3);
        assert_eq!(str_bitmap.len(), 2);
        assert_eq!(vec_bitmap.len(), 2);

        // frag 3 not in the index's frag bitmap
        for &fragment_id in str_bitmap.iter().collect::<Vec<_>>().iter() {
            assert!(
                fragment_id < 2,
                "str index bitmap should not contain fragments with unindexed data, found fragment {}",
                fragment_id
            );
        }

        // frag 3 not in the index's frag bitmap
        for &fragment_id in vec_bitmap.iter().collect::<Vec<_>>().iter() {
            assert!(
                fragment_id < 2,
                "vec index bitmap should not contain fragments with unindexed data, found fragment {}",
                fragment_id
            );
        }
    }

    #[tokio::test]
    async fn test_update_by_rowid() {
        let (dataset, _test_dir) = make_test_dataset(LanceFileVersion::Stable, true).await;

        let orig_batch = dataset.scan().with_row_id().try_into_batch().await.unwrap();
        let orig_row_ids = orig_batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let orig_ids = orig_batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        let target_idx = 5;
        let target_row_id = orig_row_ids.value(target_idx);
        let target_id = orig_ids.value(target_idx);

        let update_result = UpdateBuilder::new(dataset)
            .update_where(&format!("_rowid = {}", target_row_id))
            .unwrap()
            .set("name", "'updated'")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        assert_eq!(update_result.rows_updated, 1);

        let updated_batch = update_result
            .new_dataset
            .scan()
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();
        let updated_ids = updated_batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let updated_names = updated_batch
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        for i in 0..updated_ids.len() {
            if updated_ids.value(i) == target_id {
                assert_eq!(updated_names.value(i), "updated");
            } else {
                assert_eq!(updated_names.value(i), "foo");
            }
        }
    }

    #[tokio::test]
    async fn test_update_by_rowid_in_list() {
        let (dataset, _test_dir) = make_test_dataset(LanceFileVersion::Stable, true).await;

        let orig_batch = dataset.scan().with_row_id().try_into_batch().await.unwrap();
        let orig_row_ids = orig_batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let orig_ids = orig_batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        let target_indices = [3, 7, 15];
        let target_row_ids: Vec<u64> = target_indices
            .iter()
            .map(|&i| orig_row_ids.value(i))
            .collect();
        let target_ids: std::collections::HashSet<i64> =
            target_indices.iter().map(|&i| orig_ids.value(i)).collect();
        let in_list: String = target_row_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        let update_result = UpdateBuilder::new(dataset)
            .update_where(&format!("_rowid IN ({})", in_list))
            .unwrap()
            .set("name", "'updated'")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        assert_eq!(update_result.rows_updated, 3);

        let updated_batch = update_result
            .new_dataset
            .scan()
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();
        let updated_ids = updated_batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let updated_names = updated_batch
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        for i in 0..updated_ids.len() {
            if target_ids.contains(&updated_ids.value(i)) {
                assert_eq!(updated_names.value(i), "updated");
            } else {
                assert_eq!(updated_names.value(i), "foo");
            }
        }
    }

    fn count_data_files(base_dir: &str) -> usize {
        let data_dir = std::path::Path::new(base_dir).join("data");
        if !data_dir.exists() {
            return 0;
        }
        std::fs::read_dir(data_dir)
            .unwrap()
            .filter(|e| e.as_ref().unwrap().path().is_file())
            .count()
    }

    /// Site 4 in PR #6320: when `UpdateJob::apply_deletions` fails after the new
    /// rewrite fragments have been written, those new data files must be cleaned up.
    #[tokio::test]
    async fn test_update_cleans_up_data_on_apply_deletions_failure() {
        use crate::utils::test::FailingProxyStore;
        use lance_io::object_store::ObjectStoreParams;

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(0..30)),
                Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                    "foo", 30,
                ))),
            ],
        )
        .unwrap();

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        // Prefix `/` so Windows drive letters (e.g. `C:`) don't get parsed as
        // the URL authority.
        let path_prefix = if test_uri.starts_with('/') { "" } else { "/" };
        let routed_uri = format!("file-object-store://{path_prefix}{test_uri}");

        let write_params = WriteParams {
            max_rows_per_file: 10,
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(batch)], schema.clone());
        Dataset::write(batches, &routed_uri, Some(write_params))
            .await
            .unwrap();

        let baseline_files = count_data_files(test_uri);
        assert!(baseline_files > 0);

        // Fail writes to `_deletions/`: this is where `apply_deletions` writes
        // the new deletion file. The rewrite fragments (in `data/`) are written
        // earlier and should be successfully created, then cleaned up on failure.
        let failing = Arc::new(FailingProxyStore::new());
        failing.fail_when("put", "_deletions", "injected deletions failure");
        failing.fail_when("put_multipart", "_deletions", "injected deletions failure");

        let dataset = DatasetBuilder::from_uri(&routed_uri)
            .with_read_params(ReadParams {
                store_options: Some(ObjectStoreParams {
                    object_store_wrapper: Some(failing.clone()),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .load()
            .await
            .unwrap();

        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id < 5")
            .unwrap()
            .set("name", "'bar'")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await;

        assert!(
            result.is_err(),
            "Update should fail when deletion-file write fails"
        );

        assert_eq!(
            count_data_files(test_uri),
            baseline_files,
            "Rewritten data files should be cleaned up on apply_deletions failure"
        );
    }

    #[tokio::test]
    async fn test_update_with_blob() {
        use arrow_array::LargeBinaryArray;
        use arrow_schema::Field;
        use lance_arrow::BLOB_META_KEY;

        let test_dir = TempStrDir::default();
        let blob_meta = HashMap::from([(BLOB_META_KEY.to_string(), "true".to_string())]);
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("blobs", DataType::LargeBinary, true).with_metadata(blob_meta),
            Field::new("id", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(LargeBinaryArray::from(vec![
                    Some(b"foo".as_slice()),
                    Some(b"bar".as_slice()),
                    Some(b"baz".as_slice()),
                ])),
                Arc::new(Int64Array::from(vec![0, 1, 2])),
            ],
        )
        .unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let dataset = Dataset::write(
            reader,
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Perform an update: update the "blobs" column where id = 1
        let dataset = Arc::new(dataset);
        let updated_dataset = UpdateBuilder::new(dataset)
            .update_where("id = 1")
            .unwrap()
            .set("blobs", "arrow_cast('updated_bar', 'LargeBinary')")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset;

        // Verify the updated value
        let mut scanner = updated_dataset.scan();
        // Read as binary to assert actual value
        scanner.blob_handling(BlobHandling::AllBinary);
        let batches = scanner.try_into_batch().await.unwrap();
        let blobs = batches.column_by_name("blobs").unwrap().as_binary::<i64>();
        let ids = batches
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int64Type>();

        // Find the index of id = 1
        let idx = ids.values().iter().position(|&x| x == 1).unwrap();
        assert_eq!(blobs.value(idx), b"updated_bar");

        let idx_foo = ids.values().iter().position(|&x| x == 0).unwrap();
        assert_eq!(blobs.value(idx_foo), b"foo");
    }

    #[tokio::test]
    async fn test_v23_update_after_update_compaction_uses_sort_preserving_merge() {
        let test_dir = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(0..20)),
                Arc::new(Int64Array::from_iter_values(0..20)),
            ],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                max_rows_per_file: 4,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["id"],
                IndexType::Scalar,
                Some("id_idx".to_owned()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        let first = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id % 3 = 0")
            .unwrap()
            .set("value", "value + 100")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset;
        let mut compacted = first.as_ref().clone();
        compact_files(
            &mut compacted,
            CompactionOptions {
                target_rows_per_fragment: 20,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(compacted.manifest.fragments.len(), 1);

        let second = UpdateBuilder::new(Arc::new(compacted))
            .update_where("id % 5 = 0")
            .unwrap()
            .set("value", "value + 1000")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset;
        let layout = second.manifest.row_address_layout.as_ref().unwrap();
        let fragments = second.manifest.fragments.as_ref();
        let order_plan = plan_v2_3_update_logical_order(fragments, layout).unwrap();
        assert!(!order_plan.requires_full_logical_sort);
        assert!(
            !order_plan.logical_run_ends.is_empty(),
            "the rewritten logical stripe must form a second monotonic run"
        );

        let mut scanner = second.scan();
        scanner.with_row_id();
        scanner.blob_handling(BlobHandling::AllBinary);
        scanner.filter("id = 7").unwrap();
        let index_evaluations = std::sync::atomic::AtomicUsize::new(0);
        let physical_plan = create_v2_3_update_scan_plan_impl(
            &scanner,
            fragments,
            &order_plan,
            Some(&index_evaluations),
        )
        .await
        .unwrap();
        let plan_text = displayable(physical_plan.as_ref()).indent(true).to_string();
        assert!(
            plan_text.contains("SortPreservingMergeExec"),
            "expected a sort-preserving logical merge, got:\n{plan_text}"
        );
        assert!(
            !plan_text
                .lines()
                .any(|line| line.trim_start().starts_with("SortExec:")),
            "fast placement codecs must not globally sort UPDATE input:\n{plan_text}"
        );
        assert!(
            !plan_text.contains("ScalarIndexQuery"),
            "the split plan must consume the precomputed index result:\n{plan_text}"
        );
        assert_eq!(
            index_evaluations.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "scalar-index evaluation and logical-to-physical translation must initialize once"
        );
        let selected = execute_plan(physical_plan, scanner.execution_options())
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(selected.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
        assert_eq!(
            index_evaluations.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "executing split runs must not repeat scalar-index evaluation or translation"
        );

        let third = UpdateBuilder::new(second)
            .update_where("id = 7")
            .unwrap()
            .set("value", "value + 10000")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset;
        let batch = third.scan().try_into_batch().await.unwrap();
        let values = batch["id"]
            .as_primitive::<Int64Type>()
            .values()
            .iter()
            .copied()
            .zip(
                batch["value"]
                    .as_primitive::<Int64Type>()
                    .values()
                    .iter()
                    .copied(),
            )
            .collect::<HashMap<_, _>>();
        for id in 0..20_i64 {
            let expected = id
                + if id % 3 == 0 { 100 } else { 0 }
                + if id % 5 == 0 { 1000 } else { 0 }
                + if id == 7 { 10000 } else { 0 };
            assert_eq!(values[&id], expected, "id={id}");
        }
    }

    #[test]
    fn test_v23_update_order_plan_falls_back_for_explicit_map() {
        let source = lance_table::format::RowAddressLogicalDomain::new(4, 2, 1).unwrap();
        let selection = LogicalRowAddressSelection::from_full_domains(&[source]).unwrap();
        let explicit = lance_table::format::ExplicitMapRowAddressPlacement {
            sources: vec![lance_table::format::SparseSelectionSource {
                source,
                selection: Arc::new(selection),
                excluded: None,
            }],
            object_path: "data/_row_addresses/locator.lance".to_owned(),
            object_size: 128,
            pages: vec![lance_table::format::ExplicitMapPage {
                first_logical_address: 4_u64 << 32,
                last_logical_address: (4_u64 << 32) | 1,
                row_start: 0,
                row_count: 2,
                content_fingerprint: vec![11; 16],
            }],
            destinations: vec![lance_table::format::ExplicitMapDestination {
                physical_fragment_id: 7,
                destination_start: 0,
                row_count: 2,
                row_id_file_path: "data/_row_addresses/row_ids.lance".to_owned(),
                row_id_file_size: 64,
                row_id_pages: vec![lance_table::format::ExplicitMapRowIdPage {
                    row_start: 0,
                    row_count: 2,
                    content_fingerprint: vec![12; 16],
                }],
            }],
            base_id: None,
        };
        let mut layout = RowAddressLayout::new(uuid::Uuid::new_v4());
        layout
            .placements
            .push(RowAddressPlacement::ExplicitMap(explicit));
        let fragments = vec![Fragment {
            id: 7,
            files: Vec::new(),
            deletion_file: None,
            row_id_meta: None,
            physical_rows: Some(2),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            native_logical_domain: None,
        }];

        let order_plan = plan_v2_3_update_logical_order(&fragments, &layout).unwrap();
        assert!(order_plan.requires_full_logical_sort);
        assert!(order_plan.logical_run_ends.is_empty());
    }

    #[test]
    fn test_v23_update_order_plan_uses_fast_codec_routing_summaries() {
        use lance_table::format::{
            DirectRowAddressPlacement, ExtentListRowAddressPlacement, PackedRunRowAddressPlacement,
            RowAddressExtent, RowAddressLogicalDomain, SelectedRowAddressPlacement,
            SparseSelectionRowAddressPlacement, SparseSelectionSource,
        };

        let domain =
            |logical_fragment_id| RowAddressLogicalDomain::new(logical_fragment_id, 4, 1).unwrap();
        let selection = |logical_fragment_id, start_slot, end_slot| {
            Arc::new(
                LogicalRowAddressSelection::from_ranges(vec![
                    lance_table::format::LogicalRowAddressRange::new(
                        logical_fragment_id,
                        start_slot,
                        end_slot,
                    ),
                ])
                .unwrap(),
            )
        };
        let mut layout = RowAddressLayout::new(uuid::Uuid::new_v4());
        layout
            .placements
            .push(RowAddressPlacement::Direct(DirectRowAddressPlacement {
                source: domain(0),
                destination_fragment_id: 0,
                destination_start: 0,
                excluded: Some(selection(0, 0, 1)),
            }));
        layout.placements.push(RowAddressPlacement::PackedRun(
            PackedRunRowAddressPlacement::from_sources(vec![domain(1), domain(2)], 1, 0).unwrap(),
        ));
        layout
            .placements
            .push(RowAddressPlacement::Selected(SelectedRowAddressPlacement {
                source: domain(3),
                selection: selection(3, 0, 4),
                destination_fragment_id: 2,
                destination_start: 0,
                excluded: Some(selection(3, 3, 4)),
            }));
        layout.placements.push(RowAddressPlacement::ExtentList(
            ExtentListRowAddressPlacement {
                source: domain(4),
                extents: vec![RowAddressExtent {
                    source_start: 1,
                    length: 2,
                    destination_fragment_id: 3,
                    destination_start: 0,
                }],
            },
        ));
        layout.placements.push(RowAddressPlacement::SparseSelection(
            SparseSelectionRowAddressPlacement {
                sources: vec![
                    SparseSelectionSource {
                        source: domain(5),
                        selection: selection(5, 0, 4),
                        excluded: Some(selection(5, 3, 4)),
                    },
                    SparseSelectionSource {
                        source: domain(6),
                        selection: selection(6, 0, 4),
                        excluded: Some(selection(6, 0, 1)),
                    },
                ],
                destination_fragment_id: 4,
                destination_start: 0,
            },
        ));
        let fragments = (0..5)
            .map(|id| Fragment {
                id,
                files: Vec::new(),
                deletion_file: None,
                row_id_meta: None,
                physical_rows: Some(if id == 1 || id == 4 { 8 } else { 4 }),
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                native_logical_domain: None,
            })
            .collect::<Vec<_>>();

        let order_plan = plan_v2_3_update_logical_order(&fragments, &layout).unwrap();
        assert!(!order_plan.requires_full_logical_sort);
        assert!(order_plan.logical_run_ends.is_empty());
    }

    #[test]
    fn test_v23_update_order_plan_falls_back_for_non_monotonic_fragment() {
        let source = lance_table::format::RowAddressLogicalDomain::new(4, 4, 1).unwrap();
        let mut layout = RowAddressLayout::new(uuid::Uuid::new_v4());
        layout.placements.push(RowAddressPlacement::ExtentList(
            lance_table::format::ExtentListRowAddressPlacement {
                source,
                extents: vec![
                    lance_table::format::RowAddressExtent {
                        source_start: 0,
                        length: 2,
                        destination_fragment_id: 7,
                        destination_start: 2,
                    },
                    lance_table::format::RowAddressExtent {
                        source_start: 2,
                        length: 2,
                        destination_fragment_id: 7,
                        destination_start: 0,
                    },
                ],
            },
        ));
        let fragments = vec![Fragment {
            id: 7,
            files: Vec::new(),
            deletion_file: None,
            row_id_meta: None,
            physical_rows: Some(4),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            native_logical_domain: None,
        }];

        let order_plan = plan_v2_3_update_logical_order(&fragments, &layout).unwrap();
        assert!(order_plan.requires_full_logical_sort);
        assert!(order_plan.logical_run_ends.is_empty());
    }

    #[tokio::test]
    async fn test_v23_repeated_updates_preserve_logical_row_addresses() {
        let test_dir = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(0..20)),
                Arc::new(Int64Array::from_iter_values(0..20)),
            ],
        )
        .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                max_rows_per_file: 10,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        async fn addresses_by_id(dataset: &Dataset) -> HashMap<i64, u64> {
            let mut scanner = dataset.scan();
            scanner.project(&["id", ROW_ID]).unwrap();
            let batch = scanner.try_into_batch().await.unwrap();
            batch["id"]
                .as_primitive::<Int64Type>()
                .values()
                .iter()
                .copied()
                .zip(
                    batch[ROW_ID]
                        .as_primitive::<arrow_array::types::UInt64Type>()
                        .values()
                        .iter()
                        .copied(),
                )
                .collect()
        }

        let original_addresses = addresses_by_id(&dataset).await;
        let first = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id % 3 = 0")
            .unwrap()
            .set("value", "value + 100")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset;
        assert_eq!(addresses_by_id(&first).await, original_addresses);

        let second = UpdateBuilder::new(first)
            .update_where("id % 3 = 0")
            .unwrap()
            .set("value", "value + 1000")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset;
        assert_eq!(addresses_by_id(&second).await, original_addresses);
        let layout = second.manifest.row_address_layout.as_ref().unwrap();
        assert!(
            layout.generation_regions.is_empty(),
            "fields without queryable indices must not retain generation regions"
        );
        let value_field_id = second.schema().field("value").unwrap().id;
        assert_eq!(
            layout
                .index_commit_floors
                .iter()
                .find(|floor| floor.field_id == value_field_id)
                .unwrap()
                .generation,
            second.manifest.version
        );

        let mut scanner = second.scan();
        scanner.project(&["id", "value"]).unwrap();
        let batch = scanner.try_into_batch().await.unwrap();
        let values = batch["id"]
            .as_primitive::<Int64Type>()
            .values()
            .iter()
            .copied()
            .zip(
                batch["value"]
                    .as_primitive::<Int64Type>()
                    .values()
                    .iter()
                    .copied(),
            )
            .collect::<HashMap<_, _>>();
        for id in 0..20_i64 {
            let expected =
                id + if id % 3 == 0 { 100 } else { 0 } + if id % 3 == 0 { 1000 } else { 0 };
            assert_eq!(values[&id], expected, "id={id}");
        }
    }

    #[tokio::test]
    async fn test_v23_update_retires_prior_deleted_owner_with_fragment() {
        let test_dir = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![0, 1])),
                Arc::new(Int64Array::from(vec![10, 11])),
            ],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let mut scanner = dataset.scan();
        scanner.project(&[ROW_ID]).unwrap();
        let row_ids = scanner.try_into_batch().await.unwrap()[ROW_ID]
            .as_primitive::<arrow_array::types::UInt64Type>()
            .values()
            .to_vec();

        dataset.delete("id = 0").await.unwrap();
        let updated = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 1")
            .unwrap()
            .set("value", "value + 100")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset;

        let logical = row_ids
            .iter()
            .copied()
            .map(LogicalRowAddress::try_from)
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let resolutions = updated
            .row_address_router()
            .unwrap()
            .resolve_many(&logical)
            .unwrap();
        assert!(matches!(
            resolutions[0],
            lance_table::format::PlacementResolution::NotLive
        ));
        assert!(matches!(
            resolutions[1],
            lance_table::format::PlacementResolution::Mapped { .. }
        ));
        updated.manifest.validate_row_address_contract().unwrap();
    }

    #[tokio::test]
    async fn test_v23_update_admission_fails_before_writing_data_objects() {
        fn file_count(path: &std::path::Path) -> usize {
            std::fs::read_dir(path)
                .into_iter()
                .flatten()
                .flatten()
                .map(|entry| {
                    if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                        file_count(&entry.path())
                    } else {
                        1
                    }
                })
                .sum()
        }

        let test_dir = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(0..20)),
                Arc::new(Int64Array::from_iter_values(0..20)),
            ],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let fragments = dataset.manifest.fragments.clone();
        let max_logical_fragment_id = dataset.manifest.max_logical_fragment_id;
        let manifest = Arc::make_mut(&mut dataset.manifest);
        let layout = Arc::make_mut(manifest.row_address_layout.as_mut().unwrap());
        layout.debt_summary.metadata_bytes_written_since_maintenance =
            lance_table::format::ROW_ADDRESS_W_FAST;
        layout.refresh_fingerprint_with_fragments(fragments.as_ref(), max_logical_fragment_id);

        let before = file_count(std::path::Path::new(test_dir.as_str()));
        let error = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id % 2 = 0")
            .unwrap()
            .set("value", "value + 1")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap_err();
        assert!(error.to_string().contains("ProjectedEpochBytes"));
        assert_eq!(
            file_count(std::path::Path::new(test_dir.as_str())),
            before,
            "placement admission must run before writing data or deletion objects"
        );
    }
}
