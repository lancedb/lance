// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, sync::Arc};

use arrow_array::{BooleanArray, RecordBatch, UInt64Array, cast::AsArray, types::UInt64Type};
use arrow_schema::SchemaRef;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_plan::execution_plan::CardinalityEffect;
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties, Statistics,
    execution_plan::{Boundedness, EmissionType},
};
use datafusion_physical_expr::EquivalenceProperties;
use futures::StreamExt;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::address::{LogicalRowAddress, RowAddress};
use lance_core::{Error, ROW_ID, ROW_ID_FIELD};
use lance_select::{RowAddrTreeMap, RowSetOps};
use lance_table::format::{
    Fragment, LogicalIndexCoverage, LogicalRowAddressRange, LogicalRowAddressSelection,
};
use roaring::{RoaringBitmap, RoaringTreemap};
use uuid::Uuid;

use crate::Dataset;
use lance_index::scalar::lance_format::OpenedIndexFile;

#[derive(Debug, Clone, Default)]
pub struct LogicalMissingRows {
    ranges: Arc<[LogicalRowAddressRange]>,
    sparse: Option<Arc<LogicalRowAddressSelection>>,
}

impl LogicalMissingRows {
    pub(crate) fn new(
        ranges: Vec<LogicalRowAddressRange>,
        sparse: Option<LogicalRowAddressSelection>,
    ) -> Self {
        Self {
            ranges: ranges.into(),
            sparse: sparse.map(Arc::new),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.ranges.is_empty() && self.sparse.as_ref().is_none_or(|rows| rows.is_empty())
    }

    fn contains_in_ranges(&self, address: LogicalRowAddress) -> bool {
        let key = (address.logical_fragment_id(), address.immutable_slot());
        let index = self
            .ranges
            .partition_point(|range| (range.logical_fragment_id, range.end_slot) <= key);
        self.ranges.get(index).is_some_and(|range| {
            range.logical_fragment_id == address.logical_fragment_id()
                && range.start_slot <= address.immutable_slot()
                && address.immutable_slot() < range.end_slot
        })
    }

    fn contains(&self, address: LogicalRowAddress) -> lance_core::Result<bool> {
        if self.contains_in_ranges(address) {
            return Ok(true);
        }
        self.sparse
            .as_ref()
            .map(|rows| rows.contains(address))
            .unwrap_or(Ok(false))
    }

    fn from_bitmap(rows: RoaringTreemap) -> lance_core::Result<Self> {
        let mut ranges = Vec::new();
        let mut sparse_domains = Vec::new();
        for (logical_fragment_id, slots) in rows.bitmaps() {
            if let (Some(start), Some(end)) = (slots.min(), slots.max())
                && slots.len() == u64::from(end - start) + 1
            {
                ranges.push(LogicalRowAddressRange::new(
                    logical_fragment_id,
                    start,
                    end + 1,
                ));
            } else if !slots.is_empty() {
                sparse_domains.push((logical_fragment_id, slots.clone()));
            }
        }
        let sparse = sparse_domains.into_iter().collect::<RoaringTreemap>();
        Ok(Self::new(
            ranges,
            (!sparse.is_empty())
                .then(|| LogicalRowAddressSelection::from_bitmap(sparse))
                .transpose()?,
        ))
    }
}

#[derive(Debug, Clone)]
pub struct LogicalCoverageSegment {
    raw_selections: Arc<[LogicalRowAddressSelection]>,
    invalidated_rows: Arc<RowAddrTreeMap>,
}

impl LogicalCoverageSegment {
    fn contains_effective(&self, row_id: u64) -> lance_core::Result<bool> {
        if self.invalidated_rows.contains(row_id) {
            return Ok(false);
        }
        let address = LogicalRowAddress::try_from(row_id)?;
        for selection in self.raw_selections.iter() {
            if selection.contains(address)? {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

#[derive(Debug, Clone)]
pub struct LogicalCoverageGroup {
    segments: Arc<[LogicalCoverageSegment]>,
    index_ids: Arc<[Uuid]>,
    covers_all_current_slots: bool,
    covered_physical_fragments: Arc<RoaringBitmap>,
    fallback_physical_fragments: Arc<RoaringBitmap>,
    fully_covered_logical_fragments: Arc<RoaringBitmap>,
    missing_logical_rows: Arc<LogicalMissingRows>,
    preopened_index_files: Arc<HashMap<Uuid, OpenedIndexFile>>,
    resolved_index_coverages: Arc<HashMap<Uuid, LogicalIndexCoverage>>,
}

impl LogicalCoverageGroup {
    pub(crate) fn from_resolved_segments(
        segments: Vec<(Vec<LogicalRowAddressSelection>, Arc<RowAddrTreeMap>)>,
    ) -> Self {
        Self {
            segments: segments
                .into_iter()
                .map(
                    |(raw_selections, invalidated_rows)| LogicalCoverageSegment {
                        raw_selections: raw_selections.into(),
                        invalidated_rows,
                    },
                )
                .collect(),
            index_ids: Arc::from([]),
            covers_all_current_slots: false,
            covered_physical_fragments: Arc::new(RoaringBitmap::new()),
            fallback_physical_fragments: Arc::new(RoaringBitmap::new()),
            fully_covered_logical_fragments: Arc::new(RoaringBitmap::new()),
            missing_logical_rows: Arc::new(LogicalMissingRows::default()),
            preopened_index_files: Arc::new(HashMap::new()),
            resolved_index_coverages: Arc::new(HashMap::new()),
        }
    }

    pub(crate) fn from_resolved_index_segments(
        index_ids: Vec<Uuid>,
        segments: Vec<(Vec<LogicalRowAddressSelection>, Arc<RowAddrTreeMap>)>,
    ) -> Self {
        let mut group = Self::from_resolved_segments(segments);
        group.index_ids = index_ids.into();
        group
    }

    pub(crate) fn all_current(
        index_ids: Vec<Uuid>,
        fully_covered_logical_fragments: RoaringBitmap,
        covered_physical_fragments: RoaringBitmap,
    ) -> Self {
        Self {
            segments: Arc::from([]),
            index_ids: index_ids.into(),
            covers_all_current_slots: true,
            covered_physical_fragments: Arc::new(covered_physical_fragments),
            fallback_physical_fragments: Arc::new(RoaringBitmap::new()),
            fully_covered_logical_fragments: Arc::new(fully_covered_logical_fragments),
            missing_logical_rows: Arc::new(LogicalMissingRows::default()),
            preopened_index_files: Arc::new(HashMap::new()),
            resolved_index_coverages: Arc::new(HashMap::new()),
        }
    }

    fn try_from_selections(
        base: &[LogicalRowAddressSelection],
        invalidated: &[LogicalRowAddressSelection],
    ) -> lance_core::Result<Self> {
        fn selection_union(
            selections: &[LogicalRowAddressSelection],
        ) -> lance_core::Result<RoaringTreemap> {
            let mut rows = RoaringTreemap::new();
            for selection in selections {
                rows |= selection.to_roaring_treemap()?;
            }
            Ok(rows)
        }

        let invalidated_rows = selection_union(invalidated)?;
        let mut invalidated_map = RowAddrTreeMap::new();
        for (logical_fragment_id, slots) in invalidated_rows.bitmaps() {
            invalidated_map.insert_bitmap(logical_fragment_id, slots.clone());
        }
        Ok(Self::from_resolved_segments(vec![(
            base.to_vec(),
            Arc::new(invalidated_map),
        )]))
    }

    pub(crate) fn with_snapshot_summary(
        mut self,
        covered_physical_fragments: RoaringBitmap,
        fallback_physical_fragments: RoaringBitmap,
        fully_covered_logical_fragments: RoaringBitmap,
        missing_logical_rows: LogicalMissingRows,
    ) -> Self {
        self.covered_physical_fragments = Arc::new(covered_physical_fragments);
        self.fallback_physical_fragments = Arc::new(fallback_physical_fragments);
        self.fully_covered_logical_fragments = Arc::new(fully_covered_logical_fragments);
        self.missing_logical_rows = Arc::new(missing_logical_rows);
        self
    }

    pub(crate) fn with_preopened_index_files(
        mut self,
        preopened_index_files: HashMap<Uuid, OpenedIndexFile>,
    ) -> Self {
        self.preopened_index_files = Arc::new(preopened_index_files);
        self
    }

    pub(crate) fn with_resolved_index_coverages(
        mut self,
        resolved_index_coverages: HashMap<Uuid, LogicalIndexCoverage>,
    ) -> Self {
        self.resolved_index_coverages = Arc::new(resolved_index_coverages);
        self
    }

    pub(crate) fn preopened_index_file(&self, index_id: &Uuid) -> Option<OpenedIndexFile> {
        self.preopened_index_files.get(index_id).cloned()
    }

    pub(crate) fn resolved_index_coverage(&self, index_id: &Uuid) -> Option<LogicalIndexCoverage> {
        self.resolved_index_coverages.get(index_id).cloned()
    }

    pub(crate) fn index_ids(&self) -> &[Uuid] {
        self.index_ids.as_ref()
    }

    /// Return this segment group's effective logical coverage. `None` is the
    /// identity set representing all current logical rows.
    pub(crate) fn effective_rows(&self) -> lance_core::Result<Option<RoaringTreemap>> {
        if self.covers_all_current_slots {
            return Ok(None);
        }
        let mut effective = RoaringTreemap::new();
        for segment in self.segments.iter() {
            let mut rows = RoaringTreemap::new();
            for selection in segment.raw_selections.iter() {
                rows |= selection.to_roaring_treemap()?;
            }
            for (logical_fragment_id, invalidated) in segment.invalidated_rows.iter() {
                let start = u64::from(*logical_fragment_id) << 32;
                match invalidated {
                    lance_select::RowAddrSelection::Full => {
                        rows.remove_range(start..start + (1_u64 << 32));
                    }
                    lance_select::RowAddrSelection::Partial(slots) => {
                        let invalidated = RoaringTreemap::from_iter(
                            slots.iter().map(|slot| start | u64::from(slot)),
                        );
                        rows -= invalidated;
                    }
                }
            }
            effective |= rows;
        }
        Ok(Some(effective))
    }

    pub(crate) fn effective_rows_by_domain(&self) -> lance_core::Result<Option<RowAddrTreeMap>> {
        Ok(self.effective_rows()?.map(|rows| {
            let mut rows_by_domain = RowAddrTreeMap::new();
            for (logical_fragment_id, slots) in rows.bitmaps() {
                rows_by_domain.insert_bitmap(logical_fragment_id, slots.clone());
            }
            rows_by_domain
        }))
    }

    pub(crate) fn contains_effective(&self, row_id: u64) -> lance_core::Result<bool> {
        if self.covers_all_current_slots {
            return Ok(true);
        }
        for segment in self.segments.iter() {
            if segment.contains_effective(row_id)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(crate) fn fallback_physical_fragments(&self) -> &RoaringBitmap {
        self.fallback_physical_fragments.as_ref()
    }

    pub(crate) fn covered_physical_fragments(&self) -> &RoaringBitmap {
        self.covered_physical_fragments.as_ref()
    }

    pub(crate) fn fully_covered_logical_fragments(&self) -> &RoaringBitmap {
        self.fully_covered_logical_fragments.as_ref()
    }

    pub(crate) fn missing_logical_rows(&self) -> &LogicalMissingRows {
        self.missing_logical_rows.as_ref()
    }
}

impl DeepSizeOf for LogicalCoverageGroup {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.segments
            .iter()
            .map(|segment| {
                segment
                    .raw_selections
                    .iter()
                    .map(|selection| selection.deep_size_of_children(context))
                    .sum::<usize>()
                    + segment.invalidated_rows.deep_size_of_children(context)
            })
            .sum::<usize>()
            + self.index_ids.len() * std::mem::size_of::<Uuid>()
            + self.covered_physical_fragments.serialized_size()
            + self.fallback_physical_fragments.serialized_size()
            + self.fully_covered_logical_fragments.serialized_size()
            + self.missing_logical_rows.ranges.len() * std::mem::size_of::<LogicalRowAddressRange>()
            + self
                .missing_logical_rows
                .sparse
                .as_ref()
                .map(|selection| selection.deep_size_of_children(context))
                .unwrap_or_default()
            + self.preopened_index_files.len() * std::mem::size_of::<Uuid>()
            + self
                .preopened_index_files
                .values()
                .map(|opened| opened.deep_size_of_children(context))
                .sum::<usize>()
            + self
                .resolved_index_coverages
                .iter()
                .map(|(index_id, coverage)| {
                    std::mem::size_of_val(index_id) + coverage.deep_size_of_children(context)
                })
                .sum::<usize>()
    }
}

pub fn merged_missing_logical_rows(
    groups: &[LogicalCoverageGroup],
) -> lance_core::Result<LogicalMissingRows> {
    let mut ranges = groups
        .iter()
        .flat_map(|group| group.missing_logical_rows().ranges.iter().copied())
        .collect::<Vec<_>>();
    ranges.sort_unstable();
    let mut merged = Vec::<LogicalRowAddressRange>::with_capacity(ranges.len());
    for range in ranges {
        if let Some(previous) = merged.last_mut()
            && previous.logical_fragment_id == range.logical_fragment_id
            && range.start_slot <= previous.end_slot
        {
            previous.end_slot = previous.end_slot.max(range.end_slot);
        } else {
            merged.push(range);
        }
    }
    let mut sparse = RoaringTreemap::new();
    for group in groups {
        if let Some(group_sparse) = group.missing_logical_rows().sparse.as_ref() {
            sparse |= group_sparse.to_roaring_treemap()?;
        }
    }
    Ok(LogicalMissingRows::new(
        merged,
        (!sparse.is_empty())
            .then(|| LogicalRowAddressSelection::from_bitmap(sparse))
            .transpose()?,
    ))
}

/// Return the exact missing logical rows whose current owner is in `fragments`.
///
/// An omitted fragment scope means the whole dataset. This distinction matters
/// for vector index segment scans: selecting index segments without selecting
/// fragments must not add a flat-search branch for rows outside those segments.
pub async fn merged_missing_logical_rows_in_fragments(
    dataset: &Dataset,
    groups: &[LogicalCoverageGroup],
    fragments: Option<&[Fragment]>,
) -> lance_core::Result<LogicalMissingRows> {
    let missing_rows = merged_missing_logical_rows(groups)?;
    let Some(fragments) = fragments else {
        return Ok(missing_rows);
    };
    if missing_rows.is_empty() {
        return Ok(missing_rows);
    }

    let target_fragments = fragments
        .iter()
        .map(|fragment| u32::try_from(fragment.id))
        .collect::<std::result::Result<RoaringBitmap, _>>()
        .map_err(|_| Error::invalid_input("physical fragment id exceeds u32"))?;
    let all_fragments = dataset
        .manifest
        .fragments
        .iter()
        .map(|fragment| u32::try_from(fragment.id))
        .collect::<std::result::Result<RoaringBitmap, _>>()
        .map_err(|_| Error::invalid_input("physical fragment id exceeds u32"))?;
    if target_fragments == all_fragments {
        return Ok(missing_rows);
    }

    const RESOLVE_BATCH_SIZE: usize = 64 * 1024;
    let mut retained = RoaringTreemap::new();
    for fragment in fragments {
        let fragment_id = u32::try_from(fragment.id)
            .map_err(|_| Error::invalid_input("physical fragment id exceeds u32"))?;
        let physical_rows = u32::try_from(
            fragment
                .physical_rows
                .ok_or_else(|| Error::internal("fragment is missing physical row count"))?,
        )
        .map_err(|_| Error::invalid_input("physical fragment row count exceeds u32"))?;
        for start in (0..physical_rows).step_by(RESOLVE_BATCH_SIZE) {
            let end = physical_rows.min(start.saturating_add(RESOLVE_BATCH_SIZE as u32));
            let physical = (start..end)
                .map(|offset| RowAddress::new_from_parts(fragment_id, offset))
                .collect::<Vec<_>>();
            for logical in dataset
                .resolve_current_physical_row_ids_async(&physical)
                .await?
                .into_iter()
                .flatten()
            {
                if missing_rows.contains(logical)? {
                    retained.insert(logical.raw());
                }
            }
        }
    }
    LogicalMissingRows::from_bitmap(retained)
}

#[derive(Debug)]
pub struct LogicalRowIdRangeExec {
    rows: Arc<LogicalMissingRows>,
    row_count: usize,
    batch_size: usize,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl LogicalRowIdRangeExec {
    pub(crate) fn new(rows: LogicalMissingRows, batch_size: usize) -> Self {
        let schema = Arc::new(arrow_schema::Schema::new(vec![ROW_ID_FIELD.clone()]));
        let row_count = rows
            .ranges
            .iter()
            .map(LogicalRowAddressRange::len)
            .sum::<u64>()
            .saturating_add(
                rows.sparse
                    .as_ref()
                    .map(|selection| selection.cardinality())
                    .unwrap_or_default(),
            )
            .min(usize::MAX as u64) as usize;
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            rows: Arc::new(rows),
            row_count,
            batch_size: batch_size.max(1),
            schema,
            properties,
        }
    }
}

impl DisplayAs for LogicalRowIdRangeExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "LogicalRowIdRanges: ranges={}, rows={}",
            self.rows.ranges.len(),
            self.row_count
        )
    }
}

impl ExecutionPlan for LogicalRowIdRangeExec {
    fn name(&self) -> &str {
        "LogicalRowIdRangeExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(self)
        } else {
            Err(DataFusionError::Internal(
                "LogicalRowIdRangeExec has no children".to_string(),
            ))
        }
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "LogicalRowIdRangeExec has no partition {partition}"
            )));
        }
        let rows = self.rows.clone();
        let batch_size = self.batch_size;
        let schema = self.schema.clone();
        let stream = futures::stream::unfold(
            (rows, 0_usize, None::<u32>, 0_u64),
            move |(rows, mut range_index, mut next_slot, mut sparse_ordinal)| {
                let schema = schema.clone();
                async move {
                    let mut row_ids = Vec::with_capacity(batch_size);
                    while row_ids.len() < batch_size && range_index < rows.ranges.len() {
                        let range = rows.ranges[range_index];
                        let slot = next_slot.unwrap_or(range.start_slot);
                        let remaining = range.end_slot.saturating_sub(slot) as usize;
                        let take = remaining.min(batch_size - row_ids.len());
                        let prefix = u64::from(range.logical_fragment_id) << 32;
                        row_ids.extend(
                            (slot..slot + take as u32).map(|slot| prefix | u64::from(slot)),
                        );
                        if take == remaining {
                            range_index += 1;
                            next_slot = None;
                        } else {
                            next_slot = Some(slot + take as u32);
                        }
                    }
                    if let Some(sparse) = rows.sparse.as_ref() {
                        while row_ids.len() < batch_size && sparse_ordinal < sparse.cardinality() {
                            let address = match sparse.select(sparse_ordinal) {
                                Ok(Some(address)) => Ok(address),
                                Ok(None) => Err(DataFusionError::Internal(
                                    "logical missing selection ended before its cardinality"
                                        .to_string(),
                                )),
                                Err(error) => Err(DataFusionError::External(Box::new(error))),
                            };
                            sparse_ordinal += 1;
                            match address {
                                Ok(address) if !rows.contains_in_ranges(address) => {
                                    row_ids.push(address.raw())
                                }
                                Ok(_) => {}
                                Err(error) => {
                                    return Some((
                                        Err(error),
                                        (rows, range_index, next_slot, sparse_ordinal),
                                    ));
                                }
                            }
                        }
                    }
                    if row_ids.is_empty() {
                        None
                    } else {
                        let batch = RecordBatch::try_new(
                            schema,
                            vec![Arc::new(UInt64Array::from(row_ids))],
                        )
                        .map_err(DataFusionError::from);
                        Some((batch, (rows, range_index, next_slot, sparse_ordinal)))
                    }
                }
            },
        );
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            stream,
        )))
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Statistics> {
        Ok(Statistics {
            num_rows: datafusion::common::stats::Precision::Inexact(self.row_count),
            ..Statistics::new_unknown(self.schema.as_ref())
        })
    }
}

/// Filters a `_rowid` stream by exact v2.3 logical index coverage.
///
/// The indexed branch uses the coverage itself. The fallback branch uses its
/// complement, so the union has neither gaps nor duplicate live rows even when
/// one physical fragment packs several logical domains.
#[derive(Clone)]
pub struct LogicalCoverageFilterExec {
    input: Arc<dyn ExecutionPlan>,
    coverage_groups: Arc<Vec<LogicalCoverageGroup>>,
    include_covered: bool,
    row_id_position: usize,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl std::fmt::Debug for LogicalCoverageFilterExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogicalCoverageFilterExec")
            .field("input", &self.input)
            .field("coverage_group_count", &self.coverage_groups.len())
            .field("include_covered", &self.include_covered)
            .finish()
    }
}

impl LogicalCoverageFilterExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        coverages: impl IntoIterator<Item = LogicalIndexCoverage>,
        include_covered: bool,
    ) -> Result<Self> {
        let selections = coverages
            .into_iter()
            .flat_map(|coverage| coverage.shards)
            .map(|shard| {
                shard.selection.ok_or_else(|| {
                    DataFusionError::Internal(
                        "logical coverage detail was not resolved from its index anchor"
                            .to_string(),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Self::try_new_groups(input, vec![selections], include_covered)
    }

    pub fn try_new_groups(
        input: Arc<dyn ExecutionPlan>,
        coverage_groups: Vec<Vec<LogicalRowAddressSelection>>,
        include_covered: bool,
    ) -> Result<Self> {
        Self::try_new_effective_groups(
            input,
            coverage_groups
                .into_iter()
                .map(|base| {
                    LogicalCoverageGroup::try_from_selections(&base, &[])
                        .map_err(|error| DataFusionError::External(Box::new(error)))
                })
                .collect::<Result<Vec<_>>>()?,
            include_covered,
        )
    }

    pub fn try_new_effective_groups(
        input: Arc<dyn ExecutionPlan>,
        coverage_groups: Vec<LogicalCoverageGroup>,
        include_covered: bool,
    ) -> Result<Self> {
        let row_id_position = input
            .schema()
            .fields()
            .iter()
            .position(|field| field.name() == ROW_ID)
            .ok_or_else(|| {
                DataFusionError::Internal(
                    "LogicalCoverageFilterExec requires a _rowid column".to_string(),
                )
            })?;
        let properties = input.properties().clone();
        Ok(Self {
            input,
            coverage_groups: Arc::new(coverage_groups),
            include_covered,
            row_id_position,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    fn filter_batch(
        batch: &RecordBatch,
        coverage_groups: &[LogicalCoverageGroup],
        include_covered: bool,
        row_id_position: usize,
    ) -> Result<RecordBatch> {
        let row_ids = batch
            .column(row_id_position)
            .as_primitive_opt::<UInt64Type>()
            .ok_or_else(|| {
                DataFusionError::Internal(
                    "LogicalCoverageFilterExec _rowid column is not UInt64".to_string(),
                )
            })?;
        let mut keep = Vec::with_capacity(row_ids.len());
        for row_id in row_ids.iter() {
            let covered = if let Some(row_id) = row_id {
                let address = LogicalRowAddress::try_from(row_id)
                    .map_err(|error| DataFusionError::External(Box::new(error)))?;
                if coverage_groups.is_empty() {
                    false
                } else {
                    let mut covered = true;
                    for group in coverage_groups {
                        if !group
                            .contains_effective(address.raw())
                            .map_err(|error| DataFusionError::External(Box::new(error)))?
                        {
                            covered = false;
                            break;
                        }
                    }
                    covered
                }
            } else {
                false
            };
            keep.push(covered == include_covered);
        }
        arrow_select::filter::filter_record_batch(batch, &BooleanArray::from(keep))
            .map_err(Into::into)
    }
}

impl DisplayAs for LogicalCoverageFilterExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "LogicalCoverageFilterExec: include_covered={}",
            self.include_covered
        )
    }
}

impl ExecutionPlan for LogicalCoverageFilterExec {
    fn name(&self) -> &str {
        "LogicalCoverageFilterExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "LogicalCoverageFilterExec requires one child, got {}",
                children.len()
            )));
        }
        let mut replacement = (*self).clone();
        replacement.input = children[0].clone();
        replacement.properties = children[0].properties().clone();
        Ok(Arc::new(replacement))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let coverage_groups = self.coverage_groups.clone();
        let include_covered = self.include_covered;
        let row_id_position = self.row_id_position;
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let stream = self.input.execute(partition, context)?.map(move |batch| {
            let _timer = baseline.elapsed_compute().timer();
            Self::filter_batch(
                &batch?,
                coverage_groups.as_ref(),
                include_covered,
                row_id_position,
            )
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.input.schema(),
            stream,
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Statistics> {
        Ok(Statistics::new_unknown(self.schema().as_ref()))
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::LowerEqual
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use lance_select::RowAddrSelection;

    use super::*;

    #[test]
    fn effective_rows_by_domain_scales_to_many_segments_and_domains() {
        const DOMAIN_COUNT: u32 = 100_000;
        const SEGMENT_COUNT: u32 = 128;

        let segments = (0..SEGMENT_COUNT)
            .map(|segment| {
                let mut invalidated = RowAddrTreeMap::new();
                let ranges = (segment..DOMAIN_COUNT)
                    .step_by(SEGMENT_COUNT as usize)
                    .map(|logical_fragment_id| {
                        invalidated.insert((u64::from(logical_fragment_id) << 32) | 1);
                        LogicalRowAddressRange::new(logical_fragment_id, 0, 2)
                    })
                    .collect::<Vec<_>>();
                (
                    vec![LogicalRowAddressSelection::from_ranges(ranges).unwrap()],
                    Arc::new(invalidated),
                )
            })
            .collect::<Vec<_>>();
        let group = LogicalCoverageGroup::from_resolved_segments(segments);

        let rows_by_domain = group.effective_rows_by_domain().unwrap().unwrap();
        assert_eq!(rows_by_domain.iter().count(), DOMAIN_COUNT as usize);
        for logical_fragment_id in [0, 1, 127, 128, 50_000, 99_999] {
            let Some(RowAddrSelection::Partial(slots)) = rows_by_domain.get(&logical_fragment_id)
            else {
                panic!("missing logical domain {logical_fragment_id}");
            };
            assert_eq!(slots.iter().collect::<Vec<_>>(), vec![0]);
        }
    }
}
