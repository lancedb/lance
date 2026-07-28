// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Scalar-index probe for the merge-insert target table.

use std::sync::Arc;

use arrow_array::{RecordBatch, UInt64Array, cast::AsArray, types::UInt64Type};
use arrow_schema::{Schema as ArrowSchema, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    catalog::{Session, TableProvider},
    error::DataFusionError,
    execution::TaskContext,
    logical_expr::{Expr, TableType},
    physical_expr::{Distribution, EquivalenceProperties},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
        SendableRecordBatchStream,
        coalesce_partitions::CoalescePartitionsExec,
        execution_plan::{Boundedness, EmissionType},
        stream::RecordBatchStreamAdapter,
        union::UnionExec,
    },
};
use futures::TryStreamExt;
use lance_arrow::SchemaExt;
use lance_core::{
    ROW_ADDR_FIELD, ROW_ID_FIELD,
    datatypes::{BlobHandling, OnMissing},
};
use lance_table::format::Fragment;
use roaring::RoaringTreemap;

use crate::Dataset;
use crate::io::exec::{
    filtered_read::{FilteredReadExec, FilteredReadOptions},
    project,
    scalar_index::{INDEX_LOOKUP_SCHEMA, IndexLookup, MapIndexExec},
};

/// Drops row addresses an earlier batch already emitted, so a probe's candidate
/// stream is a set.
///
/// [`MapIndexExec`] evaluates one query per input batch and emits that batch's
/// matches, so an over-matching probe can reach the same target row from two
/// different batches. Reading a target row twice would give one source row two
/// candidate matches in the join, and the merge rejects that as ambiguous — so
/// the duplicates have to go before the take.
#[derive(Debug)]
struct DistinctRowAddrsExec {
    input: Arc<dyn ExecutionPlan>,
    properties: Arc<PlanProperties>,
}

impl DistinctRowAddrsExec {
    fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(INDEX_LOOKUP_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self { input, properties }
    }

    fn retain_unseen(seen: &mut RoaringTreemap, batch: &RecordBatch) -> RecordBatch {
        let addrs = batch.column(0).as_primitive::<UInt64Type>();
        let unseen: UInt64Array = addrs
            .values()
            .iter()
            .copied()
            .filter(|addr| seen.insert(*addr))
            .collect();
        RecordBatch::try_new(INDEX_LOOKUP_SCHEMA.clone(), vec![Arc::new(unseen)])
            .expect("a UInt64 column always matches INDEX_LOOKUP_SCHEMA")
    }
}

impl DisplayAs for DistinctRowAddrsExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "DistinctRowAddrs")
            }
            DisplayFormatType::TreeRender => write!(f, "DistinctRowAddrs"),
        }
    }
}

impl ExecutionPlan for DistinctRowAddrsExec {
    fn name(&self) -> &str {
        "DistinctRowAddrsExec"
    }

    fn schema(&self) -> SchemaRef {
        INDEX_LOOKUP_SCHEMA.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        let [input] = <[_; 1]>::try_from(children).map_err(|children| {
            DataFusionError::Internal(format!(
                "DistinctRowAddrsExec requires exactly one child, got {}",
                children.len()
            ))
        })?;
        Ok(Arc::new(Self::new(input)))
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // De-duplicating across batches only works if every batch arrives here.
        vec![Distribution::SinglePartition]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion::common::Result<SendableRecordBatchStream> {
        let mut seen = RoaringTreemap::new();
        let stream = self
            .input
            .execute(partition, context)?
            .map_ok(move |batch| Self::retain_unseen(&mut seen, &batch));
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            INDEX_LOOKUP_SCHEMA.clone(),
            stream,
        )))
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
}

/// A [`TableProvider`] for the merge-insert target that reaches candidate rows
/// through scalar indices instead of scanning the whole table.
///
/// It reports the same schema as
/// [`LanceTableProvider`](crate::datafusion::LanceTableProvider) built with row
/// id and row address, so the rest of the merge-insert plan — the join, the
/// action expression, the write sink — is identical either way. Only the target
/// scan differs:
///
/// ```text
/// Union
/// ├─ FilteredReadExec              take the projected columns of probed rows
/// │  └─ DistinctRowAddrsExec       one candidate row address at most once
/// │     └─ MapIndexExec            AND of one IsIn probe per indexed key
/// │        └─ CoalescePartitionsExec
/// │           └─ <source scan #2>  key columns only
/// └─ FilteredReadExec              fragments no index covers
/// ```
///
/// The probe is allowed to over-match and must not under-match, because the
/// downstream join is what applies the real key predicate. Three things follow
/// from that:
///
/// - Only the indexed subset of the merge keys needs to be probed. A composite
///   key with one indexed column still prunes by that column, and per-column
///   `IsIn` lists do not correlate values across the tuple anyway.
/// - Fragments that any chosen index does not cover would be invisible to the
///   probe, so they are scanned and unioned in.
/// - Over-matching is only harmless while each candidate row reaches the join
///   once. Probes are evaluated per source batch, so two batches can name the
///   same target row and the candidates are de-duplicated before the take.
///
/// The source is scanned a second time here to collect key values. Callers must
/// only build this over a re-scannable source.
#[derive(Debug)]
pub(super) struct IndexProbeTarget {
    dataset: Arc<Dataset>,
    source: Arc<dyn TableProvider>,
    lookups: Vec<IndexLookup>,
    unindexed_fragments: Arc<Vec<Fragment>>,
    blob_handling: BlobHandling,
    full_schema: SchemaRef,
}

impl IndexProbeTarget {
    pub(super) fn try_new(
        dataset: Arc<Dataset>,
        source: Arc<dyn TableProvider>,
        lookups: Vec<IndexLookup>,
        unindexed_fragments: Vec<Fragment>,
        blob_handling: BlobHandling,
    ) -> crate::Result<Self> {
        if lookups.is_empty() {
            return Err(crate::Error::internal(
                "IndexProbeTarget requires at least one index lookup",
            ));
        }
        let full_schema = ArrowSchema::from(dataset.schema())
            .try_with_column(ROW_ID_FIELD.clone())?
            .try_with_column(ROW_ADDR_FIELD.clone())?;
        Ok(Self {
            dataset,
            source,
            lookups,
            unindexed_fragments: Arc::new(unindexed_fragments),
            blob_handling,
            full_schema: Arc::new(full_schema),
        })
    }

    /// The columns DataFusion asked for, in the order it expects them back.
    fn projected_schema(
        &self,
        projection: Option<&Vec<usize>>,
    ) -> datafusion::common::Result<ArrowSchema> {
        match projection {
            Some(indices) => Ok(self.full_schema.project(indices)?),
            None => Ok(self.full_schema.as_ref().clone()),
        }
    }

    /// Scan the source's key columns and turn them into candidate row addresses.
    async fn probe(
        &self,
        state: &dyn Session,
        columns: &lance_core::datatypes::Projection,
        output_schema: &ArrowSchema,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        let source_schema = self.source.schema();
        let mut key_indices = self
            .lookups
            .iter()
            .map(|lookup| source_schema.index_of(&lookup.column))
            .collect::<Result<Vec<_>, _>>()?;
        key_indices.sort_unstable();
        key_indices.dedup();
        let keys = self
            .source
            .scan(state, Some(&key_indices), &[], None)
            .await?;

        // `MapIndexExec` matches its input columns to `lookups` positionally and
        // reads a single partition, so the keys are reordered and coalesced first.
        let lookup_schema = ArrowSchema::new(
            self.lookups
                .iter()
                .map(|lookup| source_schema.field_with_name(&lookup.column).cloned())
                .collect::<Result<Vec<_>, _>>()?,
        );
        let keys = Arc::new(project(keys, &lookup_schema)?);
        let keys = Arc::new(CoalescePartitionsExec::new(keys));

        let probe = Arc::new(MapIndexExec::new_multi(
            self.dataset.clone(),
            self.lookups.clone(),
            keys,
        ));
        let probe = Arc::new(DistinctRowAddrsExec::new(probe));
        let take = Arc::new(FilteredReadExec::try_new(
            self.dataset.clone(),
            FilteredReadOptions::new(columns.clone()),
            Some(probe),
        )?);
        Ok(Arc::new(project(take, output_schema)?))
    }

    /// Scan the fragments the probe cannot see.
    fn scan_unindexed(
        &self,
        columns: &lance_core::datatypes::Projection,
        output_schema: &ArrowSchema,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        let options = FilteredReadOptions::new(columns.clone())
            .with_fragments(self.unindexed_fragments.clone());
        let scan = Arc::new(FilteredReadExec::try_new(
            self.dataset.clone(),
            options,
            None,
        )?);
        Ok(Arc::new(project(scan, output_schema)?))
    }
}

#[async_trait]
impl TableProvider for IndexProbeTarget {
    fn schema(&self) -> SchemaRef {
        self.full_schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        // Neither is reachable from the merge-insert plan, and both would be
        // wrong to drop: `supports_filters_pushdown` rejects every filter, and
        // nothing puts a limit above the target scan.
        if !filters.is_empty() {
            return Err(DataFusionError::Internal(format!(
                "IndexProbeTarget cannot apply pushed-down filters, got {} of them",
                filters.len()
            )));
        }
        if let Some(limit) = limit {
            return Err(DataFusionError::Internal(format!(
                "IndexProbeTarget cannot apply a pushed-down limit of {limit}"
            )));
        }

        let output_schema = self.projected_schema(projection)?;
        let columns = self
            .dataset
            .empty_projection()
            .with_blob_handling(self.blob_handling.clone())
            .union_columns(
                output_schema.fields().iter().map(|field| field.name()),
                OnMissing::Error,
            )?;

        let probe = self.probe(state, &columns, &output_schema).await?;
        if self.unindexed_fragments.is_empty() {
            return Ok(probe);
        }
        let unindexed = self.scan_unindexed(&columns, &output_schema)?;
        Ok(UnionExec::try_new(vec![probe, unindexed])?)
    }
}
