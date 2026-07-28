// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Scalar-index probe for the merge-insert target table.

use std::sync::Arc;

use arrow_schema::{Schema as ArrowSchema, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    catalog::{Session, TableProvider},
    error::DataFusionError,
    logical_expr::{Expr, TableType},
    physical_plan::{ExecutionPlan, coalesce_partitions::CoalescePartitionsExec, union::UnionExec},
};
use lance_arrow::SchemaExt;
use lance_core::{
    ROW_ADDR_FIELD, ROW_ID_FIELD,
    datatypes::{BlobHandling, OnMissing},
};
use lance_table::format::Fragment;

use crate::Dataset;
use crate::io::exec::{
    filtered_read::{FilteredReadExec, FilteredReadOptions},
    project,
    scalar_index::{IndexLookup, MapIndexExec},
};

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
/// │  └─ CoalescePartitionsExec
/// │     └─ MapIndexExec            AND of one IsIn probe per indexed key
/// │        └─ <source scan #2>     key columns only
/// └─ FilteredReadExec              fragments no index covers
/// ```
///
/// The probe is allowed to over-match and must not under-match, because the
/// downstream join is what applies the real key predicate. Two things follow
/// from that:
///
/// - Only the indexed subset of the merge keys needs to be probed. A composite
///   key with one indexed column still prunes by that column, and per-column
///   `IsIn` lists do not correlate values across the tuple anyway.
/// - Fragments that any chosen index does not cover would be invisible to the
///   probe, so they are scanned and unioned in.
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
