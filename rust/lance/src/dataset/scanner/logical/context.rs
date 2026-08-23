// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stage 2: the bridge between async planning and synchronous rules.
//!
//! An `OptimizerRule` is synchronous, so a rule cannot load index metadata. This struct is how a
//! rule gets it anyway: everything is fetched once, up front, while we are still in an async
//! context, and the rules hold an `Arc` to the result.
//!
//! The prefetch set is deliberately small — **all index metadata, plus the manifest's fragment
//! metadata**. The latter is already in memory (`Manifest::fragments`, carrying `physical_rows`,
//! `deletion_file`, and `overlays`), which is what makes so many of the imperative path's `await`
//! points collapse into the synchronous derivations below.
//!
//! [`ScalarIndexInfo`](crate::index::ScalarIndexInfo) is the existing instance of this pattern —
//! its doc comment describes the same contract for scalar indices. This generalizes it to vector
//! indices rather than inventing a second mechanism.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::LogicalPlan;
use lance_core::utils::address::RowAddress;
use lance_index::metrics::NoOpMetricsCollector;
use lance_linalg::distance::DistanceType;
use lance_select::mask::RowAddrTreeMap;
use lance_table::format::{Fragment, IndexMetadata};
use roaring::RoaringBitmap;

use lance_index::scalar::expression::{IndexInformationProvider, ScalarIndexExpr};

use super::scan_index::with_lance_source;
use super::{TakeSettings, VectorSearchNode};
use crate::Result;
use crate::dataset::overlay::{collect_overlay_stale_rows_for_segment, overlaid_fragments};
use crate::dataset::rowids::{live_row_addrs_to_row_ids, translate_addr_treemap_to_row_ids};
use crate::dataset::scanner::TakeOperation;
use crate::dataset::{Dataset, row_offsets_to_row_addresses};
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt, ScalarIndexInfo};

/// What a data overlay did to an index's entries.
///
/// A data overlay committed after an index was built, touching a field that index covers, makes
/// the index's entries for the overlaid rows describe values that no longer exist. This is the
/// row-level half of index coverage, and it is prefetched for the same reason fragment coverage is:
/// deciding it needs the index metadata and, under stable row ids, a row-id translation.
#[derive(Debug, Clone, Default)]
pub enum OverlayStaleness {
    /// No overlay touches a field this index covers.
    #[default]
    None,
    /// Exactly these rows' entries are stale; the rest of the index is current.
    Rows(Arc<RowAddrTreeMap>),
}

/// The rows whose entries in `segments` an overlay has invalidated, in the domain index results
/// use.
///
/// Index results are in the row-id domain, and a physical row address equals its row id only when
/// the dataset does not use stable row ids — hence the translation, which is the one piece of this
/// that needs I/O and the reason it lives in stage 2.
pub async fn overlay_staleness(
    dataset: &Dataset,
    segments: &[IndexMetadata],
    fragments: &[Fragment],
) -> Result<OverlayStaleness> {
    // Overlays are rare; this is the check that keeps the whole mechanism off the common path.
    let overlaid = overlaid_fragments(fragments);
    if overlaid.is_empty() {
        return Ok(OverlayStaleness::None);
    }

    let mut stale: HashMap<u32, RoaringBitmap> = HashMap::new();
    for segment in segments {
        collect_overlay_stale_rows_for_segment(segment, &overlaid, &mut stale, dataset.schema())?;
    }
    if stale.is_empty() {
        return Ok(OverlayStaleness::None);
    }

    let mut rows = RowAddrTreeMap::new();
    for (fragment, offsets) in stale {
        rows.insert_bitmap(fragment, offsets);
    }
    if dataset.manifest.uses_stable_row_ids() {
        rows = translate_addr_treemap_to_row_ids(dataset, &rows).await?;
    }
    Ok(OverlayStaleness::Rows(Arc::new(rows)))
}

/// Everything known about the vector index covering one column.
#[derive(Debug, Clone)]
pub struct VectorIndexInfo {
    /// The index's delta segments; a search fans out over all of them.
    pub segments: Vec<IndexMetadata>,
    /// The metric the index was built with. Resolving this is the one prefetch that may have to
    /// open the index itself: indices written before `details` carried the metric have nowhere
    /// else to record it, and "user asked for a different metric, so fall back to brute force"
    /// is a planning decision that cannot wait until execution.
    pub metric: DistanceType,
    /// Row-level coverage: which of this index's entries a data overlay has invalidated.
    pub staleness: OverlayStaleness,
}

impl VectorIndexInfo {
    /// Fragments covered by at least one of this index's segments.
    ///
    /// A segment with no fragment bitmap predates the field, so nothing can be assumed about its
    /// coverage; treating it as covering nothing would wrongly route indexed rows to a flat scan,
    /// so callers get `None` and should fall back.
    fn indexed_fragments(&self) -> Option<RoaringBitmap> {
        let mut covered = RoaringBitmap::new();
        for segment in &self.segments {
            covered |= segment.fragment_bitmap.as_ref()?;
        }
        Some(covered)
    }
}

/// Index and fragment state, captured once per planning invocation.
#[derive(Debug)]
pub struct ScanPlanningContext {
    /// From the manifest, so this costs nothing — but holding it here is what lets the
    /// derivations below be honest about doing no I/O.
    fragments: Arc<Vec<Fragment>>,
    vector: HashMap<String, VectorIndexInfo>,
    /// Whether `Scanner::with_index_segments` pinned which segments a search may use.
    ///
    /// It makes the caller's chosen segments the whole answer: rows they do not cover are out of
    /// scope rather than a coverage gap, so no brute-force branch fills it. Mirrors
    /// `Scanner::knn_combined`.
    segments_pinned: bool,
    /// `Scanner::fast_search`: skip rows no index covers. A rule needs it to know whether to
    /// build the flat branch at all, and it is not derivable from the plan.
    fast_search: bool,
    /// Read settings for any take a rule introduces, so a rewrite cannot lose the scanner's
    /// fragment restriction.
    take_settings: TakeSettings,
    /// What the filter parser needs to turn a predicate into a scalar index query. Already the
    /// canonical instance of this whole pattern — see the module docs.
    scalar_indices: Arc<ScalarIndexInfo>,
    /// Row-level coverage for every index in the dataset, keyed by index name.
    ///
    /// Keyed by name rather than by column because that is how a scalar index query names what it
    /// consulted. Empty unless some fragment carries an overlay, which is what keeps this off the
    /// common path.
    index_staleness: HashMap<String, OverlayStaleness>,
    /// Row selections for take-shaped predicates whose resolution needed I/O, keyed by the
    /// predicate's values rather than its expression: the rules see it after `PushDownFilter` has
    /// moved it, and the values are what survive that unchanged.
    take_rows: HashMap<String, Arc<RowAddrTreeMap>>,
}

impl ScanPlanningContext {
    /// Walk the unoptimized plan for what it will need, then fetch all of it.
    ///
    /// This is the only `async fn` in stages 2 through 4 that is allowed to do I/O.
    ///
    /// Everything comes from the plan — the dataset and the scan-wide settings from its leaf, the
    /// searched columns from its extension nodes. Nothing reads the `Scanner`, so any plan built
    /// over a [`LanceScanSource`](super::source::LanceScanSource) can be planned this way, not just
    /// one the scanner's builder produced.
    pub async fn collect(plan: &LogicalPlan) -> Result<Self> {
        let mut leaf = None;
        let mut searches: Vec<(String, Option<DistanceType>)> = Vec::new();
        let mut takes: Vec<TakeOperation> = Vec::new();
        plan.apply(|node| {
            takes.extend(take_operations(node));
            if let LogicalPlan::Extension(extension) = node
                && let Some(search) = extension.node.as_any().downcast_ref::<VectorSearchNode>()
            {
                let requested = search
                    .distance_type_requested()
                    .then(|| search.distance_type());
                searches.push((search.query().column.clone(), requested));
            }
            if leaf.is_none() {
                leaf = with_lance_source(node, |source| {
                    (source.dataset().clone(), source.options().clone())
                });
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        let Some((dataset, options)) = leaf else {
            return Err(crate::Error::internal(
                "a Lance scan plan reached stage 2 without a Lance scan leaf".to_string(),
            ));
        };

        let fragments = options
            .fragments
            .clone()
            .unwrap_or_else(|| dataset.fragments().clone());

        let mut vector = HashMap::with_capacity(searches.len());
        if !searches.is_empty() {
            let indices = dataset.load_indices().await?;
            for (column, requested_metric) in searches {
                if vector.contains_key(&column) {
                    continue;
                }
                if let Some(info) = load_vector_index(
                    &dataset,
                    options.index_segments.as_deref(),
                    requested_metric,
                    &indices,
                    &column,
                    &fragments,
                )
                .await?
                {
                    vector.insert(column, info);
                }
            }
        }

        let scalar_indices = Arc::new(dataset.scalar_index_info().await?);
        let index_staleness = prefetch_index_staleness(&dataset, &fragments).await?;
        let take_rows = resolve_takes(&dataset, takes).await?;

        Ok(Self {
            fragments,
            vector,
            fast_search: options.fast_search,
            segments_pinned: options.index_segments.is_some(),
            take_settings: TakeSettings {
                fragments: options.fragments.clone(),
                batch_size: options.batch_size.map(|size| size as u32),
            },
            scalar_indices,
            index_staleness,
            take_rows,
        })
    }

    /// The rows a take-shaped predicate selects, if resolving it needed I/O and stage 2 did it.
    pub fn take_rows(&self, take: &TakeOperation) -> Option<&Arc<RowAddrTreeMap>> {
        self.take_rows.get(&take_key(take))
    }

    pub fn scalar_indices(&self) -> &ScalarIndexInfo {
        &self.scalar_indices
    }

    /// Whether any fragment this scan will read carries a data overlay.
    ///
    /// The one check that keeps overlay handling off the common path, and the reason
    /// [`Self::index_query_staleness`] can answer `None` without consulting anything.
    pub fn has_overlays(&self) -> bool {
        !self.index_staleness.is_empty()
    }

    /// The rows a scalar index query's results cannot be trusted for.
    ///
    /// A query may consult several indices, so this is the union of what each one lost. One
    /// untrustworthy index poisons the whole query, since the results are combined before anything
    /// can tell them apart.
    pub fn index_query_staleness(&self, query: &ScalarIndexExpr) -> OverlayStaleness {
        let mut stale = RowAddrTreeMap::new();
        let mut found = false;
        let mut queue = vec![query];
        while let Some(expr) = queue.pop() {
            match expr {
                ScalarIndexExpr::Not(inner) => queue.push(inner),
                ScalarIndexExpr::And(lhs, rhs) | ScalarIndexExpr::Or(lhs, rhs) => {
                    queue.push(lhs);
                    queue.push(rhs);
                }
                ScalarIndexExpr::Query(search) => {
                    match self.index_staleness.get(&search.index_name) {
                        Some(OverlayStaleness::Rows(rows)) => {
                            stale |= rows.as_ref().clone();
                            found = true;
                        }
                        Some(OverlayStaleness::None) | None => {}
                    }
                }
            }
        }
        match found {
            true => OverlayStaleness::Rows(Arc::new(stale)),
            false => OverlayStaleness::None,
        }
    }

    /// The fragments these index segments can return a row from, limited to what this scan reads.
    ///
    /// A segment with no fragment bitmap could have indexed anything, so the answer widens to
    /// every fragment. Mirrors `Scanner::get_indexed_frags`.
    pub fn segment_coverage(&self, segments: &[IndexMetadata]) -> RoaringBitmap {
        let scanned = RoaringBitmap::from_iter(self.fragments.iter().map(|f| f.id as u32));
        let mut covered = RoaringBitmap::new();
        for segment in segments {
            let Some(bitmap) = segment.fragment_bitmap.as_ref() else {
                return scanned;
            };
            covered |= bitmap;
        }
        covered & scanned
    }

    /// The fragments a scalar index query can answer for, or `None` if some index in it cannot
    /// say which fragments it covers.
    ///
    /// Mirrors `Scanner::fragments_covered_by_index_query`, intersecting for `Or` as well as
    /// `And`: an `Or` is only answerable where *both* sides are, since a fragment one side cannot
    /// speak for could still contain a matching row.
    pub fn index_query_coverage(&self, query: &ScalarIndexExpr) -> Option<RoaringBitmap> {
        match query {
            ScalarIndexExpr::Not(inner) => self.index_query_coverage(inner),
            ScalarIndexExpr::And(lhs, rhs) | ScalarIndexExpr::Or(lhs, rhs) => {
                Some(self.index_query_coverage(lhs)? & self.index_query_coverage(rhs)?)
            }
            ScalarIndexExpr::Query(search) => self
                .scalar_indices
                .fragment_bitmap(&search.column, &search.index_name),
        }
    }

    pub fn vector_index(&self, column: &str) -> Option<&VectorIndexInfo> {
        self.vector.get(column)
    }

    pub fn fast_search(&self) -> bool {
        self.fast_search
    }

    /// Whether a vector index's coverage gap should be filled by a brute-force branch.
    ///
    /// Pinned segments are the whole answer, so rows they miss are out of scope — unless the
    /// caller also named fragments, which asks for those fragments' rows however they have to be
    /// found. Mirrors `Scanner::knn_combined`.
    pub fn fills_vector_coverage_gaps(&self) -> bool {
        !self.segments_pinned || self.take_settings.fragments.is_some()
    }

    pub fn take_settings(&self) -> &TakeSettings {
        &self.take_settings
    }

    /// The subset of `segments` that could contribute a row this scan is allowed to return.
    ///
    /// A segment with no fragment bitmap predates the field, so its coverage is unknown and it has
    /// to be kept.
    pub fn reachable_segments(&self, segments: &[IndexMetadata]) -> Vec<IndexMetadata> {
        let target = RoaringBitmap::from_iter(self.fragments.iter().map(|f| f.id as u32));
        segments
            .iter()
            .filter(|segment| {
                segment
                    .fragment_bitmap
                    .as_ref()
                    .is_none_or(|covered| !covered.is_disjoint(&target))
            })
            .cloned()
            .collect()
    }

    /// Fragments the given vector index does not cover — synchronous, because it is just the
    /// index's fragment bitmap subtracted from the manifest's fragment list.
    ///
    /// This is the derivation that decides between a single-branch ANN plan and the two-branch
    /// combined plan, and in the imperative path it is an `await`.
    pub fn unindexed_fragments(&self, column: &str) -> Option<Vec<Fragment>> {
        self.partition_fragments(column, false)
    }

    /// The complement of [`Self::unindexed_fragments`].
    pub fn indexed_fragments(&self, column: &str) -> Option<Vec<Fragment>> {
        self.partition_fragments(column, true)
    }

    fn partition_fragments(&self, column: &str, covered_side: bool) -> Option<Vec<Fragment>> {
        let covered = self.vector_index(column)?.indexed_fragments()?;
        Some(
            self.fragments
                .iter()
                .filter(|fragment| covered.contains(fragment.id as u32) == covered_side)
                .cloned()
                .collect(),
        )
    }
}

/// Row-level coverage for every index in the dataset, keyed by index name.
///
/// Which indices a scan consults is not known until filter pushdown has settled, which is after
/// every stage that is allowed to do I/O. Since the index list is short and this whole map is
/// skipped unless an overlay exists, computing all of them up front is cheaper than an async hop
/// later — and it is what lets the coverage split be one synchronous rule.
async fn prefetch_index_staleness(
    dataset: &Arc<Dataset>,
    fragments: &[Fragment],
) -> Result<HashMap<String, OverlayStaleness>> {
    if overlaid_fragments(fragments).is_empty() {
        return Ok(HashMap::new());
    }

    let mut segments: HashMap<String, Vec<IndexMetadata>> = HashMap::new();
    for index in dataset.load_indices().await?.iter() {
        segments
            .entry(index.name.clone())
            .or_default()
            .push(index.clone());
    }

    let mut staleness = HashMap::with_capacity(segments.len());
    for (name, segments) in segments {
        staleness.insert(
            name,
            overlay_staleness(dataset, &segments, fragments).await?,
        );
    }
    Ok(staleness)
}

async fn load_vector_index(
    dataset: &Dataset,
    index_segments: Option<&Vec<uuid::Uuid>>,
    requested_metric: Option<DistanceType>,
    indices: &[IndexMetadata],
    column: &str,
    fragments: &[Fragment],
) -> Result<Option<VectorIndexInfo>> {
    let Ok(field_id) = dataset.schema().field_id(column) else {
        return Ok(None);
    };

    let segments = match index_segments {
        Some(requested) => select_requested_segments(requested, indices, column, field_id)?,
        None => {
            let Some(index) = indices
                .iter()
                .find(|index| index.fields.contains(&field_id))
            else {
                return Ok(None);
            };
            dataset.load_indices_by_name(&index.name).await?.to_vec()
        }
    };

    let metric = match crate::index::vector::details::metric_type_from_index_metadata(&segments[0])
    {
        Some(metric) => metric,
        None => dataset
            .open_vector_index(column, &segments[0].uuid, &NoOpMetricsCollector)
            .await?
            .metric_type(),
    };
    if let Some(requested) = requested_metric
        && index_segments.is_some()
        && requested != metric
    {
        // Without an explicit segment list a metric mismatch falls back to brute force. With one
        // it cannot: the caller named these segments, so silently not using them would answer a
        // different question than the one asked.
        return Err(crate::Error::invalid_input(format!(
            "with_index_segments requested metric {requested:?} but the selected index segments use {metric:?}",
        )));
    }

    let staleness = overlay_staleness(dataset, &segments, fragments).await?;
    Ok(Some(VectorIndexInfo {
        segments,
        metric,
        staleness,
    }))
}

/// The segments `with_index_segments` named, validated against the index metadata.
///
/// Mirrors `Scanner::vector_search`: every named segment must exist, cover the queried column, and
/// belong to the same logical index as the others. Narrowing to the fragments this scan reads
/// happens later, in `ScanPlanningContext::reachable_segments`, which every search goes through.
fn select_requested_segments(
    requested: &[uuid::Uuid],
    indices: &[IndexMetadata],
    column: &str,
    field_id: i32,
) -> Result<Vec<IndexMetadata>> {
    let wanted = requested.iter().copied().collect::<HashSet<_>>();
    let selected = indices
        .iter()
        .filter(|index| wanted.contains(&index.uuid))
        .cloned()
        .collect::<Vec<_>>();

    if selected.len() != wanted.len() {
        let found = selected
            .iter()
            .map(|index| index.uuid)
            .collect::<HashSet<_>>();
        let missing = wanted
            .difference(&found)
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        return Err(crate::Error::invalid_input(format!(
            "with_index_segments referenced unknown index segments: {missing:?}",
        )));
    }
    if selected
        .iter()
        .any(|index| !index.fields.contains(&field_id))
    {
        return Err(crate::Error::invalid_input(format!(
            "with_index_segments contained a segment that does not belong to vector column '{column}'",
        )));
    }
    let name = &selected[0].name;
    if selected.iter().any(|index| &index.name != name) {
        return Err(crate::Error::invalid_input(
            "with_index_segments must reference segments from a single logical index".to_string(),
        ));
    }
    Ok(selected)
}

/// The take-shaped predicates a plan node carries, if any.
///
/// Both positions are checked because stage 2 runs before `PushDownFilter`: a predicate written by
/// the builder is a `Filter`, while one the DataFrame API pushed is already on the scan.
fn take_operations(node: &LogicalPlan) -> Vec<TakeOperation> {
    let predicates: Vec<_> = match node {
        LogicalPlan::Filter(filter) => vec![filter.predicate.clone()],
        LogicalPlan::TableScan(scan) => scan.filters.clone(),
        _ => vec![],
    };
    predicates
        .iter()
        .filter_map(|predicate| Some(TakeOperation::try_from_expr(predicate)?.0))
        .collect()
}

/// Identifies a take by what it selects, so the same take found at two points in planning agrees.
pub fn take_key(take: &TakeOperation) -> String {
    format!("{take:?}")
}

/// Turn `_rowaddr` and `_rowoffset` takes into the row ids a read can be restricted to.
///
/// `TakeOperation::RowIds` is absent because it needs no translation — the rule handles it inline.
async fn resolve_takes(
    dataset: &Arc<Dataset>,
    takes: Vec<TakeOperation>,
) -> Result<HashMap<String, Arc<RowAddrTreeMap>>> {
    let mut resolved = HashMap::new();
    for take in takes {
        let key = take_key(&take);
        if resolved.contains_key(&key) {
            continue;
        }
        let addrs = match &take {
            TakeOperation::RowIds(_) => continue,
            TakeOperation::RowAddrs(addrs) => addrs.clone(),
            TakeOperation::RowOffsets(offsets) => {
                let mut addrs =
                    row_offsets_to_row_addresses(&dataset.get_fragments(), offsets).await?;
                // An offset can only name a live row, so a tombstone means the offset ran past
                // the end of the dataset. Mirrors `Scanner::take_source`.
                addrs.retain(|addr| *addr != RowAddress::TOMBSTONE_ROW);
                addrs
            }
        };
        let row_ids = live_row_addrs_to_row_ids(dataset, addrs.into_iter().map(Some)).await?;
        resolved.insert(
            key,
            Arc::new(RowAddrTreeMap::from_iter(row_ids.into_iter().flatten())),
        );
    }
    Ok(resolved)
}
