// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Resolution of data overlay files on read.
//!
//! An overlay supplies new values for a subset of `(physical offset, field)`
//! cells. To resolve a field's values for a set of physical row offsets, the
//! overlays that cover that field are walked **newest to oldest**: the first
//! overlay that covers an offset wins, and its value is taken at the offset's
//! **rank** (the 0-based count of set bits below it) in the field's coverage
//! bitmap. An offset that no overlay covers falls through to the base value.
//!
//! The offsets are supplied explicitly (one per base row), so a single code path
//! serves both the scan (a contiguous physical range) and `take` (arbitrary
//! physical offsets) read paths.
//!
//! Deletions take precedence over overlays, but that is handled downstream: the
//! merge runs on physical rows *before* the deletion filter, so an overlay value
//! for a deleted offset is computed and then dropped with the row — making it
//! inert, exactly as the specification requires, with no special handling here.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_select::interleave::interleave;
use futures::StreamExt;
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;

use lance_table::format::overlay::DataOverlayFile;
use lance_table::utils::stream::ReadBatchFut;

use crate::dataset::fragment::{FileFragment, FragReadConfig, GenericFileReader};

/// Order a fragment's overlays from newest to oldest for read resolution.
///
/// Precedence is by `committed_version` (higher is newer); ties are broken by
/// position in the fragment's `overlays` list, where a later entry is newer.
/// Returns indices into `overlays`.
pub fn overlay_indices_newest_first(overlays: &[DataOverlayFile]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..overlays.len()).collect();
    indices.sort_by(|&a, &b| {
        overlays[b]
            .committed_version
            .cmp(&overlays[a].committed_version)
            .then(b.cmp(&a))
    });
    indices
}

/// How a batch of physical row offsets routes onto a field's overlays.
///
/// Produced by [`route_overlays`] from the coverage bitmaps alone — before any
/// value column is read — so the caller can fetch only the ranks it actually
/// needs (see [`OverlayRouting::needed_ranks`]) instead of the whole column, and
/// then assemble the merged column with [`assemble_overlay_column`].
pub struct OverlayRouting {
    /// `interleave` source/position pairs, one per output row. Source `0` is the
    /// base column (position = the row's index); source `k + 1` is overlay `k`'s
    /// fetched values (position = the row's index within `needed_ranks[k]`).
    indices: Vec<(usize, usize)>,
    /// `needed_ranks[k]` is the sorted, deduplicated set of coverage ranks that
    /// overlay `k` must supply for this batch — the indices to fetch from its
    /// value column.
    needed_ranks: Vec<Vec<u32>>,
    /// Whether any row routes to an overlay at all (false ⇒ pure fall-through).
    any_overlay: bool,
}

impl OverlayRouting {
    /// The ranks each overlay (newest-first) must fetch from its value column.
    pub fn needed_ranks(&self) -> &[Vec<u32>] {
        &self.needed_ranks
    }

    /// True when no row is covered by any overlay, so the base column is the
    /// answer unchanged and no value-column reads are needed.
    pub fn all_fall_through(&self) -> bool {
        !self.any_overlay
    }
}

/// Decide, for each physical offset in `offsets`, which source supplies its
/// value: the newest overlay whose coverage contains it (taken at the offset's
/// 0-based rank in that coverage), or the base column if none covers it.
///
/// Reads only the coverage bitmaps (newest-first), so it can run before the
/// value columns are fetched and tells the caller exactly which ranks to fetch.
///
/// A scan reads a contiguous physical range, so when `offsets` is contiguous
/// ascending we take a bitmap-major fast path that visits only each coverage's
/// in-range bits — `O(covered + K)` — instead of probing every offset against
/// every coverage. `take` supplies arbitrary offsets and uses the general path.
pub fn route_overlays(
    offsets: &[u32],
    coverages_newest_first: &[&RoaringBitmap],
) -> OverlayRouting {
    match contiguous_base(offsets) {
        Some(base) => route_contiguous(base, offsets.len(), coverages_newest_first),
        None => route_arbitrary(offsets, coverages_newest_first),
    }
}

/// The starting offset if `offsets` is a contiguous ascending run
/// `[base, base + 1, ...]`, else `None` (including when empty).
fn contiguous_base(offsets: &[u32]) -> Option<u32> {
    let base = *offsets.first()?;
    offsets
        .iter()
        .enumerate()
        .all(|(i, &offset)| offset as u64 == base as u64 + i as u64)
        .then_some(base)
}

/// Fast path for a contiguous batch: offset `o` is output row `o - base`, so a
/// coverage's bits route to rows directly without per-offset probing.
///
/// For each coverage we intersect with the batch's offset range, which is a
/// container-level operation that drops a non-overlapping batch (e.g. a scan
/// batch past a contiguous coverage's bits) in `O(containers)` without touching
/// individual cells. The in-range bits then carry **consecutive** coverage ranks
/// starting at the count of bits below `base` (one `rank` lookup) — no bits lie
/// between them by construction — so ranks need no running count. Coverages are
/// processed newest-first with a "first claim wins" guard for precedence.
fn route_contiguous(
    base: u32,
    len: usize,
    coverages_newest_first: &[&RoaringBitmap],
) -> OverlayRouting {
    let mut needed_ranks: Vec<Vec<u32>> = vec![Vec::new(); coverages_newest_first.len()];
    let mut routed: Vec<Option<(usize, usize)>> = vec![None; len];
    let range_end = (base as u64 + len as u64).min(u32::MAX as u64) as u32;
    let mut batch_range = RoaringBitmap::new();
    batch_range.insert_range(base..range_end);

    for (k, coverage) in coverages_newest_first.iter().enumerate() {
        let in_range = *coverage & &batch_range;
        if in_range.is_empty() {
            continue;
        }
        // 0-based rank of the first in-range cell: the coverage bits below `base`.
        let base_rank = if base == 0 {
            0
        } else {
            coverage.rank(base - 1) as u32
        };
        for (i, offset) in in_range.iter().enumerate() {
            let row = (offset - base) as usize;
            if routed[row].is_none() {
                routed[row] = Some((k, needed_ranks[k].len()));
                needed_ranks[k].push(base_rank + i as u32);
            }
        }
    }

    let mut any_overlay = false;
    let indices = routed
        .into_iter()
        .enumerate()
        .map(|(i, routed)| match routed {
            None => (0, i),
            Some((k, pos)) => {
                any_overlay = true;
                (k + 1, pos)
            }
        })
        .collect();

    OverlayRouting {
        indices,
        needed_ranks,
        any_overlay,
    }
}

/// General path for arbitrary (e.g. `take`) offsets: probe each offset against
/// the coverages newest-first. `take` batches are small, so the `O(N * K)`
/// probing here is not a bottleneck.
fn route_arbitrary(offsets: &[u32], coverages_newest_first: &[&RoaringBitmap]) -> OverlayRouting {
    let mut rank_sets: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); coverages_newest_first.len()];
    let mut raw: Vec<Option<(usize, u32)>> = Vec::with_capacity(offsets.len());
    for &offset in offsets {
        let mut routed = None;
        for (k, coverage) in coverages_newest_first.iter().enumerate() {
            if coverage.contains(offset) {
                // 0-based rank: number of set bits strictly below `offset`.
                let rank = coverage.rank(offset) as u32 - 1;
                rank_sets[k].insert(rank);
                routed = Some((k, rank));
                break;
            }
        }
        raw.push(routed);
    }

    let needed_ranks: Vec<Vec<u32>> = rank_sets
        .iter()
        .map(|ranks| ranks.iter().copied().collect())
        .collect();
    let rank_positions: Vec<HashMap<u32, usize>> = needed_ranks
        .iter()
        .map(|ranks| ranks.iter().enumerate().map(|(pos, &r)| (r, pos)).collect())
        .collect();

    let mut any_overlay = false;
    let indices = raw
        .into_iter()
        .enumerate()
        .map(|(i, routed)| match routed {
            None => (0, i),
            Some((k, rank)) => {
                any_overlay = true;
                (k + 1, rank_positions[k][&rank])
            }
        })
        .collect();

    OverlayRouting {
        indices,
        needed_ranks,
        any_overlay,
    }
}

/// Assemble the merged column from `base` and the per-overlay values fetched for
/// the ranks [`route_overlays`] asked for.
///
/// `fetched_newest_first[k]` holds overlay `k`'s values for `routing`'s
/// `needed_ranks[k]`, in that order. The result has the same length and data
/// type as `base`; a covered offset whose overlay value is NULL resolves **to**
/// NULL (distinct from a fall-through, which keeps its base value).
pub fn assemble_overlay_column(
    base: &ArrayRef,
    routing: &OverlayRouting,
    fetched_newest_first: &[ArrayRef],
) -> Result<ArrayRef> {
    if routing.all_fall_through() {
        return Ok(base.clone());
    }
    if fetched_newest_first.len() != routing.needed_ranks.len() {
        return Err(Error::invalid_input(format!(
            "overlay assembly got {} value columns but routing expects {}",
            fetched_newest_first.len(),
            routing.needed_ranks.len()
        )));
    }
    for (k, values) in fetched_newest_first.iter().enumerate() {
        if values.len() != routing.needed_ranks[k].len() {
            return Err(Error::invalid_input(format!(
                "overlay value column {} has {} values but {} ranks were requested",
                k,
                values.len(),
                routing.needed_ranks[k].len()
            )));
        }
    }

    let mut sources: Vec<&dyn Array> = Vec::with_capacity(fetched_newest_first.len() + 1);
    sources.push(base.as_ref());
    for values in fetched_newest_first {
        sources.push(values.as_ref());
    }
    interleave(&sources, &routing.indices).map_err(Error::from)
}

/// One overlay's contribution to one projected field: which physical offsets it
/// covers, and an opened (but unread) reader over the overlay file from which the
/// field's value column is fetched by coverage rank at merge time.
#[derive(Debug, Clone)]
struct LoadedFieldOverlay {
    /// Physical offsets this overlay covers for the field.
    coverage: Arc<RoaringBitmap>,
    /// Reader over the overlay data file, projected to this field; shared across
    /// the fields that the same file covers.
    reader: Arc<dyn GenericFileReader>,
    /// Single-field projection used when fetching the value column.
    field_projection: Arc<Schema>,
}

/// The overlays that apply to a single projected field, ordered newest-first.
/// `field_name` is the top-level read-batch column name the plan applies to.
#[derive(Debug, Clone)]
pub struct FieldOverlayPlan {
    field_name: String,
    overlays_newest_first: Vec<LoadedFieldOverlay>,
}

/// Open the overlay value-column readers for `fragment`'s projected fields,
/// ordered newest-first, ready to be merged into base batches on read.
///
/// Each contributing overlay *file* is opened once (its metadata loaded),
/// projected to the fields it covers that the read also projects. The value
/// columns themselves are **not** read here — the per-batch merge fetches only
/// the coverage ranks it needs (see [`merge_overlay_batch`]), so a `take` of a
/// few rows no longer reads a whole overlay column.
///
/// For each projected (top-level) field, the fragment's overlays are walked
/// newest-first; an overlay contributes if its `data_file.fields` includes the
/// field. Overlays on nested (non-top-level) fields are not yet supported and are
/// simply not matched here.
pub async fn load_overlay_plan(
    fragment: &FileFragment,
    projection: &Schema,
    read_config: &FragReadConfig,
) -> Result<Vec<FieldOverlayPlan>> {
    let order = overlay_indices_newest_first(&fragment.metadata.overlays);

    // Open each contributing overlay file once, concurrently. The reader is
    // shared (via `Arc`) by every field that file covers.
    //
    // TODO(overlay perf): these reads use the default reader priority. Once
    // we benchmark take/scan over overlays, decide whether overlay value
    // reads should inherit `read_config.reader_priority` (or get a dedicated
    // priority) so they schedule alongside the base reads.
    let opened: Vec<Option<Arc<dyn GenericFileReader>>> =
        futures::future::try_join_all(order.iter().map(|&overlay_idx| async move {
            let overlay = &fragment.metadata.overlays[overlay_idx];
            let covered: Vec<lance_core::datatypes::Field> = projection
                .fields
                .iter()
                .filter(|f| overlay.data_file.fields.contains(&f.id))
                .cloned()
                .collect();
            if covered.is_empty() {
                return Ok::<_, Error>(None);
            }
            let schema = Schema {
                fields: covered,
                metadata: Default::default(),
            };
            Ok(fragment
                .open_reader(&overlay.data_file, Some(&schema), read_config)
                .await?
                .map(Arc::from))
        }))
        .await?;

    let mut plans = Vec::new();
    for field in &projection.fields {
        let mut overlays_newest_first = Vec::new();
        for (slot, &overlay_idx) in order.iter().enumerate() {
            let overlay = &fragment.metadata.overlays[overlay_idx];
            let Some(field_pos) = overlay
                .data_file
                .fields
                .iter()
                .position(|&id| id == field.id)
            else {
                continue;
            };
            let Some(reader) = &opened[slot] else {
                continue;
            };
            overlays_newest_first.push(LoadedFieldOverlay {
                coverage: overlay.coverage_for_field(field_pos)?,
                reader: reader.clone(),
                field_projection: Arc::new(Schema {
                    fields: vec![field.clone()],
                    metadata: Default::default(),
                }),
            });
        }
        if !overlays_newest_first.is_empty() {
            plans.push(FieldOverlayPlan {
                field_name: field.name.clone(),
                overlays_newest_first,
            });
        }
    }
    Ok(plans)
}

/// Resolve overlays for one base batch: route each projected field against the
/// batch's physical `offsets`, fetch only the coverage ranks the batch touches
/// (concurrently with the base read), and assemble the merged columns. Fields
/// with no plan, and the row-id/row-address system columns, pass through.
pub async fn merge_overlay_batch(
    base: ReadBatchFut,
    offsets: &[u32],
    plans: &[FieldOverlayPlan],
) -> Result<RecordBatch> {
    let field_work = futures::future::try_join_all(plans.iter().map(|plan| async move {
        let coverages: Vec<&RoaringBitmap> = plan
            .overlays_newest_first
            .iter()
            .map(|overlay| overlay.coverage.as_ref())
            .collect();
        let routing = route_overlays(offsets, &coverages);
        if routing.all_fall_through() {
            return Ok::<_, Error>((plan.field_name.as_str(), None));
        }
        let fetched = futures::future::try_join_all(
            plan.overlays_newest_first
                .iter()
                .zip(routing.needed_ranks())
                .map(|(overlay, ranks)| {
                    fetch_overlay_ranks(
                        overlay.reader.as_ref(),
                        overlay.field_projection.clone(),
                        ranks,
                    )
                }),
        )
        .await?;
        Ok((plan.field_name.as_str(), Some((routing, fetched))))
    }));

    // The base read and every overlay value read proceed concurrently.
    let (batch, resolved) = futures::future::try_join(base, field_work).await?;

    let schema = batch.schema();
    let mut columns = batch.columns().to_vec();
    for (field_name, work) in resolved {
        let Some((routing, fetched)) = work else {
            continue;
        };
        let Some(idx) = schema.index_of(field_name).ok() else {
            // The plan's field is not in this batch's projection; skip it.
            continue;
        };
        columns[idx] = assemble_overlay_column(&columns[idx], &routing, &fetched)?;
    }
    Ok(RecordBatch::try_new(schema, columns)?)
}

/// Fetch one overlay's value column at the given coverage `ranks` (sorted and
/// unique). Returns a column of `ranks.len()` values aligned with `ranks`; an
/// empty `ranks` reads nothing and yields an empty column.
async fn fetch_overlay_ranks(
    reader: &dyn GenericFileReader,
    projection: Arc<Schema>,
    ranks: &[u32],
) -> Result<ArrayRef> {
    if ranks.is_empty() {
        return Ok(arrow_array::new_empty_array(
            &projection.fields[0].data_type(),
        ));
    }
    let mut tasks = reader
        .take_all_tasks(ranks, ranks.len() as u32, projection, None)
        .await?;
    let mut chunks: Vec<ArrayRef> = Vec::new();
    while let Some(task) = tasks.next().await {
        let batch = task.task.await?;
        chunks.push(batch.column(0).clone());
    }
    let chunk_refs: Vec<&dyn arrow_array::Array> = chunks.iter().map(|a| a.as_ref()).collect();
    Ok(arrow_select::concat::concat(&chunk_refs)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray, UInt32Array};
    use std::sync::Arc;

    fn i32_array(values: impl IntoIterator<Item = Option<i32>>) -> ArrayRef {
        Arc::new(Int32Array::from_iter(values))
    }

    fn bitmap(offsets: impl IntoIterator<Item = u32>) -> RoaringBitmap {
        RoaringBitmap::from_iter(offsets)
    }

    /// Physical offsets for a contiguous range `[start, start + len)`.
    fn offsets(start: u32, len: usize) -> Vec<u32> {
        (start..start + len as u32).collect()
    }

    /// Drive the production flow purely in memory: route against the coverage
    /// bitmaps, then fetch just the requested ranks from each overlay's *full*
    /// value column (exactly what the rank-pushdown `take` does on disk), then
    /// assemble. `overlays_newest_first` holds each overlay's `(coverage, full
    /// value column indexed by rank)`.
    fn resolve(
        base: &ArrayRef,
        offsets: &[u32],
        overlays_newest_first: &[(RoaringBitmap, ArrayRef)],
    ) -> ArrayRef {
        let coverages: Vec<&RoaringBitmap> = overlays_newest_first.iter().map(|(c, _)| c).collect();
        let routing = route_overlays(offsets, &coverages);
        let fetched: Vec<ArrayRef> = overlays_newest_first
            .iter()
            .zip(routing.needed_ranks())
            .map(|((_, full), ranks)| {
                let indices = UInt32Array::from(ranks.clone());
                arrow_select::take::take(full.as_ref(), &indices, None).unwrap()
            })
            .collect();
        assemble_overlay_column(base, &routing, &fetched).unwrap()
    }

    fn assert_i32_eq(actual: &ArrayRef, expected: impl IntoIterator<Item = Option<i32>>) {
        let actual = actual.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(actual, &Int32Array::from_iter(expected));
    }

    #[test]
    fn test_no_overlays_returns_base() {
        let base = i32_array([Some(1), Some(2), Some(3)]);
        let resolved = resolve(&base, &offsets(0, 3), &[]);
        assert_i32_eq(&resolved, [Some(1), Some(2), Some(3)]);
    }

    #[test]
    fn test_single_overlay_rank_addressing() {
        // Base ages [30, 25, 40, 22]; overlay sets offset 1 -> 26 (rank 0).
        let base = i32_array([Some(30), Some(25), Some(40), Some(22)]);
        let overlay = (bitmap([1]), i32_array([Some(26)]));
        let resolved = resolve(&base, &offsets(0, 4), &[overlay]);
        assert_i32_eq(&resolved, [Some(30), Some(26), Some(40), Some(22)]);
    }

    #[test]
    fn test_rank_addressing_multiple_offsets() {
        // Coverage {0, 2, 3} -> values at ranks 0,1,2.
        let base = i32_array([Some(10), Some(11), Some(12), Some(13)]);
        let overlay = (
            bitmap([0, 2, 3]),
            i32_array([Some(100), Some(120), Some(130)]),
        );
        let resolved = resolve(&base, &offsets(0, 4), &[overlay]);
        assert_i32_eq(&resolved, [Some(100), Some(11), Some(120), Some(130)]);
    }

    #[test]
    fn test_newest_overlay_wins() {
        // Two overlays both cover offset 1; the newest (first in the slice) wins.
        let base = i32_array([Some(0), Some(1), Some(2)]);
        let newest = (bitmap([1]), i32_array([Some(999)]));
        let older = (bitmap([1, 2]), i32_array([Some(111), Some(222)]));
        let resolved = resolve(&base, &offsets(0, 3), &[newest, older]);
        // offset 1 -> newest (999); offset 2 -> only older covers it (222).
        assert_i32_eq(&resolved, [Some(0), Some(999), Some(222)]);
    }

    #[test]
    fn test_null_override_vs_fall_through() {
        // A covered offset with a NULL value overrides the cell to NULL; an
        // absent offset falls through to the base.
        let base = i32_array([Some(1), Some(2), Some(3)]);
        let overlay = (bitmap([0]), i32_array([None]));
        let resolved = resolve(&base, &offsets(0, 3), &[overlay]);
        assert_i32_eq(&resolved, [None, Some(2), Some(3)]);
    }

    #[test]
    fn test_physical_start_offset() {
        // The batch covers physical rows [10, 13); the overlay covers offset 11.
        let base = i32_array([Some(0), Some(0), Some(0)]);
        let overlay = (bitmap([11]), i32_array([Some(7)]));
        let resolved = resolve(&base, &offsets(10, 3), &[overlay]);
        assert_i32_eq(&resolved, [Some(0), Some(7), Some(0)]);
    }

    #[test]
    fn test_string_column_merge() {
        let base: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let overlay = (
            bitmap([0, 2]),
            Arc::new(StringArray::from(vec!["A", "C"])) as ArrayRef,
        );
        let resolved = resolve(&base, &offsets(0, 3), &[overlay]);
        let expected: ArrayRef = Arc::new(StringArray::from(vec!["A", "b", "C"]));
        assert_eq!(&resolved, &expected);
    }

    #[test]
    fn test_non_contiguous_offsets() {
        // `take` supplies arbitrary, non-contiguous physical offsets. The base
        // rows correspond to offsets 5, 1, 8 (in that order); the overlay covers
        // offsets {1, 8} with values at ranks 0, 1.
        let base = i32_array([Some(50), Some(10), Some(80)]);
        let overlay = (bitmap([1, 8]), i32_array([Some(11), Some(88)]));
        let resolved = resolve(&base, &[5, 1, 8], &[overlay]);
        // offset 5 uncovered -> base 50; offset 1 -> rank 0 (11); offset 8 -> rank 1 (88).
        assert_i32_eq(&resolved, [Some(50), Some(11), Some(88)]);
    }

    #[test]
    fn test_routing_dedups_repeated_ranks() {
        // A `take` may request the same offset twice; both rows must route to the
        // same rank, and that rank is fetched only once.
        let coverage = bitmap([2, 5]);
        let routing = route_overlays(&[5, 2, 5], &[&coverage]);
        // Offset 5 is rank 1, offset 2 is rank 0: distinct ranks {0, 1}, sorted.
        assert_eq!(routing.needed_ranks(), &[vec![0, 1]]);
        let full = i32_array([Some(20), Some(50)]); // values at ranks 0, 1
        let fetched = vec![
            arrow_select::take::take(
                full.as_ref(),
                &UInt32Array::from(routing.needed_ranks()[0].clone()),
                None,
            )
            .unwrap(),
        ];
        let base = i32_array([Some(0), Some(0), Some(0)]);
        let resolved = assemble_overlay_column(&base, &routing, &fetched).unwrap();
        assert_i32_eq(&resolved, [Some(50), Some(20), Some(50)]);
    }

    #[test]
    fn test_assemble_value_count_mismatch_errors() {
        let coverage = bitmap([0, 1]);
        let routing = route_overlays(&[0, 1], &[&coverage]);
        let base = i32_array([Some(1), Some(2)]);
        // One value supplied for two requested ranks is a caller bug.
        let fetched = vec![i32_array([Some(9)])];
        assert!(assemble_overlay_column(&base, &routing, &fetched).is_err());
    }

    #[test]
    fn test_contiguous_fast_path_matches_general() {
        // The contiguous fast path must produce byte-for-byte identical routing
        // to the general offset-major path for any contiguous batch. Fuzz a range
        // of bases, lengths, overlay counts, and coverage densities — including
        // bits outside the batch range — and compare both fields.
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for _ in 0..500 {
            let base = next() % 64;
            let len = (next() % 48 + 1) as usize;
            let num_overlays = (next() % 5) as usize;
            let coverages: Vec<RoaringBitmap> = (0..num_overlays)
                .map(|_| {
                    let density = next() % 101;
                    let mut b = RoaringBitmap::new();
                    for off in base.saturating_sub(3)..base + len as u32 + 3 {
                        if next() % 100 < density {
                            b.insert(off);
                        }
                    }
                    b
                })
                .collect();
            let refs: Vec<&RoaringBitmap> = coverages.iter().collect();
            let contiguous_offsets: Vec<u32> = (base..base + len as u32).collect();

            let fast = route_contiguous(base, len, &refs);
            let general = route_arbitrary(&contiguous_offsets, &refs);
            assert_eq!(fast.indices, general.indices, "indices differ");
            assert_eq!(
                fast.needed_ranks, general.needed_ranks,
                "needed_ranks differ"
            );
            assert_eq!(fast.any_overlay, general.any_overlay, "any_overlay differs");
        }
    }

    #[test]
    fn test_overlay_ordering_newest_first() {
        use lance_table::format::DataFile;
        use lance_table::format::overlay::OverlayCoverage;
        let mk = |version: u64| DataOverlayFile {
            data_file: DataFile::new_legacy_from_fields("o.lance", vec![1], None),
            coverage: OverlayCoverage::dense(RoaringBitmap::new()),
            committed_version: version,
        };
        // List order [v2, v5, v3]; newest-first should be v5(idx1), v3(idx2), v2(idx0).
        let overlays = vec![mk(2), mk(5), mk(3)];
        assert_eq!(overlay_indices_newest_first(&overlays), vec![1, 2, 0]);

        // Equal versions: later list position is newer.
        let overlays = vec![mk(4), mk(4)];
        assert_eq!(overlay_indices_newest_first(&overlays), vec![1, 0]);
    }
}
