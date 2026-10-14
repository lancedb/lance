// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Ordered, deletion-aware lookup of unique keys through ordinary scalar indices.

use std::collections::{BTreeMap, HashMap};

use arrow_array::{Array, cast::AsArray, make_array, types::UInt64Type};
use arrow_row::{RowConverter, SortField};
use futures::TryStreamExt;
use lance_core::utils::address::RowAddress;
use lance_core::{Error, ROW_ADDR, Result};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::registry::VALUE_COLUMN_NAME;
use lance_table::format::IndexMetadata;
use lance_table::format::overlay::staleness::overlay_affects_index;
use roaring::RoaringBitmap;

use crate::Dataset;
use crate::index::{DatasetIndexExt, scalar::open_scalar_index};

struct LookupSegment<'a> {
    metadata: &'a IndexMetadata,
    coverage: RoaringBitmap,
}

#[derive(Default)]
struct LookupGroup<'a> {
    segments: Vec<LookupSegment<'a>>,
    coverage: RoaringBitmap,
}

impl Dataset {
    /// Resolve unique keys to live physical row addresses, preserving input order.
    ///
    /// Returns one entry per key: repeated keys repeat their address, and null,
    /// missing, or deleted keys yield `None`. If a requested non-null key occurs
    /// in more than one live row, this returns an error. Uniqueness is checked
    /// for the requested keys, not asserted for the rest of the column.
    ///
    /// An ordinary address-domain BTree index supplies covered fragments; other
    /// fragments are scanned in batches. Without an index the entire column is
    /// scanned. Keys must have exactly the column's Arrow type. This method
    /// operates on this dataset snapshot, including its branch lineage.
    pub(crate) async fn lookup_row_addresses(
        &self,
        column: &str,
        keys: &dyn Array,
    ) -> Result<Vec<Option<RowAddress>>> {
        let schema = self.index_schema()?;
        let field = schema.field(column).ok_or_else(|| {
            Error::invalid_input(format!("Lookup column '{column}' does not exist"))
        })?;
        if keys.data_type() != &field.data_type() {
            return Err(Error::invalid_input(format!(
                "Lookup keys for '{column}' have type {}, expected {}",
                keys.data_type(),
                field.data_type()
            )));
        }
        if u32::try_from(keys.len()).is_err() {
            return Err(Error::invalid_input(format!(
                "Lookup accepts at most {} keys, got {}",
                u32::MAX,
                keys.len()
            )));
        }
        let mut addresses = vec![None; keys.len()];
        if keys.len() == keys.null_count() {
            return Ok(addresses);
        }

        let indices = self.load_indices().await?;
        let mut groups: BTreeMap<&str, LookupGroup<'_>> = BTreeMap::new();
        for index in indices.iter().filter(|index| {
            index.keyed_field() == Some(field.id)
                && (!self.manifest.uses_stable_row_ids() || index.results_are_row_addrs())
                && index
                    .index_details
                    .as_ref()
                    .is_some_and(|details| details.type_url.ends_with("BTreeIndexDetails"))
        }) {
            // Unknown coverage cannot distinguish an absent key from an unindexed
            // one. Such a segment contributes nothing; the column is scanned.
            let Some(coverage) = &index.fragment_bitmap else {
                continue;
            };
            let mut coverage = coverage & self.fragment_bitmap.as_ref();
            // Overlay rows can change their key without moving their address.
            // Read their fragments from the column until index maintenance
            // incorporates them, using the same field/version staleness rule
            // as scalar-filter planning.
            for fragment in self
                .fragments()
                .iter()
                .filter(|fragment| !fragment.overlays.is_empty())
            {
                if fragment.overlays.iter().any(|overlay| {
                    overlay_affects_index(
                        overlay,
                        index.keyed_fields(),
                        index.dataset_version,
                        &schema,
                    )
                }) {
                    coverage.remove(fragment.id as u32);
                }
            }
            let group = groups.entry(&index.name).or_default();
            group.coverage |= &coverage;
            group.segments.push(LookupSegment {
                metadata: index,
                coverage,
            });
        }
        let group = groups
            .into_values()
            .max_by_key(|group| group.coverage.len())
            .unwrap_or_default();

        // The index returns key positions rather than a set, so no sorting or
        // rejoining of the caller's keys is needed, even with duplicate inputs.
        for LookupSegment {
            metadata: segment,
            coverage,
        } in group.segments
        {
            if coverage.is_empty() {
                continue;
            }
            let index = open_scalar_index(self, column, segment, &NoOpMetricsCollector).await?;
            let matches = index.lookup(keys, &NoOpMetricsCollector).await?;
            let mut by_fragment = BTreeMap::<u32, Vec<(usize, RowAddress)>>::new();
            for (key, address) in matches.key_indices.iter().zip(matches.row_ids.iter()) {
                let (Some(key), Some(address)) = (key, address) else {
                    return Err(Error::index(
                        "Scalar lookup returned a null key position or address",
                    ));
                };
                let address = RowAddress::from(address);
                if coverage.contains(address.fragment_id()) {
                    by_fragment
                        .entry(address.fragment_id())
                        .or_default()
                        .push((key as usize, address));
                }
            }
            for (fragment_id, matches) in by_fragment {
                let Some(fragment) = self.get_fragment(fragment_id as usize) else {
                    continue;
                };
                let deletions = fragment.get_deletion_vector().await?;
                let physical_rows = fragment.physical_rows().await?;
                for (key, address) in matches {
                    if address.row_offset() as usize >= physical_rows {
                        return Err(Error::index(format!(
                            "Lookup returned address {address} outside fragment {fragment_id} with {physical_rows} rows"
                        )));
                    }
                    if deletions
                        .as_ref()
                        .is_some_and(|deleted| deleted.contains(address.row_offset()))
                    {
                        continue;
                    }
                    assign_unique_address(&mut addresses, key, address, column)?;
                }
            }
        }

        let uncovered = self
            .fragments()
            .iter()
            .filter(|fragment| !group.coverage.contains(fragment.id as u32))
            .cloned()
            .collect::<Vec<_>>();
        if uncovered.is_empty() {
            return Ok(addresses);
        }

        // Retain only the requested keys and consume scan batches incrementally;
        // an uncovered tail never constructs another dataset-wide reverse index.
        let keys = make_array(keys.to_data());
        let converter = RowConverter::new(vec![SortField::new(keys.data_type().clone())])?;
        let key_rows = converter.convert_columns(std::slice::from_ref(&keys))?;
        let mut positions = HashMap::with_capacity(keys.len() - keys.null_count());
        for position in 0..keys.len() {
            if keys.is_valid(position) {
                positions
                    .entry(key_rows.row(position))
                    .or_insert_with(Vec::new)
                    .push(position);
            }
        }
        let mut scan = self.scan();
        scan.with_fragments(uncovered)
            .with_row_address()
            .use_scalar_index(false)
            .project_with_transform(&[(VALUE_COLUMN_NAME, column)])?;
        // Scans can themselves contain row-ID takes. Box this boundary so the
        // two async execution paths do not recursively expand their future types.
        let mut stream = Box::pin(scan.try_into_stream()).await?;
        while let Some(batch) = stream.try_next().await? {
            let values = batch
                .column_by_name(VALUE_COLUMN_NAME)
                .ok_or_else(|| Error::internal("Lookup scan omitted the key column"))?;
            let row_addresses = batch
                .column_by_name(ROW_ADDR)
                .ok_or_else(|| Error::internal("Lookup scan omitted the row address column"))?
                .as_primitive::<UInt64Type>();
            let rows = converter.convert_columns(std::slice::from_ref(values))?;
            for offset in 0..batch.num_rows() {
                if values.is_null(offset) || row_addresses.is_null(offset) {
                    continue;
                }
                if let Some(keys) = positions.get(&rows.row(offset)) {
                    let address = RowAddress::from(row_addresses.value(offset));
                    for key in keys {
                        assign_unique_address(&mut addresses, *key, address, column)?;
                    }
                }
            }
        }
        Ok(addresses)
    }
}

fn assign_unique_address(
    addresses: &mut [Option<RowAddress>],
    key: usize,
    address: RowAddress,
    column: &str,
) -> Result<()> {
    let num_keys = addresses.len();
    let entry = addresses.get_mut(key).ok_or_else(|| {
        Error::index(format!(
            "Lookup returned key position {key} for {num_keys} keys"
        ))
    })?;
    if let Some(previous) = *entry
        && previous != address
    {
        return Err(Error::invalid_input(format!(
            "Lookup column '{column}' is not unique for key at position {key}: live addresses {previous} and {address}"
        )));
    }
    *entry = Some(address);
    Ok(())
}
