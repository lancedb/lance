// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::Dataset;
use crate::Result;
use arrow_array::{RecordBatch, UInt32Array};
use arrow_schema::DataType;
use futures::join;
use lance_arrow::RecordBatchExt;
use lance_core::datatypes::Field;
use lance_table::format::{Fragment, Manifest};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// The delta dataset between two versions of a dataset.
pub struct DeltaDataset {
    /// The base version number for comparison.
    pub(crate) begin_version: u64,
    /// Manifest of the beginning version.
    pub(crate) begin_manifest: Arc<Manifest>,
    /// The current version number.
    pub(crate) end_version: u64,
    /// Manifest of the ending version.
    pub(crate) end_manifest: Arc<Manifest>,
    /// Base path of the dataset.
    pub(crate) base_dataset: Dataset,
}

impl DeltaDataset {
    async fn extract_modified_fragments(
        &self,
        begin_frags: &HashMap<u64, &Fragment>,
        end_frags: &HashMap<u64, &Fragment>,
        added_fragments: Vec<Fragment>,
        removed_fragments: Vec<Fragment>,
    ) -> Result<Vec<Fragment>> {
        let (begin_dataset, end_dataset) = join!(
            self.base_dataset.checkout_version(self.begin_version),
            self.base_dataset.checkout_version(self.end_version)
        );
        let (begin_dataset, end_dataset) = (begin_dataset?, end_dataset?);

        let added_ids: HashSet<u64> = added_fragments.iter().map(|f| f.id).collect();
        let removed_ids: HashSet<u64> = removed_fragments.iter().map(|f| f.id).collect();

        let common_ids: HashSet<u64> = begin_frags
            .keys()
            .filter(|id| end_frags.contains_key(id))
            .copied()
            .collect();
        let candidate_ids = common_ids
            .difference(&added_ids)
            .copied()
            .collect::<HashSet<_>>()
            .difference(&removed_ids)
            .copied()
            .collect::<Vec<_>>();

        let mut modified = Vec::with_capacity(candidate_ids.len());

        for &id in &candidate_ids {
            let (begin_frag, end_frag) = (begin_frags[&id], end_frags[&id]);

            if begin_frag != end_frag {
                modified.push(end_frag.clone());
                continue;
            }

            let (begin_reader, end_reader) = (
                begin_dataset.get_fragment(begin_frag.id as usize),
                end_dataset.get_fragment(end_frag.id as usize),
            );

            let (begin_dv, end_dv) = join!(
                async {
                    match begin_reader {
                        Some(reader) => reader.get_deletion_vector().await,
                        None => Ok(None),
                    }
                },
                async {
                    match end_reader {
                        Some(reader) => reader.get_deletion_vector().await,
                        None => Ok(None),
                    }
                }
            );

            match (begin_dv?, end_dv?) {
                (Some(begin_dv), Some(end_dv)) if begin_dv != end_dv => {
                    modified.push(end_frag.clone())
                }
                (Some(_), None) | (None, Some(_)) => modified.push(end_frag.clone()),
                _ => (),
            }
        }

        Ok(modified)
    }

    /// Computes the differences between the two versions.
    pub async fn diff_metadata(&self) -> Result<DeltaMetadata> {
        let begin_frags: HashMap<u64, &Fragment> = self
            .begin_manifest
            .fragments
            .iter()
            .map(|f| (f.id, f))
            .collect();
        let end_frags: HashMap<u64, &Fragment> = self
            .end_manifest
            .fragments
            .iter()
            .map(|f| (f.id, f))
            .collect();

        let begin_ids: HashSet<u64> = begin_frags.keys().copied().collect();
        let end_ids: HashSet<u64> = end_frags.keys().copied().collect();

        let added_fragments: Vec<Fragment> = self
            .end_manifest
            .fragments
            .iter()
            .filter(|f| !begin_ids.contains(&f.id))
            .cloned()
            .collect();

        let removed_fragments: Vec<Fragment> = self
            .begin_manifest
            .fragments
            .iter()
            .filter(|f| !end_ids.contains(&f.id))
            .cloned()
            .collect();

        let modified_fragments = self
            .extract_modified_fragments(
                &begin_frags,
                &end_frags,
                added_fragments.clone(),
                removed_fragments.clone(),
            )
            .await;

        let schema_diff = if self.begin_manifest.schema != self.end_manifest.schema {
            Some(self.compute_schema_diff())
        } else {
            None
        };

        Ok(DeltaMetadata {
            added_fragments: Some(added_fragments),
            removed_fragments: Some(removed_fragments),
            modified_fragments: Some(modified_fragments?),
            schema_changes: schema_diff,
        })
    }

    pub async fn diff(&self) -> Result<DeltaData> {
        let metadata = self.diff_metadata().await;

        let mut added_records = Vec::new();
        let mut removed_records = Vec::new();

        if let Some(added_frags) = metadata
            .as_ref()
            .ok()
            .and_then(|m| m.added_fragments.as_ref())
        {
            for frag in added_frags {
                if let Some(fragment_reader) = self.base_dataset.get_fragment(frag.id as usize) {
                    if let Ok(batch) = fragment_reader.scan().try_into_batch().await {
                        added_records.push(batch);
                    }
                }

                // if it has deletion vector, we need to handle it.
                // because in some middle versions, some data may be deleted
                if let Some(frag_reader) = self.base_dataset.get_fragment(frag.id as usize) {
                    if let Ok(Some(dv)) = frag_reader.get_deletion_vector().await {
                        if !dv.is_empty() {
                            if let Ok(batch) = frag_reader.scan().try_into_batch().await {
                                let indices = UInt32Array::from(dv.iter().collect::<Vec<u32>>());
                                if let Ok(deleted_rows) = batch.take(&indices) {
                                    removed_records.push(deleted_rows);
                                }
                            }
                        }
                    }
                }
            }
        }

        let old_dataset = self
            .base_dataset
            .checkout_version(self.begin_version)
            .await?;

        if let Some(removed_frags) = metadata
            .as_ref()
            .ok()
            .and_then(|m| m.removed_fragments.as_ref())
        {
            for frag in removed_frags {
                if let Some(fragment_reader) = old_dataset.get_fragment(frag.id as usize) {
                    if let Ok(batch) = fragment_reader.scan().try_into_batch().await {
                        removed_records.push(batch);
                    }
                }
            }
        }

        let (begin_dataset, end_dataset) = join!(
            self.base_dataset.checkout_version(self.begin_version),
            self.base_dataset.checkout_version(self.end_version)
        );

        let begin_dataset = begin_dataset?;
        let end_dataset = end_dataset?;

        if let Some(modified_frags) = metadata
            .as_ref()
            .ok()
            .and_then(|m| m.modified_fragments.as_ref())
        {
            for frag in modified_frags {
                let (begin_frag, end_frag) = (
                    begin_dataset.get_fragment(frag.id as usize),
                    end_dataset.get_fragment(frag.id as usize),
                );

                match (begin_frag, end_frag) {
                    // Case 3: Both fragments exist (original logic)
                    (Some(begin_fragment), Some(end_fragment)) => {
                        let (begin_dv, end_dv) = join!(
                            begin_fragment.get_deletion_vector(),
                            end_fragment.get_deletion_vector()
                        );

                        let begin_dv = begin_dv.unwrap();
                        let end_dv = end_dv.unwrap();

                        let new_deletions = match (&begin_dv, &end_dv) {
                            (None, None) => continue,
                            (None, Some(end_vec)) => end_vec.iter().collect(),
                            (Some(_), None) => Vec::new(),
                            (Some(begin_vec), Some(end_vec)) => {
                                let begin_set: HashSet<_> = begin_vec.iter().collect();
                                end_vec
                                    .iter()
                                    .filter(|idx| !begin_set.contains(idx))
                                    .collect()
                            }
                        };

                        if !new_deletions.is_empty() {
                            if let Ok(batch) = end_fragment.scan().try_into_batch().await {
                                let indices = UInt32Array::from(new_deletions);
                                if let Ok(deleted_rows) = batch.take(&indices) {
                                    removed_records.push(deleted_rows);
                                }
                            }
                        }
                    }

                    // Case 1: Only begin fragment exists
                    (Some(begin_fragment), None) => {
                        if let Ok(Some(dv)) = begin_fragment.get_deletion_vector().await {
                            if !dv.is_empty() {
                                if let Ok(batch) = begin_fragment.scan().try_into_batch().await {
                                    let indices = UInt32Array::from(dv.iter().collect::<Vec<_>>());
                                    if let Ok(deleted_rows) = batch.take(&indices) {
                                        removed_records.push(deleted_rows);
                                    }
                                }
                            }
                        }
                    }

                    // Case 2: Only end fragment exists
                    (None, Some(end_fragment)) => {
                        if let Ok(Some(dv)) = end_fragment.get_deletion_vector().await {
                            if !dv.is_empty() {
                                if let Ok(batch) = end_fragment.scan().try_into_batch().await {
                                    let indices = UInt32Array::from(dv.iter().collect::<Vec<_>>());
                                    if let Ok(deleted_rows) = batch.take(&indices) {
                                        removed_records.push(deleted_rows);
                                    }
                                }
                            }
                        }
                    }

                    _ => {}
                }
            }
        }
        Ok(DeltaData {
            added_record_batches: Some(added_records),
            removed_record_batches: Some(removed_records),
        })
    }

    /// Compute schema differences.
    fn compute_schema_diff(&self) -> DeltaSchema {
        let begin_map: HashMap<_, _> = self
            .begin_manifest
            .schema
            .fields
            .iter()
            .map(|f| (f.name.as_str(), f))
            .collect();
        let end_map: HashMap<_, _> = self
            .end_manifest
            .schema
            .fields
            .iter()
            .map(|f| (f.name.as_str(), f))
            .collect();

        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut modified = Vec::new();

        for (name, field) in &end_map {
            match begin_map.get(name) {
                None => added.push((*field).clone()),
                Some(begin_field) if begin_field != field => {
                    modified.push(FieldChange {
                        name: name.to_string(),
                        old_type: begin_field.data_type().clone(),
                        new_type: field.data_type().clone(),
                    });
                }
                _ => {}
            }
        }

        for (name, field) in begin_map {
            if !end_map.contains_key(name) {
                removed.push((*field).clone());
            }
        }

        DeltaSchema {
            added_fields: Some(added).filter(|v| !v.is_empty()),
            removed_fields: Some(removed).filter(|v| !v.is_empty()),
            modified_fields: Some(modified).filter(|v| !v.is_empty()),
        }
    }
}

/// The metadata differences between two versions.
#[derive(Debug)]
pub struct DeltaMetadata {
    /// Fragments added between begin and end version.
    pub added_fragments: Option<Vec<Fragment>>,
    /// Fragments removed between begin and end version.
    pub removed_fragments: Option<Vec<Fragment>>,
    /// Fragments modified between begin and end version.
    pub modified_fragments: Option<Vec<Fragment>>,
    /// Schema changes between begin and end version, if any.
    pub schema_changes: Option<DeltaSchema>,
}

/// The data differences between two versions.
#[derive(Debug)]
pub struct DeltaData {
    /// Added record batches between begin and end version.
    /// Note: It collects all the physical added records
    /// include records causes by update operation.
    pub added_record_batches: Option<Vec<RecordBatch>>,
    /// Removed record batches between begin and end version.
    /// Note: It collects all the physical removed records
    /// include records caused by update operation.
    pub removed_record_batches: Option<Vec<RecordBatch>>,
}

/// The schema differences between two versions.
#[derive(Debug)]
pub struct DeltaSchema {
    /// Added fields between two versions.
    /// Note: It contains all the physical added fields
    /// include fields due to update operation, e.g. rename.
    pub added_fields: Option<Vec<Field>>,
    /// Removed fields between two versions.
    /// Note it contains all the physical removed fields
    /// include fields due to remove operation, e.g. rename.
    pub removed_fields: Option<Vec<Field>>,
    /// Modified fields between two versions.
    /// It mostly for cast type of field operation.
    pub modified_fields: Option<Vec<FieldChange>>,
}

/// Change in a specific field of the schema.
#[derive(Debug)]
pub struct FieldChange {
    pub name: String,
    pub old_type: DataType,
    pub new_type: DataType,
}

#[cfg(test)]
mod tests {

    use crate::dataset::{
        ColumnAlteration, Dataset, NewColumnTransform, UpdateBuilder, WriteMode, WriteParams,
    };
    use all_asserts::assert_true;
    use arrow_array::{
        Float32Array, Int32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray,
    };
    use arrow_schema::{DataType, Field, Field as ArrowField, Schema as ArrowSchema};
    use lance_encoding::version::LanceFileVersion;
    use std::ops::Range;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn create_test_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]))
    }

    async fn create_test_dataset(
        uri: &str,
        data: Range<i64>,
        max_rows_per_file: Option<usize>,
    ) -> Dataset {
        let schema = create_test_schema();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(data.clone())),
                Arc::new(StringArray::from_iter_values(
                    std::iter::repeat("foo").take(data.count()),
                )),
            ],
        )
        .unwrap();

        let write_params = if let Some(max_rows_per_file) = max_rows_per_file {
            Some(WriteParams {
                max_rows_per_file,
                ..Default::default()
            })
        } else {
            Some(WriteParams::default())
        };

        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            uri,
            write_params,
        )
        .await
        .unwrap()
    }

    #[lance_test_macros::test(tokio::test)]
    async fn test_simple_append_diff() {
        let test_dir = tempdir().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..20))],
        )
        .unwrap()];

        let test_uri = test_dir.path().to_str().unwrap();
        let mut write_params = WriteParams {
            max_rows_per_file: 20,
            data_storage_version: Some(LanceFileVersion::Stable),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        // v1, write initial batch
        Dataset::write(batches, test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(20..40))],
        )
        .unwrap()];
        write_params.mode = WriteMode::Append;
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        // v2, append a new batch
        Dataset::write(batches, test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        let actual_ds = Dataset::open(test_uri).await.unwrap();

        assert_eq!(actual_ds.version().version, 2);
        // diff meta v2 and v1
        let delta_meta = actual_ds.diff_metadata(1).await.unwrap();
        assert_eq!(delta_meta.added_fragments.unwrap().len(), 1);
        assert_eq!(delta_meta.modified_fragments.unwrap().len(), 0);
        assert_eq!(delta_meta.removed_fragments.unwrap().len(), 0);

        // diff data v2 and v1
        let delta_ds = actual_ds.diff(1).await.unwrap();
        let added_record_batches = delta_ds.added_record_batches.unwrap();
        assert_eq!(added_record_batches.len(), 1);
        assert_eq!(added_record_batches.get(0).unwrap().num_rows(), 20);

        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(40..65))],
        )
        .unwrap()];
        write_params.mode = WriteMode::Append;
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        // v3, append a new batch
        Dataset::write(batches, test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        let actual_ds = Dataset::open(test_uri).await.unwrap();
        let delta_meta = actual_ds.diff_metadata(1).await.unwrap();

        // diff meta v3 and v1
        assert_eq!(delta_meta.added_fragments.unwrap().len(), 3);
        assert_eq!(delta_meta.modified_fragments.unwrap().len(), 0);
        assert_eq!(delta_meta.removed_fragments.unwrap().len(), 0);

        // diff data v2 and v1
        let delta_ds = actual_ds.diff(1).await.unwrap();
        let added_record_batches = delta_ds.added_record_batches.unwrap();
        assert_eq!(added_record_batches.len(), 3);
        let total_added_rows: usize = added_record_batches.iter().map(|rb| rb.num_rows()).sum();
        assert_eq!(total_added_rows, 45);
    }

    #[lance_test_macros::test(tokio::test)]
    async fn test_simple_update_diff_inner_fragment() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let ds = create_test_dataset(test_uri, 0..30, Some(10)).await;

        let update_result = UpdateBuilder::new(Arc::new(ds))
            .update_where("id = 5")
            .unwrap()
            .set("name", "'bar' || cast(id as string)")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        let dataset = Arc::try_unwrap(update_result.new_dataset).unwrap();

        // diff meta v2 and v1
        let delta_meta = dataset.diff_metadata(1).await.unwrap();
        assert_eq!(delta_meta.added_fragments.unwrap().len(), 1);
        assert_eq!(delta_meta.modified_fragments.unwrap().len(), 1);
        assert_eq!(delta_meta.removed_fragments.unwrap().len(), 0);

        // diff data v2 and v1
        let delta_ds = dataset.diff(1).await.unwrap();
        let added_record_batches = delta_ds.added_record_batches.unwrap();
        let deleted_record_batches = delta_ds.removed_record_batches.unwrap();
        assert_eq!(added_record_batches.len(), 1);
        assert_eq!(deleted_record_batches.len(), 1);
        assert_eq!(added_record_batches.get(0).unwrap().num_rows(), 1);
        let total_removed_rows: usize = deleted_record_batches.iter().map(|rb| rb.num_rows()).sum();
        assert_eq!(total_removed_rows, 1);

        let update_result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 26")
            .unwrap()
            .set("name", "'bar' || cast(id as string)")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        let dataset = Arc::try_unwrap(update_result.new_dataset).unwrap();

        // diff meta v2 and v3
        let delta_meta = dataset.diff_metadata(2).await.unwrap();
        assert_eq!(delta_meta.added_fragments.unwrap().len(), 1);
        assert_eq!(delta_meta.modified_fragments.unwrap().len(), 1);
        assert_eq!(delta_meta.removed_fragments.unwrap().len(), 0);

        // diff data v3 and v2
        let delta_ds = dataset.diff(2).await.unwrap();
        let added_record_batches = delta_ds.added_record_batches.unwrap();
        let deleted_record_batches = delta_ds.removed_record_batches.unwrap();
        assert_eq!(added_record_batches.len(), 1);
        assert_eq!(deleted_record_batches.len(), 1);
        let total_added_rows: usize = added_record_batches.iter().map(|rb| rb.num_rows()).sum();
        assert_eq!(total_added_rows, 1);
        let total_removed_rows: usize = deleted_record_batches.iter().map(|rb| rb.num_rows()).sum();
        assert_eq!(total_removed_rows, 1);

        // v4, update one row
        let update_result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 2")
            .unwrap()
            .set("name", "'bar' || cast(id as string)")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        let dataset = Arc::try_unwrap(update_result.new_dataset).unwrap();
        // diff meta v3 and v4
        let delta_meta = dataset.diff_metadata(3).await.unwrap();
        assert_eq!(delta_meta.added_fragments.unwrap().len(), 1);
        assert_eq!(delta_meta.modified_fragments.unwrap().len(), 1);
        assert_eq!(delta_meta.removed_fragments.unwrap().len(), 0);

        // diff data v3 and v4
        let delta_ds = dataset.diff(3).await.unwrap();
        let added_record_batches = delta_ds.added_record_batches.unwrap();
        let deleted_record_batches = delta_ds.removed_record_batches.unwrap();
        assert_eq!(added_record_batches.len(), 1);
        assert_eq!(deleted_record_batches.len(), 1);
        let total_added_rows: usize = added_record_batches.iter().map(|rb| rb.num_rows()).sum();
        assert_eq!(total_added_rows, 1);
        let total_removed_rows: usize = deleted_record_batches.iter().map(|rb| rb.num_rows()).sum();
        assert_eq!(total_removed_rows, 1);
    }

    #[lance_test_macros::test(tokio::test)]
    async fn test_simple_update_diff_across_fragment() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let ds = create_test_dataset(test_uri, 0..30, Some(10)).await;

        let update_result = UpdateBuilder::new(Arc::new(ds))
            .update_where("id < 15")
            .unwrap()
            .set("name", "'bar' || cast(id as string)")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        let dataset = Arc::try_unwrap(update_result.new_dataset).unwrap();

        println!(
            "After updating dataset rows: {}",
            dataset.count_all_rows().await.unwrap()
        );

        assert_eq!(dataset.version().version, 2);
        // diff meta v2 and v1
        let delta_meta = dataset.diff_metadata(1).await.unwrap();
        assert_eq!(delta_meta.added_fragments.unwrap().len(), 1);
        assert_eq!(delta_meta.modified_fragments.unwrap().len(), 1);
        assert_eq!(delta_meta.removed_fragments.unwrap().len(), 1);

        // diff data v2 and v1
        let delta_ds = dataset.diff(1).await.unwrap();
        let added_record_batches = delta_ds.added_record_batches.unwrap();
        let deleted_record_batches = delta_ds.removed_record_batches.unwrap();
        assert_eq!(added_record_batches.len(), 1);
        assert_eq!(deleted_record_batches.len(), 2);
        assert_eq!(added_record_batches.get(0).unwrap().num_rows(), 15);
        let total_removed_rows: usize = deleted_record_batches.iter().map(|rb| rb.num_rows()).sum();
        assert_eq!(total_removed_rows, 15);
    }

    #[lance_test_macros::test(tokio::test)]
    async fn test_simple_delete_diff_inner_fragment() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let mut ds = create_test_dataset(test_uri, 0..30, Some(10)).await;

        ds.delete("id < 5").await.unwrap();

        assert_eq!(ds.version().version, 2);
        // diff meta v2 and v1
        let delta_meta = ds.diff_metadata(1).await.unwrap();
        assert_eq!(delta_meta.added_fragments.unwrap().len(), 0);
        assert_eq!(delta_meta.modified_fragments.unwrap().len(), 1);
        assert_eq!(delta_meta.removed_fragments.unwrap().len(), 0);

        // diff data v2 and v1
        let delta_ds = ds.diff(1).await.unwrap();
        let added_record_batches = delta_ds.added_record_batches.unwrap();
        let deleted_record_batches = delta_ds.removed_record_batches.unwrap();
        assert_eq!(added_record_batches.len(), 0);
        assert_eq!(deleted_record_batches.len(), 1);
        let total_removed_rows: usize = deleted_record_batches.iter().map(|rb| rb.num_rows()).sum();
        assert_eq!(total_removed_rows, 5);
    }

    #[lance_test_macros::test(tokio::test)]
    async fn test_simple_delete_diff_across_fragment() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let mut ds = create_test_dataset(test_uri, 0..30, Some(10)).await;

        ds.delete("id < 10").await.unwrap();

        assert_eq!(ds.version().version, 2);
        // diff meta v2 and v1
        let delta_meta = ds.diff_metadata(1).await.unwrap();
        assert_eq!(delta_meta.added_fragments.unwrap().len(), 0);
        assert_eq!(delta_meta.modified_fragments.unwrap().len(), 0);
        assert_eq!(delta_meta.removed_fragments.unwrap().len(), 1);

        // diff data v2 and v1
        let delta_ds = ds.diff(1).await.unwrap();
        let added_record_batches = delta_ds.added_record_batches.unwrap();
        let deleted_record_batches = delta_ds.removed_record_batches.unwrap();
        assert_eq!(added_record_batches.len(), 0);
        assert_eq!(deleted_record_batches.len(), 1);
        let total_removed_rows: usize = deleted_record_batches.iter().map(|rb| rb.num_rows()).sum();
        assert_eq!(total_removed_rows, 10);
    }

    #[lance_test_macros::test(tokio::test)]
    async fn test_mixed_operation() {
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

        let mut write_params = WriteParams {
            max_rows_per_file: 10,
            ..Default::default()
        };

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let batches = RecordBatchIterator::new([Ok(batch)], schema.clone());

        // v1, write initial batch, 3 fragments
        let ds = Dataset::write(batches, test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        // v2, update all rows => added 1 fragment, deleted 3 fragments
        let update_result = UpdateBuilder::new(Arc::new(ds))
            .set("name", "'bar' || cast(id as string)")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        let mut dataset = Arc::try_unwrap(update_result.new_dataset).unwrap();

        // v3, delete some rows => modified 1 fragment
        dataset.delete("id < 8").await.unwrap();

        // v4, write a new batch => added 3 fragments
        let new_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(30..60)),
                Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                    "foo", 30,
                ))),
            ],
        )
        .unwrap();

        write_params = WriteParams {
            max_rows_per_file: 10,
            mode: WriteMode::Append,
            ..Default::default()
        };
        let new_batches = RecordBatchIterator::new([Ok(new_batch)], schema.clone());
        let ds = Dataset::write(new_batches, test_uri, Some(write_params))
            .await
            .unwrap();

        // diff meta v4 and v1
        let delta_meta = ds.diff_metadata(1).await.unwrap();
        assert_eq!(delta_meta.added_fragments.unwrap().len(), 4);
        assert_eq!(delta_meta.modified_fragments.unwrap().len(), 0);
        assert_eq!(delta_meta.removed_fragments.unwrap().len(), 3);

        // diff data v4 and v1
        let delta_ds = ds.diff(1).await.unwrap();
        let added_record_batches = delta_ds.added_record_batches.unwrap();
        let deleted_record_batches = delta_ds.removed_record_batches.unwrap();
        assert_eq!(added_record_batches.len(), 4);
        assert_eq!(deleted_record_batches.len(), 4);
        let total_added_rows: usize = added_record_batches.iter().map(|rb| rb.num_rows()).sum();
        assert_eq!(total_added_rows, 52);
        let total_removed_rows: usize = deleted_record_batches.iter().map(|rb| rb.num_rows()).sum();
        assert_eq!(total_removed_rows, 38);
    }

    #[lance_test_macros::test(tokio::test)]
    async fn test_diff_symmetric_between_versions() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        // v1
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..10))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch1)], schema.clone());
        Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::Stable),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // v2: append
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(10..20))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch2)], schema.clone());
        Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                data_storage_version: Some(LanceFileVersion::Stable),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // v3: append
        let batch3 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(20..30))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch3)], schema.clone());
        Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                data_storage_version: Some(LanceFileVersion::Stable),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // v3 diff v2
        let ds_v3 = Dataset::open(test_uri).await.unwrap();
        assert_eq!(ds_v3.version().version, 3);
        let meta_v3_v2 = ds_v3.diff_metadata(2).await.unwrap();

        // v2 diff v3
        let ds_v2 = ds_v3.checkout_version(2).await.unwrap();
        assert_eq!(ds_v2.version().version, 2);
        let meta_v2_v3 = ds_v2.diff_metadata(3).await.unwrap();

        assert_eq!(meta_v3_v2.added_fragments, meta_v2_v3.added_fragments);
        assert_eq!(meta_v3_v2.removed_fragments, meta_v2_v3.removed_fragments);
        assert_eq!(meta_v3_v2.modified_fragments, meta_v2_v3.modified_fragments);
    }

    #[lance_test_macros::test(tokio::test)]
    async fn test_delete_modifies_fragment() {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int64,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(0..10))],
        )
        .unwrap();

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let mut ds = Dataset::write(batches, test_uri, None).await.unwrap();

        ds.delete("id < 5").await.unwrap();

        let delta_meta = ds.diff_metadata(1).await.unwrap();
        assert_eq!(delta_meta.removed_fragments.unwrap().len(), 0);
        assert_eq!(delta_meta.added_fragments.unwrap().len(), 0);
        assert_eq!(delta_meta.modified_fragments.unwrap().len(), 1);

        let delta = ds.diff(1).await.unwrap();
        assert_eq!(delta.added_record_batches.unwrap().len(), 0);
        assert_eq!(
            delta
                .removed_record_batches
                .unwrap()
                .iter()
                .map(|b| b.num_rows())
                .sum::<usize>(),
            5
        );
    }

    #[lance_test_macros::test(tokio::test)]
    async fn test_new_fragment_with_deletions() {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int64,
            false,
        )]));
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        Dataset::write(
            RecordBatchIterator::new(vec![], schema.clone()),
            test_uri,
            None,
        )
        .await
        .unwrap();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(0..10))],
        )
        .unwrap();
        let mut new_ds = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        new_ds.delete("id < 3").await.unwrap();

        let delta_meta = new_ds.diff_metadata(1).await.unwrap();
        assert_eq!(delta_meta.added_fragments.unwrap().len(), 1);
        assert_eq!(delta_meta.modified_fragments.unwrap().len(), 0); // 新增的 Fragment 被修改
        assert_eq!(delta_meta.removed_fragments.unwrap().len(), 0);
    }

    #[lance_test_macros::test(tokio::test)]
    async fn test_mixed_operations_across_versions() {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int64,
            false,
        )]));
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let batch1 =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from_iter(0..10))])
                .unwrap();
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter(10..20))],
        )
        .unwrap();
        let mut ds = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch1), Ok(batch2)], schema.clone()),
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 10,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        ds.delete("id < 5").await.unwrap();

        let batch3 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter(20..30))],
        )
        .unwrap();
        let ds_v3 = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch3)], schema.clone()),
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let delta_meta = ds_v3.diff_metadata(1).await.unwrap();
        assert_eq!(delta_meta.added_fragments.unwrap().len(), 1);
        assert_eq!(delta_meta.modified_fragments.unwrap().len(), 1); // 两个原 Fragment 被修改
        assert_eq!(delta_meta.removed_fragments.unwrap().len(), 0);
    }

    #[lance_test_macros::test(tokio::test)]
    async fn test_add_column_diff() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        // v1: only a field
        let schema_v1 = Arc::new(ArrowSchema::new(vec![Field::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let batch_v1 = RecordBatch::try_new(
            schema_v1.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..10))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch_v1)], schema_v1.clone());
        let mut dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::Stable),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // v2: add new column "value"
        let _ = dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("value".into(), "2 * random()".into())]),
                None,
                None,
            )
            .await;

        let ds = Dataset::open(test_uri).await.unwrap();
        let delta_meta = ds.diff_metadata(1).await.unwrap();

        assert!(delta_meta.schema_changes.is_some());
        let schema_diff = delta_meta.schema_changes.unwrap();
        let added_fields: Vec<lance_core::datatypes::Field> = schema_diff.added_fields.unwrap();
        assert_eq!(added_fields.len(), 1);
        assert_eq!(added_fields[0].name, "value");
        assert_true!(schema_diff.removed_fields.is_none());
        assert_true!(schema_diff.modified_fields.is_none());
    }

    #[lance_test_macros::test(tokio::test)]
    async fn test_rename_column_diff() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let schema_v1 = Arc::new(ArrowSchema::new(vec![Field::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let batch_v1 = RecordBatch::try_new(
            schema_v1.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..10))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch_v1)], schema_v1.clone());
        let mut dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::Stable),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        dataset
            .alter_columns(&[ColumnAlteration::new("a".into())
                .rename("b".into())
                .set_nullable(true)])
            .await
            .unwrap();

        let ds = Dataset::open(test_uri).await.unwrap();
        let delta_meta = ds.diff_metadata(1).await.unwrap();

        assert!(delta_meta.schema_changes.is_some());
        let schema_diff = delta_meta.schema_changes.unwrap();
        let added_fields: Vec<lance_core::datatypes::Field> = schema_diff.added_fields.unwrap();
        let removed_fields: Vec<lance_core::datatypes::Field> = schema_diff.removed_fields.unwrap();
        assert_eq!(added_fields.len(), 1);
        assert_eq!(added_fields[0].name, "b");
        assert_eq!(removed_fields.len(), 1);
        assert_eq!(removed_fields[0].name, "a");
        assert_true!(schema_diff.modified_fields.is_none());
    }

    #[lance_test_macros::test(tokio::test)]
    async fn test_cast_column_type_diff() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let schema_v1 = Arc::new(ArrowSchema::new(vec![Field::new(
            "a",
            DataType::Float32,
            false,
        )]));
        let batch_v1 = RecordBatch::try_new(
            schema_v1.clone(),
            vec![Arc::new(Float32Array::from_iter_values(
                (0u32..10).map(|v| v as f32),
            ))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch_v1)], schema_v1.clone());
        let mut dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::Stable),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        dataset
            .alter_columns(&[ColumnAlteration::new("a".into())
                .cast_to(DataType::Float64)
                .set_nullable(true)])
            .await
            .unwrap();

        let ds = Dataset::open(test_uri).await.unwrap();
        let delta_meta = ds.diff_metadata(1).await.unwrap();

        assert!(delta_meta.schema_changes.is_some());
        let schema_diff = delta_meta.schema_changes.unwrap();
        let modified_fields = schema_diff.modified_fields.unwrap();
        assert_true!(schema_diff.added_fields.is_none());
        assert_true!(schema_diff.removed_fields.is_none());
        assert_eq!(modified_fields.len(), 1);
        assert_eq!(modified_fields[0].old_type, DataType::Float32);
        assert_eq!(modified_fields[0].new_type, DataType::Float64);
    }

    #[lance_test_macros::test(tokio::test)]
    async fn test_drop_column_diff() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let schema_v1 = Arc::new(ArrowSchema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int64, false),
        ]));
        let batch_v1 = RecordBatch::try_new(
            schema_v1.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..10)),
                Arc::new(Int64Array::from_iter_values(10..20)),
            ],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch_v1)], schema_v1.clone());
        let mut dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::Stable),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let _ = dataset.drop_columns(&["b"]).await;

        let ds = Dataset::open(test_uri).await.unwrap();
        let delta_meta = ds.diff_metadata(1).await.unwrap();

        assert!(delta_meta.schema_changes.is_some());
        let schema_diff = delta_meta.schema_changes.unwrap();
        let removed_fields = schema_diff.removed_fields.unwrap();
        assert_true!(schema_diff.added_fields.is_none());
        assert_eq!(removed_fields.len(), 1);
        assert_eq!(removed_fields[0].name, "b");
        assert_true!(schema_diff.modified_fields.is_none());
    }
}
