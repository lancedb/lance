// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

pub async fn merge_index_files(
    object_store: &ObjectStore,
    index_dir: &Path,
    store: Arc<dyn IndexStore>,
    progress: Arc<dyn IndexBuildProgress>,
) -> Result<()> {
    let metadata_path = index_dir.clone().join(METADATA_FILE);
    if object_store.exists(&metadata_path).await? {
        return Ok(());
    }

    // List all staged partition metadata files in the index directory
    let index_files = list_index_files(object_store, index_dir).await?;
    let part_metadata_files = metadata_files(&index_files);
    if part_metadata_files.is_empty() {
        return Err(Error::invalid_input_source(
            format!(
                "No partition metadata files found in index directory: {}",
                index_dir
            )
            .into(),
        ));
    }

    // Call merge_metadata_files function for inverted index
    merge_metadata_files(store, &part_metadata_files, progress).await
}

pub(super) async fn list_index_files(
    object_store: &ObjectStore,
    index_dir: &Path,
) -> Result<Vec<String>> {
    let mut index_files = Vec::new();
    let mut list_stream = object_store.read_dir_all(index_dir, None);

    while let Some(item) = list_stream.next().await {
        match item {
            Ok(meta) => {
                let location = meta.location.as_ref().trim_start_matches('/');
                let index_dir = index_dir.as_ref().trim_start_matches('/');
                let relative_path = location
                    .strip_prefix(index_dir)
                    .map(|s| s.trim_start_matches('/').to_string())
                    .unwrap_or_else(|| meta.location.filename().unwrap_or("").to_string());
                index_files.push(relative_path);
            }
            Err(err) => return Err(err),
        }
    }

    Ok(index_files)
}

fn metadata_files(index_files: &[String]) -> Vec<String> {
    index_files
        .iter()
        .filter(|file_name| {
            file_name.starts_with(&format!("{}/part_", STAGED_PARTITION_DIR))
                && file_name.ends_with("_metadata.lance")
        })
        .cloned()
        .collect()
}

#[cfg(test)]
pub(super) async fn list_metadata_files(
    object_store: &ObjectStore,
    index_dir: &Path,
) -> Result<Vec<String>> {
    let index_files = list_index_files(object_store, index_dir).await?;
    let part_metadata_files = metadata_files(&index_files);
    if part_metadata_files.is_empty() {
        return Err(Error::invalid_input_source(
            format!(
                "No partition metadata files found in index directory: {}",
                index_dir
            )
            .into(),
        ));
    }

    Ok(part_metadata_files)
}

/// Merge partition metadata files with partition ID remapping to sequential IDs starting from 0
async fn merge_metadata_files(
    store: Arc<dyn IndexStore>,
    part_metadata_files: &[String],
    progress: Arc<dyn IndexBuildProgress>,
) -> Result<()> {
    // Collect all partition IDs and params
    let mut all_partitions = Vec::new();
    let mut params = None;
    let mut token_set_format = None;
    let mut format_version = None;
    let mut deleted_fragments = RoaringBitmap::new();
    progress
        .stage_start(
            "read_partition_metadata",
            Some(part_metadata_files.len() as u64),
            "files",
        )
        .await?;

    for (idx, file_name) in part_metadata_files.iter().enumerate() {
        let reader = store.open_index_file(file_name).await?;
        let metadata = &reader.schema().metadata;

        let partitions_str = metadata.get("partitions").ok_or(Error::index(format!(
            "partitions not found in {}",
            file_name
        )))?;

        let partition_ids: Vec<u64> = serde_json::from_str(partitions_str)
            .map_err(|e| Error::index(format!("Failed to parse partitions: {}", e)))?;

        all_partitions.extend(partition_ids);

        if params.is_none() {
            let params_str = metadata
                .get("params")
                .ok_or(Error::index(format!("params not found in {}", file_name)))?;
            params = Some(
                serde_json::from_str::<InvertedIndexParams>(params_str)
                    .map_err(|e| Error::index(format!("Failed to parse params: {}", e)))?,
            );
        }

        if token_set_format.is_none()
            && let Some(name) = metadata.get(TOKEN_SET_FORMAT_KEY)
        {
            token_set_format = Some(TokenSetFormat::from_str(name)?);
        }
        if format_version.is_none() {
            format_version = Some(parse_format_version_from_metadata(metadata)?);
        }

        if reader.num_rows() > 0 {
            let metadata_batch = reader.read_range(0..1, None).await?;
            let deleted_fragments_col = metadata_batch
                .column_by_name(DELETED_FRAGMENTS_COL)
                .expect_ok()?;
            let deleted_fragments_arr = deleted_fragments_col
                .as_any()
                .downcast_ref::<BinaryArray>()
                .expect_ok()?;
            let part_deleted_fragments =
                RoaringBitmap::deserialize_from(deleted_fragments_arr.value(0))?;
            deleted_fragments.extend(part_deleted_fragments);
        }
        progress
            .stage_progress("read_partition_metadata", idx as u64 + 1)
            .await?;
    }
    progress.stage_complete("read_partition_metadata").await?;

    let mut sorted_ids = all_partitions;
    sorted_ids.sort();
    sorted_ids.dedup();

    let params = params.unwrap_or_default();
    let token_set_format = token_set_format.unwrap_or(TokenSetFormat::Arrow);
    let format_version = format_version.unwrap_or(InvertedListFormatVersion::V1);
    let remapped_partitions = finalize_partitions(
        store.clone(),
        &sorted_ids,
        &params,
        token_set_format,
        format_version,
        progress.clone(),
    )
    .await?;
    let builder = InvertedIndexBuilder::from_existing_index(
        params,
        None,
        remapped_partitions.clone(),
        token_set_format,
        None,
        deleted_fragments,
    )
    .with_format_version(format_version);
    progress
        .stage_start("write_merged_metadata", Some(1), "files")
        .await?;
    builder
        .write_metadata(&*store, &remapped_partitions)
        .await?;
    progress.stage_progress("write_merged_metadata", 1).await?;
    progress.stage_complete("write_merged_metadata").await?;

    // Cleanup staged partition metadata files
    for file_name in part_metadata_files {
        let _ = store.delete_index_file(file_name).await;
    }
    for &old_id in &sorted_ids {
        for suffix in PARTITION_FILE_SUFFIXES {
            let _ = store
                .delete_index_file(&staged_partition_file_path(old_id, suffix))
                .await;
        }
    }

    Ok(())
}

const MAX_FINALIZATION_CONCURRENCY: usize = 16;
const MAX_MERGE_GROUP_PARTITIONS: usize = 8;
// Stored size selects consolidation candidates; decoded groups still obey the builder budget.
const PARTITION_COPY_SIZE_RATIO: u64 = 8;

struct PartitionJob {
    sources: Vec<u64>,
    first_output_id: u64,
}

fn plan_partitions(
    partitions: &[u64],
    sizes: &HashMap<String, u64>,
    copy_threshold: u64,
) -> Result<Vec<PartitionJob>> {
    let mut groups = Vec::with_capacity(partitions.len());
    let mut small = Vec::new();
    let mut small_size = 0u64;
    for &partition in partitions {
        let mut size = 0u64;
        for suffix in PARTITION_FILE_SUFFIXES {
            let path = staged_partition_file_path(partition, suffix);
            let file_size = sizes
                .get(&path)
                .ok_or_else(|| Error::index(format!("Missing staged partition file {path}")))?;
            size = size.checked_add(*file_size).ok_or_else(|| {
                Error::index(format!("File sizes overflow for partition {partition}"))
            })?;
        }
        if size >= copy_threshold {
            groups.push(vec![partition]);
        } else {
            if !small.is_empty()
                && (small_size.saturating_add(size) > copy_threshold.saturating_mul(2)
                    || small.len() == MAX_MERGE_GROUP_PARTITIONS)
            {
                groups.push(std::mem::take(&mut small));
                small_size = 0;
            }
            small.push(partition);
            small_size = small_size.saturating_add(size);
        }
    }
    if !small.is_empty() {
        groups.push(small);
    }
    let mut next_id = 0u64;
    groups
        .into_iter()
        .map(|sources| {
            let first_output_id = next_id;
            next_id = next_id
                .checked_add(sources.len() as u64)
                .ok_or_else(|| Error::index("Final FTS partition IDs overflow".to_string()))?;
            Ok(PartitionJob {
                sources,
                first_output_id,
            })
        })
        .collect()
}

async fn finalize_partitions(
    store: Arc<dyn IndexStore>,
    partitions: &[u64],
    params: &InvertedIndexParams,
    token_set_format: TokenSetFormat,
    format_version: InvertedListFormatVersion,
    progress: Arc<dyn IndexBuildProgress>,
) -> Result<Vec<u64>> {
    let num_workers = resolve_num_workers(params);
    let memory_limit = resolve_worker_memory_limit_bytes(params, num_workers);
    let jobs = if format_version == InvertedListFormatVersion::V3 {
        let sizes = store
            .list_files_with_sizes()
            .await?
            .into_iter()
            .map(|file| (file.path, file.size_bytes))
            .collect();
        plan_partitions(
            partitions,
            &sizes,
            (memory_limit / PARTITION_COPY_SIZE_RATIO).max(1),
        )?
    } else {
        partitions
            .iter()
            .enumerate()
            .map(|(output, &source)| PartitionJob {
                sources: vec![source],
                first_output_id: output as u64,
            })
            .collect()
    };
    let merge_permits = Arc::new(tokio::sync::Semaphore::new(
        num_workers.clamp(1, MAX_FINALIZATION_CONCURRENCY),
    ));
    let total = partitions.len() as u64 * PARTITION_FILE_SUFFIXES.len() as u64;
    progress
        .stage_start("remap_partition_files", Some(total), "files")
        .await?;
    let mut tasks = futures::stream::iter(jobs.into_iter().map(|job| {
        let store = store.clone();
        let merge_permits = merge_permits.clone();
        async move {
            let completed = job.sources.len() as u64 * PARTITION_FILE_SUFFIXES.len() as u64;
            let output = if job.sources.len() == 1 {
                for suffix in PARTITION_FILE_SUFFIXES {
                    store
                        .copy_index_file_to(
                            &staged_partition_file_path(job.sources[0], suffix),
                            &partition_file_path(job.first_output_id, suffix),
                            store.as_ref(),
                        )
                        .await?;
                }
                vec![job.first_output_id]
            } else {
                let _permit = merge_permits
                    .acquire()
                    .await
                    .map_err(|error| Error::internal(error.to_string()))?;
                merge_partition_group(store, job, token_set_format, memory_limit).await?
            };
            Result::Ok((output, completed))
        }
    }))
    .buffer_unordered(
        store
            .io_parallelism()
            .clamp(1, MAX_FINALIZATION_CONCURRENCY),
    );
    let mut completed = 0;
    let mut output = Vec::with_capacity(partitions.len());
    while let Some((partitions, files)) = tasks.try_next().await? {
        output.extend(partitions);
        completed += files;
        progress
            .stage_progress("remap_partition_files", completed)
            .await?;
    }
    output.sort_unstable();
    progress.stage_complete("remap_partition_files").await?;
    Ok(output)
}

async fn merge_partition_group(
    store: Arc<dyn IndexStore>,
    job: PartitionJob,
    token_set_format: TokenSetFormat,
    memory_limit: u64,
) -> Result<Vec<u64>> {
    let mut accumulated: Option<InnerBuilder> = None;
    let mut output = Vec::with_capacity(job.sources.len());
    let mut output_id = job.first_output_id;
    for source in job.sources {
        let partition = InvertedPartition::load_from_paths(
            store.clone(),
            source,
            None,
            &LanceCache::no_cache(),
            token_set_format,
            [
                staged_partition_file_path(source, TOKENS_FILE),
                staged_partition_file_path(source, INVERT_LIST_FILE),
                staged_partition_file_path(source, DOCS_FILE),
            ],
        )
        .await?;
        let next = partition.into_builder().await?;
        if next.is_empty() {
            continue;
        }
        accumulated = Some(match accumulated.take() {
            Some(mut previous) => {
                if previous.memory_size().saturating_add(next.memory_size()) >= memory_limit
                    || previous.docs.len().saturating_add(next.docs.len()) > u32::MAX as usize
                {
                    previous.set_id(output_id);
                    previous
                        .write_to(store.as_ref(), PartitionWriteTarget::Final)
                        .await?;
                    output.push(output_id);
                    output_id += 1;
                    next
                } else {
                    spawn_cpu(move || {
                        previous.merge_from(next)?;
                        Result::Ok(previous)
                    })
                    .await?
                }
            }
            None => next,
        });
    }
    if let Some(mut builder) = accumulated {
        builder.set_id(output_id);
        builder
            .write_to(store.as_ref(), PartitionWriteTarget::Final)
            .await?;
        output.push(output_id);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_partitions_are_copied_and_small_groups_reserve_disjoint_ids() {
        let partitions = [1, 2, 3, 4];
        let mut sizes = HashMap::new();
        for (partition, size) in [(1, 100), (2, 10), (3, 100), (4, 10)] {
            for suffix in PARTITION_FILE_SUFFIXES {
                sizes.insert(staged_partition_file_path(partition, suffix), size);
            }
        }
        let jobs = plan_partitions(&partitions, &sizes, 300).unwrap();
        assert_eq!(
            jobs.iter()
                .map(|job| (job.sources.clone(), job.first_output_id))
                .collect::<Vec<_>>(),
            vec![(vec![1], 0), (vec![3], 1), (vec![2, 4], 2)]
        );
        sizes.remove(&staged_partition_file_path(1, DOCS_FILE));
        let error = plan_partitions(&partitions, &sizes, 300).err().unwrap();
        assert!(matches!(error, Error::Index { .. }));
        assert!(error.to_string().contains("staging/part_1_docs.lance"));
    }
}
