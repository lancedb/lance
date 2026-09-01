// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::num::NonZeroU64;
use std::sync::Arc;

use lance_core::{Error, Result};
use lance_file::concat::{
    EncodedFileInput, FileConcatOptions, FileConcatReason, FileConcatResult, FileConcatTarget,
    concat_files,
};
use lance_file::version::ConcreteFileVersion;
use lance_file::versions as file_versions;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_table::format::{DataFile, Fragment};

use crate::Dataset;
use crate::dataset::WriteParams;
use crate::dataset::fragment::write::generate_random_filename;

/// Outcome of the compaction adapter around encoded-file concatenation.
pub enum BinaryCopyOutcome {
    Written(Vec<Fragment>),
    Unsupported(FileConcatReason),
}

async fn discard_outputs(dataset: &Dataset, paths: &[object_store::path::Path]) {
    for path in paths {
        if let Err(error) = dataset.object_store.delete(path).await {
            log::warn!(
                "failed to remove abandoned binary-copy output '{}': {}",
                path,
                error
            );
        }
    }
}

async fn open_input(
    dataset: &Dataset,
    data_file: &DataFile,
    expected_num_rows: u64,
) -> Result<EncodedFileInput> {
    let object_store = dataset.object_store_for_data_file(data_file).await?;
    let full_path = dataset
        .data_file_dir(data_file)?
        .join(data_file.path.as_str());
    let scan_scheduler = ScanScheduler::new(
        object_store.clone(),
        SchedulerConfig::max_bandwidth(&object_store),
    );
    let file_scheduler = scan_scheduler
        .open_file_with_priority(&full_path, 0, &data_file.file_size_bytes)
        .await?;
    Ok(EncodedFileInput::new(file_scheduler).with_expected_num_rows(expected_num_rows))
}

fn groups(fragments: &[Fragment], max_rows_per_file: u64) -> Result<Vec<Vec<(&DataFile, u64)>>> {
    let mut groups = Vec::new();
    let mut current = Vec::new();
    let mut current_rows = 0u64;
    for (fragment_index, fragment) in fragments.iter().enumerate() {
        let physical_rows = u64::try_from(fragment.physical_rows.ok_or_else(|| {
            Error::invalid_input(format!(
                "binary-copy fragment {} does not record physical_rows",
                fragment.id
            ))
        })?)
        .map_err(|_| {
            Error::invalid_input(format!(
                "binary-copy fragment {} physical row count does not fit in u64",
                fragment.id
            ))
        })?;
        let [data_file] = fragment.files.as_slice() else {
            return Err(Error::invalid_input(format!(
                "binary-copy fragment {} must contain exactly one complete data file, found {}",
                fragment.id,
                fragment.files.len()
            )));
        };
        current.push((data_file, physical_rows));
        current_rows = current_rows.checked_add(physical_rows).ok_or_else(|| {
            Error::invalid_input("binary-copy group physical row count overflows")
        })?;
        let remaining_fragments = fragments.len() - fragment_index - 1;
        if current.len() >= 2 && current_rows >= max_rows_per_file && remaining_fragments >= 2 {
            groups.push(std::mem::take(&mut current));
            current_rows = 0;
        }
    }
    if !current.is_empty() {
        groups.push(current);
    }
    Ok(groups)
}

/// Rewrite complete data files in fragment row order through [`concat_files`].
///
/// Fragment selection, deletion eligibility, row IDs, index remapping, and
/// transactions remain owned by the surrounding compaction flow. This adapter
/// only groups complete files and translates concat output into fragments.
pub async fn rewrite_files_binary_copy(
    version: ConcreteFileVersion,
    dataset: &Dataset,
    fragments: &[Fragment],
    params: &WriteParams,
    read_batch_bytes: Option<usize>,
) -> Result<BinaryCopyOutcome> {
    if fragments.is_empty() {
        return Err(Error::invalid_input(
            "binary copy requires at least one fragment",
        ));
    }
    if fragments.len() == 1 {
        return Err(Error::invalid_input(
            "binary copy requires at least two complete input files so compaction owns every output file",
        ));
    }
    if params.max_rows_per_file == 0 {
        return Err(Error::invalid_input(
            "binary copy max_rows_per_file must be greater than zero",
        ));
    }

    let expected_mapping = file_versions::data_file_columns(version, dataset.schema());
    for fragment in fragments {
        if fragment.deletion_file.is_some() {
            return Err(Error::invalid_input(format!(
                "binary-copy fragment {} has a deletion file",
                fragment.id
            )));
        }
        if !fragment.overlays.is_empty() {
            return Err(Error::invalid_input(format!(
                "binary-copy fragment {} has {} data overlays",
                fragment.id,
                fragment.overlays.len()
            )));
        }
        let [data_file] = fragment.files.as_slice() else {
            return Err(Error::invalid_input(format!(
                "binary-copy fragment {} must contain exactly one complete data file, found {}",
                fragment.id,
                fragment.files.len()
            )));
        };
        if data_file.fields.as_ref() != expected_mapping.0.as_slice()
            || data_file.column_indices.as_ref() != expected_mapping.1.as_slice()
        {
            return Err(Error::invalid_input(format!(
                "binary-copy fragment {} data file '{}' does not cover the complete dataset schema",
                fragment.id, data_file.path
            )));
        }
    }

    let target = FileConcatTarget::new(version, Arc::new(dataset.schema().clone()));
    let options = FileConcatOptions {
        read_batch_bytes: read_batch_bytes.unwrap_or(16 * 1024 * 1024),
        ..Default::default()
    };
    let groups = groups(fragments, params.max_rows_per_file as u64)?;
    let mut output = Vec::with_capacity(groups.len());
    let mut written_paths = Vec::new();

    for group in groups {
        let mut inputs = Vec::with_capacity(group.len());
        for (data_file, physical_rows) in &group {
            match open_input(dataset, data_file, *physical_rows).await {
                Ok(input) => inputs.push(input),
                Err(error) => {
                    discard_outputs(dataset, &written_paths).await;
                    return Err(error);
                }
            }
        }

        let filename = format!("{}.lance", generate_random_filename());
        let path = dataset.data_dir().join(filename.as_str());
        let object_store = dataset.object_store.clone();
        let result = concat_files(
            &target,
            &inputs,
            move || async move { object_store.create(&path).await },
            options.clone(),
        )
        .await;
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                discard_outputs(dataset, &written_paths).await;
                return Err(error);
            }
        };

        let (data_file, num_rows) = match result {
            FileConcatResult::Written(summary) => {
                written_paths.push(dataset.data_dir().join(filename.as_str()));
                let (fields, column_indices) =
                    file_versions::data_file_columns(version, dataset.schema());
                (
                    DataFile::new(
                        filename,
                        fields,
                        column_indices,
                        version,
                        NonZeroU64::new(summary.size_bytes),
                        None,
                    ),
                    summary.num_rows,
                )
            }
            FileConcatResult::Reused(_, _) => {
                discard_outputs(dataset, &written_paths).await;
                return Err(Error::internal(
                    "binary-copy grouping produced a reused input instead of an owned output file",
                ));
            }
            FileConcatResult::Unsupported(reason) => {
                discard_outputs(dataset, &written_paths).await;
                return Ok(BinaryCopyOutcome::Unsupported(reason));
            }
        };

        let mut fragment = Fragment::new(0);
        fragment.files.push(data_file);
        fragment.physical_rows = Some(match usize::try_from(num_rows) {
            Ok(num_rows) => num_rows,
            Err(_) => {
                discard_outputs(dataset, &written_paths).await;
                return Err(Error::invalid_input(format!(
                    "binary-copy output row count {num_rows} does not fit on this platform"
                )));
            }
        });
        output.push(fragment);
    }

    Ok(BinaryCopyOutcome::Written(output))
}
