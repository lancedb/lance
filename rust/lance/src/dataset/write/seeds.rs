// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Write seed collection shared by every writer that produces a data file.
//!
//! A [`SeedCollector`] decides which columns of a data file get a seed, feeds
//! each written batch to the seed writers, and embeds the finished seeds in the
//! file before it is closed. Writers that create whole fragments and writers
//! that add or replace a column inside an existing fragment all go through it,
//! so a seeded column keeps its seed in whichever data file serves it.

use arrow_array::RecordBatch;
use lance_core::Result;
use lance_core::datatypes::Schema;
use lance_file::version::ConcreteFileVersion;
use lance_index::scalar::seed::IndexSeedWriter;

use arrow_schema::DataType;
use lance_index::scalar::zonemap::ZoneMapSeedWriter;

use super::GenericWriter;
use crate::Dataset;
use crate::index::scalar::{IndexDetails, fetch_index_details};
use crate::index::{index_is_usable, load_all_indices};

/// Zone size for seeds collected by default (no index asked for them).
/// Matches the zone map plugin's default, so a later index build from these
/// seeds sees the granularity it would have requested.
pub const DEFAULT_SEED_ROWS_PER_ZONE: u64 = 8192;

/// Whether a column of this type gets a seed when collection is on by
/// default. Mirrors the types whose statistics carry usable bounds; columns
/// outside it would pay the footer bytes for null-only records.
fn column_gets_default_seed(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_)
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::LargeBinary
    )
}

/// Collects write seeds for the columns of one data file.
#[derive(Debug, Default)]
pub struct SeedCollector {
    writers: Vec<Box<dyn IndexSeedWriter>>,
}

impl SeedCollector {
    /// A collector that writes no seeds.
    pub fn disabled() -> Self {
        Self::default()
    }

    /// Build the seed writers for the top-level columns of `write_schema`.
    ///
    /// A column gets a seed writer when an existing, usable index on it asks
    /// for one through its plugin (`consult_indices` is false for writes that
    /// discard the dataset's indices, where those seeds have no consumer), or
    /// when `collect_by_default` is set, for every supported column type: the
    /// seed then serves whatever index is created later, without rereading
    /// the column.
    pub async fn for_write(
        version: ConcreteFileVersion,
        dataset: Option<&Dataset>,
        write_schema: &Schema,
        consult_indices: bool,
        collect_by_default: bool,
    ) -> Result<Self> {
        // Legacy files have no global buffers to hold a seed.
        if version == ConcreteFileVersion::V1 {
            return Ok(Self::disabled());
        }
        let mut writers: Vec<Box<dyn IndexSeedWriter>> = Vec::new();
        if let (Some(dataset), true) = (dataset, consult_indices) {
            Self::index_driven_writers(dataset, write_schema, &mut writers).await?;
        }
        if collect_by_default {
            for field in write_schema.fields.iter() {
                if writers.iter().any(|w| w.column_name() == field.name) {
                    continue;
                }
                let data_type = field.data_type();
                if !column_gets_default_seed(&data_type) || field.is_blob_v2() {
                    continue;
                }
                writers.push(Box::new(ZoneMapSeedWriter::new(
                    &field.name,
                    DEFAULT_SEED_ROWS_PER_ZONE,
                    data_type,
                )?));
            }
        }
        Ok(Self { writers })
    }

    /// The index-driven writers: columns whose existing, usable index asks
    /// for a seed through its plugin.
    async fn index_driven_writers(
        dataset: &Dataset,
        write_schema: &Schema,
        writers: &mut Vec<Box<dyn IndexSeedWriter>>,
    ) -> Result<()> {
        let indices = load_all_indices(dataset).await?;
        for index in indices.iter().filter(|index| index_is_usable(index)) {
            // A covered index lists its carried columns in `fields` too; the
            // seed writer keys on the single keyed column. System indices
            // commit no fields at all, so this also skips them.
            let Some(field_id) = index.keyed_field() else {
                continue;
            };
            // Seeds are collected for top-level columns of the file being
            // written. Batches are observed by column name, which is the field
            // path for a top-level field.
            let Some(field) = write_schema.fields.iter().find(|f| f.id == field_id) else {
                continue;
            };
            let Ok(index_details) = fetch_index_details(dataset, &field.name, index).await else {
                continue;
            };
            let details = IndexDetails(index_details.clone());
            let Ok(plugin) = details.get_plugin() else {
                continue;
            };
            if let Some(writer) = plugin
                .create_seed_writer(&field.name, &field.data_type(), &index_details)
                .await?
            {
                writers.push(writer);
            }
        }
        Ok(())
    }

    /// Number of columns that will receive a seed.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.writers.len()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.writers.is_empty()
    }

    /// Whether any column of this write receives a seed.
    pub(crate) fn has_writers(&self) -> bool {
        !self.writers.is_empty()
    }

    /// Feed one written batch to every seed writer whose column it contains.
    pub fn observe(&mut self, batch: &RecordBatch) -> Result<()> {
        for writer in self.writers.iter_mut() {
            if let Some(column) = batch.column_by_name(writer.column_name()) {
                writer.observe_batch(column)?;
            }
        }
        Ok(())
    }

    /// Embed the finished seeds in `writer` before its `finish()` and reset
    /// the seed writers for the next data file.
    pub async fn flush(&mut self, writer: &mut dyn GenericWriter) -> Result<()> {
        for seed_writer in self.writers.iter_mut() {
            if let Some(bytes) = seed_writer.finish()? {
                let buf_index = writer.add_global_buffer(bytes).await?;
                let key = seed_writer.schema_metadata_key();
                let value = seed_writer.schema_metadata_value(buf_index);
                writer.add_schema_metadata(key, value);
            }
        }
        Ok(())
    }
}

/// Set the paired `FLAG_ZONE_WRITE_SEEDS` bits on the manifest being built
/// when this transaction publishes data files that carry version 1 seeds.
///
/// The write-path fact on the transaction is authoritative when present; a
/// transaction that lost it (serialized round trip, pre-written fragments
/// committed from outside the write path) falls back to reading the published
/// files, which no concurrent index change can race. When the built manifest
/// already inherited both sticky bits from its base, nothing needs checking.
pub async fn apply_v1_seed_feature_flags(
    manifest: &mut lance_table::format::Manifest,
    dataset: &Dataset,
    transaction: &crate::dataset::transaction::Transaction,
) -> Result<()> {
    use lance_table::feature_flags::FLAG_ZONE_WRITE_SEEDS;

    if manifest.reader_feature_flags & FLAG_ZONE_WRITE_SEEDS != 0
        && manifest.writer_feature_flags & FLAG_ZONE_WRITE_SEEDS != 0
    {
        return Ok(());
    }
    let publishes = match transaction.wrote_v1_seeds {
        Some(known) => known,
        None => transaction_publishes_v1_seeds(dataset, &transaction.operation).await?,
    };
    if publishes {
        manifest.reader_feature_flags |= FLAG_ZONE_WRITE_SEEDS;
        manifest.writer_feature_flags |= FLAG_ZONE_WRITE_SEEDS;
    }
    Ok(())
}

/// The data files an operation publishes: the files its commit makes visible
/// that were written for this transaction. Updated fragments are listed whole,
/// which over-approximates (they also carry pre-existing files); the caller
/// treats "carries a seed" as a one-way answer, so over-scanning stays safe.
/// Data overlays are excluded because no overlay write path attaches a seed
/// collector.
fn operation_published_data_files(
    operation: &crate::dataset::transaction::Operation,
) -> Vec<&lance_table::format::DataFile> {
    use crate::dataset::transaction::Operation;
    use lance_table::format::{DataFile, Fragment};

    fn fragment_files(fragments: &[Fragment]) -> impl Iterator<Item = &DataFile> {
        fragments.iter().flat_map(|fragment| fragment.files.iter())
    }

    match operation {
        Operation::Append { fragments }
        | Operation::Overwrite { fragments, .. }
        | Operation::Merge { fragments, .. } => fragment_files(fragments).collect(),
        Operation::Update {
            updated_fragments,
            new_fragments,
            ..
        } => fragment_files(updated_fragments)
            .chain(fragment_files(new_fragments))
            .collect(),
        Operation::Rewrite { groups, .. } => groups
            .iter()
            .flat_map(|group| fragment_files(&group.new_fragments))
            .collect(),
        Operation::DataReplacement { replacements } => replacements
            .iter()
            .map(|replacement| &replacement.1)
            .collect(),
        _ => Vec::new(),
    }
}

/// Whether any data file this transaction publishes carries a version 1 zone
/// write seed, read from the files themselves.
///
/// The commit must set the paired `FLAG_ZONE_WRITE_SEEDS` bits in the same
/// manifest that first references such files, so builds that cannot read
/// version 1 seeds refuse the dataset instead of misreading them. This is the
/// fallback for transactions whose [`Transaction::wrote_v1_seeds`] write-path
/// fact is unknown (rebuilt from a serialized transaction, or assembled
/// outside the lance write path from pre-written fragments). The files are
/// the durable record, so no concurrent index change can race this answer.
///
/// Short-circuits on the first version 1 seed. Concluding "no seeds" costs
/// one footer read per published file, paid only while the dataset does not
/// yet carry the sticky flags.
///
/// [`Transaction::wrote_v1_seeds`]: lance_table::transaction::Transaction::wrote_v1_seeds
pub async fn transaction_publishes_v1_seeds(
    dataset: &Dataset,
    operation: &crate::dataset::transaction::Operation,
) -> Result<bool> {
    for data_file in operation_published_data_files(operation) {
        if data_file_carries_v1_seed(dataset, data_file).await? {
            return Ok(true);
        }
    }
    Ok(false)
}

async fn data_file_carries_v1_seed(
    dataset: &Dataset,
    data_file: &lance_table::format::DataFile,
) -> Result<bool> {
    use lance_file::reader::{FileReader, FileReaderOptions};
    use lance_index::scalar::seed::SEED_META_KEY_PREFIX;
    use lance_index::scalar::zonemap::ZoneMapSeedWriter;
    use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
    use lance_io::utils::CachedFileSize;

    // Legacy files have no global buffers, so they cannot hold a seed.
    if data_file.file_version()? == ConcreteFileVersion::V1 {
        return Ok(false);
    }

    let path = dataset
        .data_file_dir(data_file)?
        .join(data_file.path.as_str());
    let store = dataset.object_store(data_file.base_id).await?;
    let scheduler = ScanScheduler::new(store.clone(), SchedulerConfig::max_bandwidth(&store));
    // A file that is not there carries no seed a reader could misread; commits
    // of fabricated fragments (tests, manifest tooling) stay possible. Other
    // errors propagate: a transient failure must not skip the flag.
    let file_scheduler = match scheduler.open_file(&path, &CachedFileSize::unknown()).await {
        Ok(file_scheduler) => file_scheduler,
        Err(lance_core::Error::NotFound { .. }) => return Ok(false),
        Err(e) => return Err(e),
    };
    let reader = FileReader::try_open(
        file_scheduler,
        None,
        Default::default(),
        &dataset.metadata_cache.file_metadata_cache(&path),
        FileReaderOptions::default(),
    )
    .await?;

    let file_schema = &reader.metadata().file_schema;
    for (key, value) in file_schema.metadata.iter() {
        if !key.starts_with(SEED_META_KEY_PREFIX) {
            continue;
        }
        // The buffer index is the portion before the first ':'. A value this
        // build cannot parse comes from a newer writer; claiming the flag for
        // it only over-fences old readers, while missing it would let them
        // misread the seed, so unparsable seeds count as version 1.
        let Some(buf_index) = value
            .split(':')
            .next()
            .and_then(|index| index.parse::<u32>().ok())
        else {
            return Ok(true);
        };
        let Ok(bytes) = reader.read_global_buffer(buf_index).await else {
            return Ok(true);
        };
        match ZoneMapSeedWriter::seed_bytes_format_version(&bytes) {
            // Version 0 buffers predate the version key and need no flag.
            Ok(None) => continue,
            Ok(Some(_)) => return Ok(true),
            Err(_) => return Ok(true),
        }
    }
    Ok(false)
}
