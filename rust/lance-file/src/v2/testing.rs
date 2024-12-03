// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::ArrowError;
use futures::TryStreamExt;
use lance_core::{
    cache::{CapacityMode, FileMetadataCache},
    datatypes::Schema,
};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_io::{
    object_store::ObjectStore,
    scheduler::{ScanScheduler, SchedulerConfig},
    ReadBatchParams,
};
use object_store::path::Path;
use tempfile::TempDir;

use crate::v2::reader::{FileReader, FileReaderOptions};

use super::writer::{FileWriter, FileWriterOptions};

pub struct FsFixture {
    _tmp_dir: TempDir,
    pub tmp_path: Path,
    pub object_store: Arc<ObjectStore>,
    pub scheduler: Arc<ScanScheduler>,
}

impl Default for FsFixture {
    fn default() -> Self {
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path: String = tmp_dir.path().to_str().unwrap().to_owned();
        let tmp_path = Path::parse(tmp_path).unwrap();
        let tmp_path = tmp_path.child("some_file.lance");
        let object_store = Arc::new(ObjectStore::local());
        let scheduler =
            ScanScheduler::new(object_store.clone(), SchedulerConfig::default_for_testing());
        Self {
            _tmp_dir: tmp_dir,
            object_store,
            tmp_path,
            scheduler,
        }
    }
}

pub struct WrittenFile {
    pub schema: Arc<Schema>,
    pub data: Vec<RecordBatch>,
    pub field_id_mapping: Vec<(u32, u32)>,
}

pub async fn write_lance_file(
    data: impl RecordBatchReader,
    fs: &FsFixture,
    options: FileWriterOptions,
) -> WrittenFile {
    let writer = fs.object_store.create(&fs.tmp_path).await.unwrap();

    let lance_schema = lance_core::datatypes::Schema::try_from(data.schema().as_ref()).unwrap();

    let mut file_writer = FileWriter::try_new(writer, lance_schema.clone(), options).unwrap();

    let data = data
        .collect::<std::result::Result<Vec<_>, ArrowError>>()
        .unwrap();

    for batch in &data {
        file_writer.write_batch(batch).await.unwrap();
    }
    let field_id_mapping = file_writer.field_id_to_column_indices().to_vec();
    file_writer.add_schema_metadata("foo", "bar");
    file_writer.finish().await.unwrap();
    WrittenFile {
        schema: Arc::new(lance_schema),
        data,
        field_id_mapping,
    }
}

pub fn test_cache() -> Arc<FileMetadataCache> {
    Arc::new(FileMetadataCache::with_capacity(
        128 * 1024 * 1024,
        CapacityMode::Bytes,
    ))
}

pub async fn read_lance_file(
    fs: &FsFixture,
    decoder_middleware: Arc<DecoderPlugins>,
    filter: FilterExpression,
) -> Vec<RecordBatch> {
    let file_scheduler = fs.scheduler.open_file(&fs.tmp_path).await.unwrap();
    let file_reader = FileReader::try_open(
        file_scheduler,
        None,
        decoder_middleware,
        &test_cache(),
        FileReaderOptions::default(),
    )
    .await
    .unwrap();

    let schema = file_reader.schema();
    assert_eq!(schema.metadata.get("foo").unwrap(), "bar");

    let batch_stream = file_reader
        .read_stream(ReadBatchParams::RangeFull, 1024, 16, filter)
        .unwrap();

    batch_stream.try_collect().await.unwrap()
}

pub async fn count_lance_file(
    fs: &FsFixture,
    decoder_middleware: Arc<DecoderPlugins>,
    filter: FilterExpression,
) -> usize {
    read_lance_file(fs, decoder_middleware, filter)
        .await
        .iter()
        .map(|b| b.num_rows())
        .sum()
}
