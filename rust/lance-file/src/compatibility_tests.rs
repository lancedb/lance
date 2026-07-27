// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::StringDictionaryBuilder;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int8Type, Int32Type};
use arrow_array::{
    Array, ArrayRef, Int32Array, LargeBinaryArray, ListArray, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use lance_core::cache::LanceCache;
use lance_core::datatypes::Schema as LanceSchema;
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_io::ReadBatchParams;
use lance_io::traits::Writer;
use lance_io::utils::CachedFileSize;
use rstest::rstest;
use tokio::io::AsyncWriteExt;

use crate::reader::{FileReader, FileReaderOptions};
use crate::testing::FsFixture;
use crate::version::ConcreteFileVersion;
use crate::versions;
use crate::versions::v1::reader::FileReader as V1Reader;
use crate::versions::v1::writer::{
    FileWriter as V1Writer, FileWriterOptions as V1WriterOptions, NotSelfDescribing,
};
use crate::writer::FileWriterOptions;

fn compatibility_fixture_batch() -> RecordBatch {
    let row_count = 4097;
    let ids = Arc::new(Int32Array::from_iter_values(0..row_count)) as ArrayRef;
    let names = Arc::new(StringArray::from_iter((0..row_count).map(|index| {
        (index % 7 != 0).then(|| format!("value-{index:04}-deterministic-fixture"))
    }))) as ArrayRef;
    let items = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(
        (0..row_count).map(|index| {
            (index % 11 != 0).then(|| {
                vec![
                    Some(index),
                    (index % 5 != 0).then_some(index * 2),
                    Some(index * 3),
                ]
            })
        }),
    )) as ArrayRef;
    let mut categories = StringDictionaryBuilder::<Int8Type>::new();
    for index in 0..row_count {
        if index % 13 == 0 {
            categories.append_null();
        } else {
            categories
                .append(match index % 3 {
                    0 => "red",
                    1 => "green",
                    _ => "blue",
                })
                .unwrap();
        }
    }
    let categories = Arc::new(categories.finish()) as ArrayRef;
    let blobs = Arc::new(LargeBinaryArray::from_iter_values(
        (0..row_count).map(|index| format!("blob-{index:04}-deterministic-payload").into_bytes()),
    )) as ArrayRef;

    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true).with_metadata(HashMap::from([(
            "lance-encoding:compression".to_string(),
            "none".to_string(),
        )])),
        Field::new(
            "items",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        ),
        Field::new(
            "category",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            true,
        )
        .with_metadata(HashMap::from([(
            "lance-encoding:dict-values-compression".to_string(),
            "none".to_string(),
        )])),
        Field::new("blob", DataType::LargeBinary, true).with_metadata(HashMap::from([(
            "lance-encoding:blob".to_string(),
            "true".to_string(),
        )])),
    ]));
    RecordBatch::try_new(schema, vec![ids, names, items, categories, blobs]).unwrap()
}

fn stable_fixture(version: ConcreteFileVersion) -> &'static [u8] {
    match version {
        ConcreteFileVersion::V1 => include_bytes!("../test_data/exact_versions/v1.lance"),
        ConcreteFileVersion::V2_0 => {
            include_bytes!("../test_data/exact_versions/v2_0.lance")
        }
        ConcreteFileVersion::V2_1 => {
            include_bytes!("../test_data/exact_versions/v2_1.lance")
        }
        ConcreteFileVersion::V2_2 => {
            include_bytes!("../test_data/exact_versions/v2_2.lance")
        }
        ConcreteFileVersion::V2_3 => {
            unreachable!("v2.3 is unstable and has no compatibility fixture")
        }
    }
}

fn assert_blob_column_eq(actual: &dyn Array, expected: &dyn Array) {
    let actual = actual.as_binary::<i64>();
    let expected = expected.as_binary::<i64>();
    assert_eq!(actual.len(), expected.len());
    for index in 0..actual.len() {
        assert_eq!(
            actual.is_null(index),
            expected.is_null(index),
            "blob validity differs at row {index}"
        );
        if actual.is_valid(index) {
            assert_eq!(
                actual.value(index),
                expected.value(index),
                "blob payload differs at row {index}"
            );
        }
    }
}

fn assert_wire_bytes_equal(actual: &[u8], expected: &[u8]) {
    if let Some(offset) = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual != expected)
    {
        panic!(
            "wire fixture first differs at byte {offset}: actual={}, expected={}",
            actual[offset], expected[offset]
        );
    }
    assert_eq!(
        actual.len(),
        expected.len(),
        "wire fixture length changed after a common {}-byte prefix",
        actual.len().min(expected.len())
    );
}

async fn write_current_fixture(
    version: ConcreteFileVersion,
    batch: &RecordBatch,
    schema: &LanceSchema,
) -> Vec<u8> {
    let fs = FsFixture::default();
    let object_writer = fs.object_store.create(&fs.tmp_path).await.unwrap();
    let options = FileWriterOptions {
        data_cache_bytes: Some(1),
        max_page_bytes: Some(1024),
        ..Default::default()
    };
    let summary = match version {
        ConcreteFileVersion::V1 => unreachable!("v1 uses its manifest-backed writer"),
        ConcreteFileVersion::V2_0 => {
            let mut writer =
                versions::v2_0::create_writer(object_writer, schema.clone(), options).unwrap();
            for offset in (0..batch.num_rows()).step_by(1024) {
                let slice = batch.slice(offset, (batch.num_rows() - offset).min(1024));
                writer.write_batch(&slice).await.unwrap();
            }
            writer.finish().await.unwrap()
        }
        ConcreteFileVersion::V2_1 => {
            let mut writer =
                versions::v2_1::create_writer(object_writer, schema.clone(), options).unwrap();
            for offset in (0..batch.num_rows()).step_by(1024) {
                let slice = batch.slice(offset, (batch.num_rows() - offset).min(1024));
                writer.write_batch(&slice).await.unwrap();
            }
            writer.finish().await.unwrap()
        }
        ConcreteFileVersion::V2_2 => {
            let mut writer =
                versions::v2_2::create_writer(object_writer, schema.clone(), options).unwrap();
            for offset in (0..batch.num_rows()).step_by(1024) {
                let slice = batch.slice(offset, (batch.num_rows() - offset).min(1024));
                writer.write_batch(&slice).await.unwrap();
            }
            writer.finish().await.unwrap()
        }
        ConcreteFileVersion::V2_3 => {
            let mut writer =
                versions::v2_3::create_writer(object_writer, schema.clone(), options).unwrap();
            for offset in (0..batch.num_rows()).step_by(1024) {
                let slice = batch.slice(offset, (batch.num_rows() - offset).min(1024));
                writer.write_batch(&slice).await.unwrap();
            }
            writer.finish().await.unwrap()
        }
    };
    fs.object_store
        .open(&fs.tmp_path)
        .await
        .unwrap()
        .get_range(0..summary.size_bytes as usize)
        .await
        .unwrap()
        .to_vec()
}

async fn assert_current_reader_roundtrip(
    fixture: &[u8],
    version: ConcreteFileVersion,
    expected: &RecordBatch,
) {
    let fs = FsFixture::default();
    let mut fixture_writer = fs.object_store.create(&fs.tmp_path).await.unwrap();
    fixture_writer.write_all(fixture).await.unwrap();
    Writer::shutdown(fixture_writer.as_mut()).await.unwrap();
    let scheduler = fs
        .scheduler
        .open_file(&fs.tmp_path, &CachedFileSize::new(fixture.len() as u64))
        .await
        .unwrap();
    let reader = FileReader::try_open(
        scheduler,
        None,
        Arc::<DecoderPlugins>::default(),
        &LanceCache::no_cache(),
        FileReaderOptions::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        ConcreteFileVersion::from(reader.metadata().version()),
        version
    );
    assert!(
        reader
            .metadata()
            .column_metadatas
            .iter()
            .any(|metadata| metadata.pages.len() > 1)
    );
    let batches = reader
        .read_stream(
            ReadBatchParams::RangeFull,
            1024,
            16,
            FilterExpression::no_filter(),
        )
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(
        batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        expected.num_rows()
    );
    assert!(
        batches
            .iter()
            .all(|actual| actual.schema_ref() == expected.schema_ref())
    );
    let mut row_offset = 0;
    for actual in &batches {
        let expected = expected.slice(row_offset, actual.num_rows());
        assert_blob_column_eq(actual.column(4).as_ref(), expected.column(4).as_ref());
        row_offset += actual.num_rows();
    }
    assert_eq!(row_offset, expected.num_rows());
}

#[rstest]
#[case::v2_0(ConcreteFileVersion::V2_0)]
#[case::v2_1(ConcreteFileVersion::V2_1)]
#[case::v2_2(ConcreteFileVersion::V2_2)]
#[tokio::test]
async fn stable_current_writer_and_reader_are_wire_compatible(
    #[case] version: ConcreteFileVersion,
) {
    let batch = compatibility_fixture_batch();
    let mut schema = LanceSchema::try_from(batch.schema().as_ref()).unwrap();
    schema.set_dictionary(&batch).unwrap();

    let actual = write_current_fixture(version, &batch, &schema).await;
    let expected = stable_fixture(version);
    assert_wire_bytes_equal(&actual, expected);
    assert_current_reader_roundtrip(expected, version, &batch).await;
}

#[tokio::test]
async fn v2_3_output_is_deterministic_within_the_current_revision() {
    let batch = compatibility_fixture_batch();
    let mut schema = LanceSchema::try_from(batch.schema().as_ref()).unwrap();
    schema.set_dictionary(&batch).unwrap();

    let first = write_current_fixture(ConcreteFileVersion::V2_3, &batch, &schema).await;
    let second = write_current_fixture(ConcreteFileVersion::V2_3, &batch, &schema).await;
    assert_eq!(first, second);
    assert_eq!(
        &first[first.len() - 8..],
        &[2, 0, 3, 0, b'L', b'A', b'N', b'C']
    );
    assert_current_reader_roundtrip(&first, ConcreteFileVersion::V2_3, &batch).await;
}

#[tokio::test]
async fn v1_writer_and_reader_are_wire_compatible() {
    let expected = stable_fixture(ConcreteFileVersion::V1);
    let batch = compatibility_fixture_batch();
    let mut schema = LanceSchema::try_from(batch.schema().as_ref()).unwrap();
    schema.set_dictionary(&batch).unwrap();
    let fs = FsFixture::default();
    let mut writer = V1Writer::<NotSelfDescribing>::try_new(
        fs.object_store.as_ref(),
        &fs.tmp_path,
        schema.clone(),
        &V1WriterOptions {
            collect_stats_for_fields: Some(Vec::new()),
        },
    )
    .await
    .unwrap();
    for offset in (0..batch.num_rows()).step_by(1024) {
        let slice = batch.slice(offset, (batch.num_rows() - offset).min(1024));
        writer.write(std::slice::from_ref(&slice)).await.unwrap();
    }
    let summary = writer.finish().await.unwrap();
    let actual = fs
        .object_store
        .open(&fs.tmp_path)
        .await
        .unwrap()
        .get_range(0..summary.size_bytes as usize)
        .await
        .unwrap();
    assert_wire_bytes_equal(actual.as_ref(), expected);

    let fixture_fs = FsFixture::default();
    let mut fixture_writer = fixture_fs
        .object_store
        .create(&fixture_fs.tmp_path)
        .await
        .unwrap();
    fixture_writer.write_all(expected).await.unwrap();
    Writer::shutdown(fixture_writer.as_mut()).await.unwrap();
    let reader = V1Reader::try_new(
        fixture_fs.object_store.as_ref(),
        &fixture_fs.tmp_path,
        schema.clone(),
    )
    .await
    .unwrap();
    let actual_batch = reader
        .read_range(0..batch.num_rows(), &schema)
        .await
        .unwrap();
    assert_eq!(reader.num_batches(), 5);
    assert_eq!(actual_batch.num_rows(), batch.num_rows());
    assert_eq!(actual_batch.column(0).to_data(), batch.column(0).to_data());
    assert_eq!(actual_batch.column(1).to_data(), batch.column(1).to_data());
    assert_blob_column_eq(actual_batch.column(4).as_ref(), batch.column(4).as_ref());
}
