#![allow(clippy::redundant_pub_crate)]
pub(crate) use std::collections::{HashMap, HashSet};
pub(crate) use std::sync::Arc;
pub(crate) use std::vec;

pub(crate) use crate::dataset::builder::DatasetBuilder;
pub(crate) use crate::dataset::tests::dataset_migrations::scan_dataset;
pub(crate) use crate::dataset::tests::dataset_transactions::{assert_results, execute_sql};
pub(crate) use crate::dataset::{
    AutoCleanupParams,
    ManifestWriteConfig,
    ProjectionRequest,
    write_manifest_file,
};
pub(crate) use crate::{Dataset, Error, Result};
pub(crate) use crate::session::Session;
pub(crate) use crate::dataset::optimize::{compact_files, CompactionOptions};
pub(crate) use crate::dataset::transaction::{DataReplacementGroup, Operation, Transaction};
pub(crate) use crate::dataset::WriteMode::Overwrite;
pub(crate) use crate::dataset::ROW_ID;
pub(crate) use crate::datatypes::Schema;
pub(crate) use crate::io::ObjectStoreParams;
pub(crate) use lance_core::ROW_ADDR;
pub(crate) use lance_table::format::{DataStorageFormat, IndexMetadata};
pub(crate) use lance_table::io::commit::ManifestNamingScheme;
pub(crate) use crate::dataset::WriteDestination;
pub(crate) use crate::dataset::UpdateBuilder;
pub(crate) use crate::index::vector::VectorIndexParams;
pub(crate) use crate::utils::test::copy_test_data_to_tmp;
pub(crate) use lance_arrow::FixedSizeListArrayExt;
pub(crate) use mock_instant::thread_local::MockClock;

pub(crate) use crate::dataset::write::{CommitBuilder, InsertBuilder, WriteMode, WriteParams};
pub(crate) use arrow::array::{as_struct_array, AsArray, GenericListBuilder, GenericStringBuilder};
pub(crate) use arrow::compute::concat_batches;
pub(crate) use arrow::datatypes::UInt64Type;
pub(crate) use arrow_array::{
    builder::StringDictionaryBuilder,
    cast::as_string_array,
    types::{Float32Type, Int32Type},
    ArrayRef, DictionaryArray, Float32Array, Int32Array, Int64Array, Int8Array,
    Int8DictionaryArray, ListArray, RecordBatchIterator, StringArray, UInt16Array, UInt32Array,
};
pub(crate) use arrow_array::{
    Array, FixedSizeListArray, GenericStringArray, Int16Array, Int16DictionaryArray,
    LargeBinaryArray, StructArray, UInt64Array,
};
pub(crate) use arrow_array::RecordBatch;
pub(crate) use arrow_array::RecordBatchReader;
pub(crate) use arrow_ord::sort::sort_to_indices;
pub(crate) use arrow_schema::{
    DataType, Field as ArrowField, Field, Fields as ArrowFields, Schema as ArrowSchema,
};
pub(crate) use lance_arrow::bfloat16::{self, BFLOAT16_EXT_NAME};
pub(crate) use lance_arrow::{ARROW_EXT_META_KEY, ARROW_EXT_NAME_KEY, BLOB_META_KEY};
pub(crate) use lance_core::utils::tempfile::{TempDir, TempStdDir, TempStrDir};
pub(crate) use lance_datagen::{array, gen_batch, BatchCount, Dimension, RowCount};
pub(crate) use lance_file::version::LanceFileVersion;
pub(crate) use lance_file::writer::FileWriter;
pub(crate) use lance_index::scalar::inverted::{
    query::{BooleanQuery, MatchQuery, Occur, Operator, PhraseQuery},
    tokenizer::InvertedIndexParams,
};
pub(crate) use lance_index::DatasetIndexExt;
pub(crate) use lance_index::scalar::FullTextSearchQuery;
pub(crate) use lance_index::{scalar::ScalarIndexParams, vector::DIST_COL, IndexType};
pub(crate) use lance_io::assert_io_eq;
pub(crate) use lance_io::utils::CachedFileSize;
pub(crate) use lance_linalg::distance::MetricType;
pub(crate) use lance_table::feature_flags;
pub(crate) use lance_table::format::{DataFile, WriterVersion};

pub(crate) use crate::datafusion::LanceTableProvider;
pub(crate) use crate::dataset::refs::branch_contents_path;
pub(crate) use datafusion::common::{assert_contains, assert_not_contains};
pub(crate) use datafusion::prelude::SessionContext;
pub(crate) use lance_arrow::json::ARROW_JSON_EXT_NAME;
pub(crate) use lance_datafusion::datagen::DatafusionDatagenExt;
pub(crate) use lance_datafusion::udf::register_functions;
pub(crate) use lance_index::scalar::inverted::query::{FtsQuery, MultiMatchQuery};
pub(crate) use lance_testing::datagen::generate_random_array;
pub(crate) use itertools::Itertools;
pub(crate) use rand::seq::SliceRandom;
pub(crate) use rand::Rng;
pub(crate) use rstest::rstest;
pub(crate) use futures::{StreamExt, TryStreamExt};
pub(crate) use std::cmp::Ordering;
pub(crate) use object_store::path::Path;
pub(crate) use lance_table::io::manifest::read_manifest;

// Used to validate that futures returned are Send.
pub(crate) fn require_send<T: Send>(t: T) -> T {
    t
}

pub(crate) async fn create_file(
    path: &std::path::Path,
    mode: WriteMode,
    data_storage_version: LanceFileVersion,
) {
    let fields = vec![
        ArrowField::new("i", DataType::Int32, false),
        ArrowField::new(
            "dict",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
    ];
    let schema = Arc::new(ArrowSchema::new(fields));
    let dict_values = StringArray::from_iter_values(["a", "b", "c", "d", "e"]);
    let batches: Vec<RecordBatch> = (0..20)
        .map(|i| {
            let mut arrays =
                vec![Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20)) as ArrayRef];
            arrays.push(Arc::new(
                DictionaryArray::try_new(
                    UInt16Array::from_iter_values((0_u16..20_u16).map(|v| v % 5)),
                    Arc::new(dict_values.clone()),
                )
                .unwrap(),
            ));
            RecordBatch::try_new(schema.clone(), arrays).unwrap()
        })
        .collect();
    let expected_batches = batches.clone();

    let test_uri = path.to_str().unwrap();
    let write_params = WriteParams {
        max_rows_per_file: 40,
        max_rows_per_group: 10,
        mode,
        data_storage_version: Some(data_storage_version),
        ..WriteParams::default()
    };
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();

    let actual_ds = Dataset::open(test_uri).await.unwrap();
    assert_eq!(actual_ds.version().version, 1);
    assert_eq!(
        actual_ds.manifest.writer_version,
        Some(WriterVersion::default())
    );
    let actual_schema = ArrowSchema::from(actual_ds.schema());
    assert_eq!(&actual_schema, schema.as_ref());

    let actual_batches = actual_ds
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    // The batch size batches the group size.
    // (the v2 writer has no concept of group size)
    if data_storage_version == LanceFileVersion::Legacy {
        for batch in &actual_batches {
            assert_eq!(batch.num_rows(), 10);
        }
    }

    // sort
    let actual_batch = concat_batches(&schema, &actual_batches).unwrap();
    let idx_arr = actual_batch.column_by_name("i").unwrap();
    let sorted_indices = sort_to_indices(idx_arr, None, None).unwrap();
    let struct_arr: StructArray = actual_batch.into();
    let sorted_arr = arrow_select::take::take(&struct_arr, &sorted_indices, None).unwrap();

    let expected_struct_arr: StructArray =
        concat_batches(&schema, &expected_batches).unwrap().into();
    assert_eq!(&expected_struct_arr, as_struct_array(sorted_arr.as_ref()));

    // Each fragments has different fragment ID
    assert_eq!(
        actual_ds
            .fragments()
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        (0..10).collect::<Vec<_>>()
    )
}
