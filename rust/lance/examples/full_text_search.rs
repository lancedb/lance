// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark of HNSW graph.
//!
//!
#![allow(clippy::print_stdout)]
use std::collections::HashSet;
use std::sync::Arc;

use all_asserts::assert_gt;
use arrow::array::AsArray;
use arrow::datatypes::UInt64Type;
use arrow_array::{LargeStringArray, RecordBatch, RecordBatchIterator, UInt64Array};
use itertools::Itertools;
use lance::Dataset;
use lance_core::ROW_ID;
use lance_index::scalar::inverted::flat_full_text_search;
use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams};
use lance_index::DatasetIndexExt;
use object_store::path::Path;

#[tokio::main]
async fn main() {
    env_logger::init();
    const TOTAL: usize = 10_000_000;
    let tempdir = tempfile::tempdir().unwrap();
    let dataset_dir = Path::from_filesystem_path(tempdir.path()).unwrap();
    let tokens = (0..10_000)
        .map(|_| random_word::gen(random_word::Lang::En))
        .collect_vec();
    let create_index = true;
    if create_index {
        let row_id_col = Arc::new(UInt64Array::from(
            (0..TOTAL).map(|i| i as u64).collect_vec(),
        ));
        let docs = (0..TOTAL)
            .map(|_| {
                let num_words = rand::random::<usize>() % 100 + 1;
                let doc = (0..num_words)
                    .map(|_| tokens[rand::random::<usize>() % tokens.len()])
                    .collect::<Vec<_>>();
                doc.join(" ")
            })
            .collect_vec();
        let doc_col = Arc::new(LargeStringArray::from(docs));
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![
                arrow_schema::Field::new("doc", arrow_schema::DataType::LargeUtf8, false),
                arrow_schema::Field::new(ROW_ID, arrow_schema::DataType::UInt64, false),
            ])
            .into(),
            vec![doc_col.clone(), row_id_col.clone()],
        )
        .unwrap();

        let batches = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
        let mut dataset = Dataset::write(batches, dataset_dir.as_ref(), None)
            .await
            .unwrap();
        let params = InvertedIndexParams::default();
        let start = std::time::Instant::now();
        dataset
            .create_index(
                &["doc"],
                lance_index::IndexType::Inverted,
                None,
                &params,
                true,
            )
            .await
            .unwrap();
        println!("create_index: {:?}", start.elapsed());
    }

    let dataset = Dataset::open(dataset_dir.as_ref()).await.unwrap();
    let query = tokens[0];
    let query = FullTextSearchQuery::new(query.to_owned()).limit(Some(10));
    println!("query: {:?}", query);
    let batch = dataset
        .scan()
        .full_text_search(query.clone())
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let index_results = batch[ROW_ID]
        .as_primitive::<UInt64Type>()
        .iter()
        .map(|v| v.unwrap())
        .collect::<HashSet<_>>();

    let start = std::time::Instant::now();
    dataset
        .scan()
        .full_text_search(query.clone())
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    println!("full_text_search: {:?}", start.elapsed());

    let batch = dataset
        .scan()
        .project(&["doc"])
        .unwrap()
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();
    let flat_results = flat_full_text_search(&[&batch], "doc", &query.query, None)
        .unwrap()
        .into_iter()
        .collect::<HashSet<_>>();
    assert_gt!(index_results.len(), 0);
    assert_eq!(index_results, flat_results);
}
