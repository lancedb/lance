// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::RecordBatch;
use lance::Dataset;

mod primitives;
mod vectors;

async fn test_scan(_original: &RecordBatch, _ds: &Dataset) {
    todo!("validate that if you scan ds, then sort by id, you get original back.")
}

async fn test_take(_original: &RecordBatch, _ds: &Dataset) {
    todo!("generate a few sets of ids and validate we can call take against the RB and the DS and get the same result.");
}

async fn test_filter(_original: &RecordBatch, _ds: &Dataset, _predicate: &str) {
    todo!("Scan ds with the predicate");
}

async fn test_ann(_original: &RecordBatch, _ds: &Dataset, _predicate: Option<&str>) {
    todo!("Scan ds with the ANN predicate");
}
