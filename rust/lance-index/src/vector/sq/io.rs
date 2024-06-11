// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{types::Float32Type, ArrayRef, RecordBatch};
use lance_core::{utils::tokio::spawn_cpu, Result, ROW_ID};
use lance_file::writer::FileWriter;
use lance_linalg::distance::DistanceType;
use lance_table::io::manifest::ManifestDescribing;

use crate::scalar::IndexWriter;
use crate::vector::{
    quantizer::Quantization,
    sq::{storage::ScalarQuantizationStorage, ScalarQuantizer},
    storage::VectorStore,
};

/// IO utility to build and write SQ storage.
pub async fn build_and_write_sq_storage(
    distance_type: DistanceType,
    row_ids: ArrayRef,
    vectors: ArrayRef,
    sq: ScalarQuantizer,
    mut writer: FileWriter<ManifestDescribing>,
) -> Result<()> {
    let storage = spawn_cpu(move || {
        let storage = build_sq_storage(row_ids, vectors, distance_type, sq)?;
        Ok(storage)
    })
    .await?;

    for batch in storage.to_batches()? {
        writer.write_record_batch(batch.clone()).await?;
    }
    writer.finish().await?;
    Ok(())
}

/// Build [ScalarQuantizationStorage] from the given vectors.
fn build_sq_storage(
    row_ids: ArrayRef,
    vectors: ArrayRef,
    distance_type: DistanceType,
    sq: ScalarQuantizer,
) -> Result<ScalarQuantizationStorage> {
    let code_column = sq.transform::<Float32Type>(vectors.as_ref())?;
    std::mem::drop(vectors);

    let batch = RecordBatch::try_from_iter_with_nullable(vec![
        (ROW_ID, row_ids, true),
        (sq.column(), code_column, false),
    ])?;
    let store =
        ScalarQuantizationStorage::try_new(sq.num_bits(), distance_type, sq.bounds(), [batch])?;

    Ok(store)
}
