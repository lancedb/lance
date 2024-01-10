// Copyright 2024 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use snafu::{location, Location};
use tracing::instrument;

use lance_core::{io::Writer, ROW_ID};
use lance_index::vector::{ivf::shuffler::shuffle_dataset, pq::ProductQuantizer};
use lance_linalg::distance::MetricType;

use crate::index::vector::ivf::{io::write_index_partitions, Ivf};
use crate::{io::RecordBatchStream, Error, Result};

/// Build specific partitions of IVF index.
///
///
#[allow(clippy::too_many_arguments)]
#[instrument(level = "debug", skip(writer, data, ivf, pq))]
pub(super) async fn build_partitions(
    writer: &mut dyn Writer,
    data: impl RecordBatchStream + Unpin + 'static,
    column: &str,
    ivf: &mut Ivf,
    pq: Arc<dyn ProductQuantizer>,
    metric_type: MetricType,
    part_range: Range<u32>,
    precomputed_partitions: Option<HashMap<u64, u32>>,
    shuffle_partition_batches: usize,
    shuffle_partition_concurrency: usize,
) -> Result<()> {
    let schema = data.schema();
    if schema.column_with_name(column).is_none() {
        return Err(Error::Schema {
            message: format!("column {} does not exist in data stream", column),
            location: location!(),
        });
    }
    if schema.column_with_name(ROW_ID).is_none() {
        return Err(Error::Schema {
            message: "ROW ID is not set when building index partitions".to_string(),
            location: location!(),
        });
    }

    let ivf_model = lance_index::vector::ivf::new_ivf_with_pq(
        ivf.centroids.values(),
        ivf.centroids.value_length() as usize,
        metric_type,
        column,
        pq.clone(),
        Some(part_range),
        precomputed_partitions,
    )?;

    let stream = shuffle_dataset(
        data,
        column,
        ivf_model,
        ivf.num_partitions() as u32,
        pq.num_sub_vectors(),
        shuffle_partition_batches,
        shuffle_partition_concurrency,
    )
    .await?;

    write_index_partitions(writer, ivf, stream, None).await?;

    Ok(())
}
