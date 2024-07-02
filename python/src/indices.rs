// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::pyarrow::ToPyArrow;
use arrow_array::Array;
use arrow_data::ArrayData;
use lance_index::vector::ivf::IvfBuildParams;
use lance_linalg::distance::DistanceType;
use pyo3::{pyfunction, types::PyModule, wrap_pyfunction, PyObject, PyResult, Python};

use crate::{dataset::Dataset, error::PythonErrorExt, RT};

async fn do_train_ivf_model(
    dataset: &Dataset,
    column: &str,
    dimension: usize,
    num_partitions: u32,
    distance_type: &str,
    sample_rate: u32,
    max_iters: u32,
) -> PyResult<ArrayData> {
    // We verify distance_type earlier so can unwrap here
    let distance_type = DistanceType::try_from(distance_type).unwrap();
    let params = IvfBuildParams {
        max_iters: max_iters as usize,
        sample_rate: sample_rate as usize,
        num_partitions: num_partitions as usize,
        ..Default::default()
    };
    let ivf_model = lance::index::vector::ivf::build_ivf_model(
        dataset.ds.as_ref(),
        column,
        dimension,
        distance_type,
        &params,
    )
    .await
    .infer_error()?;
    let centroids = ivf_model.centroids.unwrap();
    Ok(centroids.into_data())
}

#[pyfunction]
fn train_ivf_model(
    py: Python<'_>,
    dataset: &Dataset,
    column: &str,
    dimension: usize,
    num_partitions: u32,
    distance_type: &str,
    sample_rate: u32,
    max_iters: u32,
) -> PyResult<PyObject> {
    let centroids = RT.block_on(
        Some(py),
        do_train_ivf_model(
            dataset,
            column,
            dimension,
            num_partitions,
            distance_type,
            sample_rate,
            max_iters,
        ),
    )??;
    centroids.to_pyarrow(py)
}

pub fn register_indices(py: Python, m: &PyModule) -> PyResult<()> {
    let indices = PyModule::new(py, "indices")?;
    indices.add_wrapped(wrap_pyfunction!(train_ivf_model))?;
    m.add_submodule(indices)?;
    Ok(())
}
