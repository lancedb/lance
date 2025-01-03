// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::pyarrow::{PyArrowType, ToPyArrow};
use arrow_array::{Array, FixedSizeListArray};
use arrow_data::ArrayData;
use lance::index::vector::ivf::builder::write_vector_storage;
use lance::io::ObjectStore;
use lance_index::vector::ivf::shuffler::{shuffle_vectors, IvfShuffler};
use lance_index::vector::{
    ivf::{storage::IvfModel, IvfBuildParams},
    pq::{PQBuildParams, ProductQuantizer},
};
use lance_linalg::distance::DistanceType;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyModuleMethods;
use pyo3::Bound;
use pyo3::{
    pyfunction,
    types::{PyList, PyModule},
    wrap_pyfunction, PyObject, PyResult, Python,
};

use crate::fragment::FileFragment;
use crate::{
    dataset::Dataset, error::PythonErrorExt, file::object_store_from_uri_or_path_no_options, RT,
};
use lance::index::vector::ivf::write_ivf_pq_file_from_existing_index;
use lance_index::DatasetIndexExt;
use uuid::Uuid;

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
#[allow(clippy::too_many_arguments)]
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

#[allow(clippy::too_many_arguments)]
async fn do_train_pq_model(
    dataset: &Dataset,
    column: &str,
    dimension: usize,
    num_subvectors: u32,
    distance_type: &str,
    sample_rate: u32,
    max_iters: u32,
    ivf_model: IvfModel,
) -> PyResult<ArrayData> {
    // We verify distance_type earlier so can unwrap here
    let distance_type = DistanceType::try_from(distance_type).unwrap();
    let params = PQBuildParams {
        num_sub_vectors: num_subvectors as usize,
        num_bits: 8,
        max_iters: max_iters as usize,
        sample_rate: sample_rate as usize,
        ..Default::default()
    };
    let pq_model = lance::index::vector::pq::build_pq_model(
        dataset.ds.as_ref(),
        column,
        dimension,
        distance_type,
        &params,
        Some(&ivf_model),
    )
    .await
    .infer_error()?;
    Ok(pq_model.codebook.into_data())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn train_pq_model(
    py: Python<'_>,
    dataset: &Dataset,
    column: &str,
    dimension: usize,
    num_subvectors: u32,
    distance_type: &str,
    sample_rate: u32,
    max_iters: u32,
    ivf_centroids: PyArrowType<ArrayData>,
) -> PyResult<PyObject> {
    let ivf_centroids = ivf_centroids.0;
    let ivf_centroids = FixedSizeListArray::from(ivf_centroids);
    let ivf_model = IvfModel {
        centroids: Some(ivf_centroids),
        offsets: vec![],
        lengths: vec![],
    };
    let codebook = RT.block_on(
        Some(py),
        do_train_pq_model(
            dataset,
            column,
            dimension,
            num_subvectors,
            distance_type,
            sample_rate,
            max_iters,
            ivf_model,
        ),
    )??;
    codebook.to_pyarrow(py)
}

#[allow(clippy::too_many_arguments)]
async fn do_transform_vectors(
    dataset: &Dataset,
    column: &str,
    distance_type: DistanceType,
    ivf_centroids: FixedSizeListArray,
    pq_model: ProductQuantizer,
    dst_uri: &str,
    fragments: Vec<FileFragment>,
    partitions_ds_uri: Option<&str>,
) -> PyResult<()> {
    let num_rows = dataset.ds.count_rows(None).await.infer_error()?;
    let fragments = fragments.iter().map(|item| item.metadata().0).collect();
    let transform_input = dataset
        .ds
        .scan()
        .with_fragments(fragments)
        .project(&[column])
        .infer_error()?
        .with_row_id()
        .batch_size(8192)
        .try_into_stream()
        .await
        .infer_error()?;

    let (obj_store, path) = object_store_from_uri_or_path_no_options(dst_uri).await?;
    let writer = obj_store.create(&path).await.infer_error()?;
    write_vector_storage(
        &dataset.ds,
        transform_input,
        num_rows as u64,
        ivf_centroids,
        pq_model,
        distance_type,
        column,
        writer,
        partitions_ds_uri,
    )
    .await
    .infer_error()?;
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature=(dataset, column, dimension, num_subvectors, distance_type, ivf_centroids, pq_codebook, dst_uri, fragments, partitions_ds_uri=None))]
pub fn transform_vectors(
    py: Python<'_>,
    dataset: &Dataset,
    column: &str,
    dimension: usize,
    num_subvectors: u32,
    distance_type: &str,
    ivf_centroids: PyArrowType<ArrayData>,
    pq_codebook: PyArrowType<ArrayData>,
    dst_uri: &str,
    fragments: Vec<FileFragment>,
    partitions_ds_uri: Option<&str>,
) -> PyResult<()> {
    let ivf_centroids = ivf_centroids.0;
    let ivf_centroids = FixedSizeListArray::from(ivf_centroids);
    let codebook = pq_codebook.0;
    let codebook = FixedSizeListArray::from(codebook);
    let distance_type = DistanceType::try_from(distance_type).unwrap();
    let pq = ProductQuantizer::new(
        num_subvectors as usize,
        /*num_bits=*/ 8,
        dimension,
        codebook,
        distance_type,
    );
    RT.block_on(
        Some(py),
        do_transform_vectors(
            dataset,
            column,
            distance_type,
            ivf_centroids,
            pq,
            dst_uri,
            fragments,
            partitions_ds_uri,
        ),
    )?
}

async fn do_shuffle_transformed_vectors(
    unsorted_filenames: Vec<String>,
    dir_path: &str,
    ivf_centroids: FixedSizeListArray,
    shuffle_output_root_filename: &str,
) -> PyResult<Vec<String>> {
    let (obj_store, path) = ObjectStore::from_path(dir_path).infer_error()?;
    if !obj_store.is_local() {
        return Err(PyValueError::new_err(
            "shuffle_vectors input and output path is currently required to be local",
        ));
    }
    let partition_files = shuffle_vectors(
        unsorted_filenames,
        path,
        ivf_centroids,
        shuffle_output_root_filename,
    )
    .await
    .infer_error()?;
    Ok(partition_files)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn shuffle_transformed_vectors(
    py: Python<'_>,
    unsorted_filenames: Vec<String>,
    dir_path: &str,
    ivf_centroids: PyArrowType<ArrayData>,
    shuffle_output_root_filename: &str,
) -> PyResult<PyObject> {
    let ivf_centroids = ivf_centroids.0;
    let ivf_centroids = FixedSizeListArray::from(ivf_centroids);

    let result = RT.block_on(
        None,
        do_shuffle_transformed_vectors(
            unsorted_filenames,
            dir_path,
            ivf_centroids,
            shuffle_output_root_filename,
        ),
    )?;

    match result {
        Ok(partition_files) => {
            let py_list = PyList::new_bound(py, partition_files);
            Ok(py_list.into())
        }
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}

async fn do_load_shuffled_vectors(
    filenames: Vec<String>,
    dir_path: &str,
    dataset: &Dataset,
    column: &str,
    index_name: &str,
    ivf_model: IvfModel,
    pq_model: ProductQuantizer,
) -> PyResult<()> {
    let (_, path) = object_store_from_uri_or_path_no_options(dir_path).await?;
    let streams = IvfShuffler::load_partitioned_shuffles(&path, filenames)
        .await
        .infer_error()?;

    let index_id = Uuid::new_v4();

    write_ivf_pq_file_from_existing_index(
        &dataset.ds,
        column,
        index_name,
        index_id,
        ivf_model,
        pq_model,
        streams,
    )
    .await
    .infer_error()?;

    let mut ds = dataset.ds.as_ref().clone();
    ds.commit_existing_index(index_name, column, index_id)
        .await
        .infer_error()?;

    Ok(())
}

#[pyfunction]
#[pyo3(signature=(filenames, dir_path, dataset, column, ivf_centroids, pq_codebook, pq_dimension, num_subvectors, distance_type, index_name=None))]
#[allow(clippy::too_many_arguments)]
pub fn load_shuffled_vectors(
    filenames: Vec<String>,
    dir_path: &str,
    dataset: &Dataset,
    column: &str,
    ivf_centroids: PyArrowType<ArrayData>,
    pq_codebook: PyArrowType<ArrayData>,
    pq_dimension: usize,
    num_subvectors: u32,
    distance_type: &str,
    index_name: Option<&str>,
) -> PyResult<()> {
    let mut default_idx_name = column.to_string();
    default_idx_name.push_str("_idx");
    let idx_name = index_name.unwrap_or(default_idx_name.as_str());

    let ivf_centroids = ivf_centroids.0;
    let ivf_centroids = FixedSizeListArray::from(ivf_centroids);

    let ivf_model = IvfModel {
        centroids: Some(ivf_centroids),
        offsets: vec![],
        lengths: vec![],
    };

    let codebook = pq_codebook.0;
    let codebook = FixedSizeListArray::from(codebook);

    let distance_type = DistanceType::try_from(distance_type).unwrap();
    let pq_model = ProductQuantizer::new(
        num_subvectors as usize,
        /*num_bits=*/ 8,
        pq_dimension,
        codebook,
        distance_type,
    );

    RT.block_on(
        None,
        do_load_shuffled_vectors(
            filenames, dir_path, dataset, column, idx_name, ivf_model, pq_model,
        ),
    )?
}

pub fn register_indices(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let indices = PyModule::new_bound(py, "indices")?;
    indices.add_wrapped(wrap_pyfunction!(train_ivf_model))?;
    indices.add_wrapped(wrap_pyfunction!(train_pq_model))?;
    indices.add_wrapped(wrap_pyfunction!(transform_vectors))?;
    indices.add_wrapped(wrap_pyfunction!(shuffle_transformed_vectors))?;
    indices.add_wrapped(wrap_pyfunction!(load_shuffled_vectors))?;
    m.add_submodule(&indices)?;
    Ok(())
}
