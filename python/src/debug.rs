// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use lance::{datatypes::Schema, Error};
use lance_table::format::{DeletionFile, Fragment as LanceFragmentMetadata};
use pyo3::{exceptions::PyIOError, prelude::*};

use crate::{Dataset, FragmentMetadata, RT};

/// Format the Lance schema of a dataset as a string.
///
/// This can be used to view the field ids and types in the schema.
#[pyfunction]
pub fn format_schema(dataset: &PyAny) -> PyResult<String> {
    let py = dataset.py();
    let dataset = dataset.getattr("_ds")?.extract::<Py<Dataset>>()?;
    let dataset_ref = &dataset.as_ref(py).borrow().ds;
    let schema = dataset_ref.schema();
    Ok(format!("{:#?}", schema))
}

/// Print the full Lance manifest of the dataset.
#[pyfunction]
pub fn format_manifest(dataset: &PyAny) -> PyResult<String> {
    let py = dataset.py();
    let dataset = dataset.getattr("_ds")?.extract::<Py<Dataset>>()?;
    let dataset_ref = &dataset.as_ref(py).borrow().ds;
    let manifest = dataset_ref.manifest();
    Ok(format!("{:#?}", manifest))
}

// These are dead code because they just exist for the debug impl.
#[derive(Debug)]
#[allow(dead_code)]
struct PrettyPrintableFragment {
    id: u64,
    files: Vec<PrettyPrintableDataFile>,
    deletion_file: Option<DeletionFile>,
    physical_rows: Option<usize>,
}

#[derive(Debug)]
#[allow(dead_code)]
struct PrettyPrintableDataFile {
    path: String,
    fields: Vec<i32>,
    column_indices: Vec<i32>,
    schema: Schema,
    major_version: u32,
    minor_version: u32,
}

impl PrettyPrintableFragment {
    fn new(fragment: &LanceFragmentMetadata, schema: &Schema) -> Self {
        let files = fragment
            .files
            .iter()
            .map(|file| {
                let schema = schema.project_by_ids(&file.fields);
                PrettyPrintableDataFile {
                    path: file.path.clone(),
                    fields: file.fields.clone(),
                    column_indices: file.column_indices.clone(),
                    schema,
                    major_version: file.file_major_version,
                    minor_version: file.file_minor_version,
                }
            })
            .collect();

        Self {
            id: fragment.id,
            files,
            deletion_file: fragment.deletion_file.clone(),
            physical_rows: fragment.physical_rows,
        }
    }
}

/// Debug print a LanceFragment.
#[pyfunction]
pub fn format_fragment(fragment: &PyAny, dataset: &PyAny) -> PyResult<String> {
    let py = fragment.py();
    let fragment = fragment
        .getattr("_metadata")?
        .extract::<Py<FragmentMetadata>>()?;

    let dataset = dataset.getattr("_ds")?.extract::<Py<Dataset>>()?;
    let dataset_ref = &dataset.as_ref(py).borrow().ds;
    let schema = dataset_ref.schema();

    let meta = fragment.as_ref(py).borrow().inner.clone();
    let pp_meta = PrettyPrintableFragment::new(&meta, schema);
    Ok(format!("{:#?}", pp_meta))
}

/// Return a string representation of each transaction in the dataset, in
/// reverse chronological order.
///
/// If `max_transactions` is provided, only the most recent `max_transactions`
/// transactions will be returned. Defaults to 10.
#[pyfunction]
#[pyo3(signature = (dataset, /, max_transactions = 10))]
pub fn list_transactions(
    dataset: &PyAny,
    max_transactions: usize,
) -> PyResult<Vec<Option<String>>> {
    let py = dataset.py();
    let dataset = dataset.getattr("_ds")?.extract::<Py<Dataset>>()?;
    let mut dataset = dataset.as_ref(py).borrow().ds.clone();

    RT.block_on(Some(py), async move {
        let mut transactions = vec![];

        loop {
            let transaction = dataset.read_transaction().await.map_err(|err| {
                PyIOError::new_err(format!("Failed to read transaction file: {:?}", err))
            })?;
            if let Some(transaction) = transaction {
                transactions.push(Some(format!("{:#?}", transaction)));
            } else {
                transactions.push(None);
            }

            if transactions.len() >= max_transactions {
                break;
            } else {
                match dataset
                    .checkout_version(dataset.version().version - 1)
                    .await
                {
                    Ok(ds) => dataset = Arc::new(ds),
                    Err(Error::DatasetNotFound { .. }) => break,
                    Err(err) => {
                        return Err(PyIOError::new_err(format!(
                            "Failed to checkout version: {:?}",
                            err
                        )))
                    }
                }
            }
        }

        Ok(transactions)
    })?
}
