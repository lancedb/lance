// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use lance::Error;
use pyo3::{exceptions::PyIOError, prelude::*};

use crate::{Dataset, RT};

/// Print the Lance schema of a dataset.
///
/// This can be used to view the field ids and types in the schema.
#[pyfunction]
pub fn print_schema(dataset: &PyAny) -> PyResult<()> {
    let py = dataset.py();
    let dataset = dataset.getattr("_ds")?.extract::<Py<Dataset>>()?;
    let dataset_ref = &dataset.as_ref(py).borrow().ds;
    let schema = dataset_ref.schema();
    println!("{:?}", schema);
    Ok(())
}

/// Print the full Lance manifest of the dataset.
#[pyfunction]
pub fn print_manifest(dataset: &PyAny) -> PyResult<()> {
    let py = dataset.py();
    let dataset = dataset.getattr("_ds")?.extract::<Py<Dataset>>()?;
    let dataset_ref = &dataset.as_ref(py).borrow().ds;
    let manifest = dataset_ref.manifest();
    println!("{:?}", manifest);
    Ok(())
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
                transactions.push(Some(format!("{:?}", transaction)));
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
