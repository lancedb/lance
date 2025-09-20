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

//! Graph query functionality for Lance datasets

use lance_graph::{
    CypherQuery as RustCypherQuery, GraphConfig as RustGraphConfig, GraphError as RustGraphError,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use serde_json::Value as JsonValue;

/// Convert GraphError to PyErr
fn graph_error_to_pyerr(err: RustGraphError) -> PyErr {
    PyRuntimeError::new_err(format!("Graph error: {}", err))
}

/// Graph configuration for interpreting Lance datasets as property graphs
#[pyclass(module = "lance")]
#[derive(Clone)]
pub struct GraphConfig {
    inner: RustGraphConfig,
}

#[pymethods]
impl GraphConfig {
    /// Create a new GraphConfig builder
    #[staticmethod]
    fn builder() -> GraphConfigBuilder {
        GraphConfigBuilder::new()
    }

    /// Get node labels
    fn node_labels(&self) -> Vec<String> {
        // Extract from the config's node mappings
        self.inner.node_mappings.keys().cloned().collect()
    }

    /// Get relationship types
    fn relationship_types(&self) -> Vec<String> {
        // Extract from the config's relationship mappings
        self.inner.relationship_mappings.keys().cloned().collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "GraphConfig(nodes={:?}, relationships={:?})",
            self.node_labels(),
            self.relationship_types()
        )
    }
}

/// Builder for GraphConfig
#[pyclass(module = "lance")]
pub struct GraphConfigBuilder {
    inner: lance_graph::config::GraphConfigBuilder,
}

#[pymethods]
impl GraphConfigBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: RustGraphConfig::builder(),
        }
    }

    /// Add a node label mapping
    ///
    /// Parameters
    /// ----------
    /// label : str
    ///     The node label (e.g., "Person")
    /// id_field : str
    ///     The field in the dataset that serves as the node ID
    ///
    /// Returns
    /// -------
    /// None
    fn with_node_label(&mut self, label: &str, id_field: &str) -> PyResult<()> {
        self.inner = self.inner.clone().with_node_label(label, id_field);
        Ok(())
    }

    /// Add a relationship mapping
    ///
    /// Parameters
    /// ----------
    /// rel_type : str
    ///     The relationship type (e.g., "KNOWS")
    /// source_field : str
    ///     The field containing source node IDs
    /// target_field : str
    ///     The field containing target node IDs
    ///
    /// Returns
    /// -------
    /// None
    fn with_relationship(
        &mut self,
        rel_type: &str,
        source_field: &str,
        target_field: &str,
    ) -> PyResult<()> {
        self.inner = self
            .inner
            .clone()
            .with_relationship(rel_type, source_field, target_field);
        Ok(())
    }

    /// Build the GraphConfig
    ///
    /// Returns
    /// -------
    /// GraphConfig
    ///     The configured graph config
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the configuration is invalid
    fn build(&self) -> PyResult<GraphConfig> {
        let config = self.inner.clone().build().map_err(graph_error_to_pyerr)?;
        Ok(GraphConfig { inner: config })
    }
}

/// Cypher query interface for Lance datasets
#[pyclass(module = "lance")]
#[derive(Clone)]
pub struct CypherQuery {
    inner: RustCypherQuery,
}

#[pymethods]
impl CypherQuery {
    /// Create a new Cypher query
    ///
    /// Parameters
    /// ----------
    /// query_text : str
    ///     The Cypher query string
    ///
    /// Returns
    /// -------
    /// CypherQuery
    ///     The parsed query
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the query cannot be parsed
    #[new]
    fn new(query_text: &str) -> PyResult<Self> {
        let query = RustCypherQuery::new(query_text).map_err(graph_error_to_pyerr)?;
        Ok(Self { inner: query })
    }

    /// Set the graph configuration
    ///
    /// Parameters
    /// ----------
    /// config : GraphConfig
    ///     The graph configuration
    ///
    /// Returns
    /// -------
    /// CypherQuery
    ///     A new query instance with the config set
    fn with_config(&self, config: &GraphConfig) -> Self {
        Self {
            inner: self.inner.clone().with_config(config.inner.clone()),
        }
    }

    /// Add a query parameter
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Parameter name
    /// value : object
    ///     Parameter value (will be converted to JSON)
    ///
    /// Returns
    /// -------
    /// CypherQuery
    ///     A new query instance with the parameter added
    fn with_parameter(&self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Convert Python value to JSON
        let json_value = python_to_json(value)?;
        Ok(Self {
            inner: self.inner.clone().with_parameter(key, json_value),
        })
    }

    /// Get the query text
    fn query_text(&self) -> &str {
        self.inner.query_text()
    }

    /// Get query parameters
    fn parameters(&self) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            for (key, value) in self.inner.parameters() {
                let py_value = json_to_python(py, value)?;
                dict.set_item(key, py_value)?;
            }
            Ok(dict.unbind())
        })
    }

    /// Convert query to SQL
    ///
    /// Returns
    /// -------
    /// str
    ///     The generated SQL query
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If SQL generation fails
    fn to_sql(&self) -> PyResult<String> {
        self.inner.to_sql().map_err(graph_error_to_pyerr)
    }

    /// Execute query against Lance datasets
    ///
    /// Parameters
    /// ----------
    /// datasets : dict
    ///     Dictionary mapping table names to Lance datasets
    ///
    /// Returns
    /// -------
    /// pyarrow.Table
    ///     Query results as Arrow table
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If query execution fails
    fn execute(&self, py: Python, datasets: &Bound<'_, PyDict>) -> PyResult<PyObject> {
        use std::collections::HashMap;

        // Convert provided Python datasets into Arrow RecordBatches without any Python-side execution logic
        let mut arrow_datasets = HashMap::new();

        for (key, value) in datasets.iter() {
            let table_name: String = key.extract()?;

            // Accept either a Lance Dataset (has to_table) or a PyArrow Table/Reader
            let batch = if value.hasattr("to_table")? {
                let table = value.call_method0("to_table")?;
                python_table_to_record_batch(py, &table)?
            } else {
                // Assume value is a PyArrow table-like object supporting to_batches / __arrow_c_stream__
                python_table_to_record_batch(py, &value)?
            };

            arrow_datasets.insert(table_name, batch);
        }

        // Call into Rust lance-graph for execution
        let runtime = tokio::runtime::Runtime::new().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create async runtime: {}", e))
        })?;

        let result_batch = runtime
            .block_on(async { self.inner.execute(arrow_datasets).await })
            .map_err(graph_error_to_pyerr)?;

        // Return a PyArrow Table via Arrow C stream
        record_batch_to_python_table(py, &result_batch)
    }

    /// Get variables used in the query
    fn variables(&self) -> Vec<String> {
        self.inner.variables()
    }

    /// Get node labels referenced in the query
    fn node_labels(&self) -> Vec<String> {
        self.inner.ast().get_node_labels()
    }

    /// Get relationship types referenced in the query
    fn relationship_types(&self) -> Vec<String> {
        self.inner.ast().get_relationship_types()
    }

    fn __repr__(&self) -> String {
        format!("CypherQuery(\"{}\")", self.inner.query_text())
    }
}

// Helper functions to convert between Python and JSON values
fn python_to_json(value: &Bound<'_, PyAny>) -> PyResult<JsonValue> {
    if value.is_none() {
        Ok(JsonValue::Null)
    } else if let Ok(b) = value.extract::<bool>() {
        Ok(JsonValue::Bool(b))
    } else if let Ok(i) = value.extract::<i64>() {
        Ok(JsonValue::Number(i.into()))
    } else if let Ok(f) = value.extract::<f64>() {
        Ok(JsonValue::Number(
            serde_json::Number::from_f64(f)
                .ok_or_else(|| PyValueError::new_err("Invalid float value"))?,
        ))
    } else if let Ok(s) = value.extract::<String>() {
        Ok(JsonValue::String(s))
    } else {
        Err(PyValueError::new_err("Unsupported parameter type"))
    }
}

fn json_to_python(py: Python, value: &JsonValue) -> PyResult<PyObject> {
    match value {
        JsonValue::Null => Ok(py.None()),
        JsonValue::Bool(b) => Ok(b.into_py(py)),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_py(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_py(py))
            } else {
                Ok(n.to_string().into_py(py))
            }
        }
        JsonValue::String(s) => Ok(s.into_py(py)),
        JsonValue::Array(_) | JsonValue::Object(_) => {
            // For complex types, convert to string representation
            Ok(value.to_string().into_py(py))
        }
    }
}

// Helper functions for Arrow conversion
fn python_table_to_record_batch(
    py: Python,
    table: &Bound<'_, PyAny>,
) -> PyResult<arrow::record_batch::RecordBatch> {
    // Import PyArrow's Table.to_batches() and take the first batch
    let batches = table.call_method0("to_batches")?;
    let batch_list: Vec<&PyAny> = batches.extract()?;

    if batch_list.is_empty() {
        return Err(PyRuntimeError::new_err("Table has no data"));
    }

    // Use Arrow's Python integration to convert the first batch
    let batch_py = batch_list[0];

    // Convert PyArrow RecordBatch to Rust RecordBatch via FFI
    use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
    use arrow::record_batch::RecordBatch;
    use pyo3::ffi::Py_uintptr_t;

    // Get the Arrow C interface pointers
    let capsule = batch_py.call_method0("__arrow_c_stream__")?;
    let stream_ptr: Py_uintptr_t = capsule.extract()?;

    // Create ArrowArrayStream from the pointer
    let stream = unsafe {
        FFI_ArrowArrayStream::from_raw(stream_ptr as *mut arrow::ffi_stream::FFI_ArrowArrayStream)
    };

    let mut reader = ArrowArrayStreamReader::try_new(stream).map_err(|e| {
        PyRuntimeError::new_err(format!("Failed to create Arrow stream reader: {}", e))
    })?;

    // Read the first batch
    reader
        .next()
        .transpose()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read Arrow batch: {}", e)))?
        .ok_or_else(|| PyRuntimeError::new_err("No batches available"))
}

fn record_batch_to_python_table(
    py: Python,
    batch: &arrow::record_batch::RecordBatch,
) -> PyResult<PyObject> {
    use arrow::ffi_stream::{ArrowArrayStreamWriter, FFI_ArrowArrayStream};
    use pyo3::types::PyCapsule;

    // Create an Arrow stream from the batch
    let batches = vec![batch.clone()];
    let schema = batch.schema();

    let mut stream = FFI_ArrowArrayStream::empty();
    let writer =
        ArrowArrayStreamWriter::try_new(&mut stream as *mut _, schema, Some(batches.into_iter()))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create Arrow stream: {}", e)))?;

    // Convert to Python using the Arrow C interface
    let stream_ptr = &stream as *const FFI_ArrowArrayStream as usize;
    let capsule = PyCapsule::new(py, stream_ptr, None)?;

    // Import PyArrow and create Table from stream
    let pyarrow = py.import("pyarrow")?;
    let table = pyarrow.call_method1("table", (capsule,))?;

    Ok(table.into())
}

/// Register graph functionality with the Python module
pub fn register_graph_module(py: Python, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let graph_module = PyModule::new(py, "graph")?;

    graph_module.add_class::<GraphConfig>()?;
    graph_module.add_class::<GraphConfigBuilder>()?;
    graph_module.add_class::<CypherQuery>()?;

    parent_module.add_submodule(&graph_module)?;
    Ok(())
}
