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

use std::collections::HashMap;
use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::Schema;
use lance_graph::{
    CypherQuery as RustCypherQuery, GraphConfig as RustGraphConfig, GraphError as RustGraphError,
};
use pyo3::{
    exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
    IntoPyObject,
};
use serde_json::Value as JsonValue;

use crate::RT;

/// Convert GraphError to PyErr
fn graph_error_to_pyerr(err: RustGraphError) -> PyErr {
    match &err {
        RustGraphError::ParseError { .. }
        | RustGraphError::ConfigError { .. }
        | RustGraphError::PlanError { .. }
        | RustGraphError::InvalidPattern { .. } => PyValueError::new_err(err.to_string()),
        RustGraphError::UnsupportedFeature { .. } => {
            PyNotImplementedError::new_err(err.to_string())
        }
        _ => PyRuntimeError::new_err(err.to_string()),
    }
}

/// Graph configuration for interpreting Lance datasets as property graphs
#[pyclass(name = "GraphConfig", module = "lance.graph")]
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
#[pyclass(name = "GraphConfigBuilder", module = "lance.graph")]
#[derive(Clone)]
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
    /// GraphConfigBuilder
    ///     A new builder with the node mapping applied
    fn with_node_label(&self, label: &str, id_field: &str) -> Self {
        Self {
            inner: self.inner.clone().with_node_label(label, id_field),
        }
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
    /// GraphConfigBuilder
    ///     A new builder with the relationship mapping applied
    fn with_relationship(&self, rel_type: &str, source_field: &str, target_field: &str) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .with_relationship(rel_type, source_field, target_field),
        }
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
#[pyclass(name = "CypherQuery", module = "lance.graph")]
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
    fn parameters(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for (key, value) in self.inner.parameters() {
            let py_value = json_to_python(py, value)?;
            dict.set_item(key, py_value)?;
        }
        Ok(dict.unbind())
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
        // SQL generation not yet implemented in lance-graph.
        // Return the original query text for now to keep API stable.
        Ok(self.inner.query_text().to_string())
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
        // Convert datasets to Arrow batches while holding the GIL - same as before
        let arrow_datasets = python_datasets_to_batches(datasets)?;

        // Clone the inner query for use in the async block
        let inner_query = self.inner.clone();

        // Use RT.block_on with Some(py) like the scanner to_pyarrow method
        let result_batch = RT
            .block_on(Some(py), inner_query.execute(arrow_datasets))?
            .map_err(graph_error_to_pyerr)?;

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
        JsonValue::Bool(b) => {
            use pyo3::types::PyBool;
            Ok(PyBool::new(py, *b).to_owned().into())
        }
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.unbind().into())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.unbind().into())
            } else {
                Ok(n.to_string().into_pyobject(py)?.unbind().into())
            }
        }
        JsonValue::String(s) => Ok(s.as_str().into_pyobject(py)?.unbind().into()),
        JsonValue::Array(_) | JsonValue::Object(_) => {
            // For complex types, convert to string representation
            Ok(value.to_string().into_pyobject(py)?.unbind().into())
        }
    }
}

// Helper functions for Arrow conversion
fn python_datasets_to_batches(
    datasets: &Bound<'_, PyDict>,
) -> PyResult<HashMap<String, RecordBatch>> {
    let mut arrow_datasets = HashMap::new();
    for (key, value) in datasets.iter() {
        let table_name: String = key.extract()?;
        let batch = if is_lance_dataset(&value)? {
            // Handle Lance datasets using scan() -> to_pyarrow() pattern that works elsewhere
            lance_dataset_to_record_batch(&value)?
        } else if value.hasattr("to_table")? {
            let table = value.call_method0("to_table")?;
            python_any_to_record_batch(&table)?
        } else {
            python_any_to_record_batch(&value)?
        };
        let batch = normalize_record_batch(batch)?;
        arrow_datasets.insert(table_name, batch);
    }
    Ok(arrow_datasets)
}

fn normalize_record_batch(batch: RecordBatch) -> PyResult<RecordBatch> {
    if batch.schema().metadata().is_empty() {
        return Ok(batch);
    }

    // DataFusion expects stable, metadata-free schemas across optimization passes.
    // Rebuild the schema without the PyArrow-specific metadata to avoid mismatches.
    let columns = batch.columns().to_vec();
    let fields = batch
        .schema()
        .fields()
        .iter()
        .map(|field| (**field).clone())
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, columns).map_err(|e| {
        PyRuntimeError::new_err(format!("Failed to strip metadata from Arrow batch: {}", e))
    })
}

// Check if a Python object is a Lance dataset
fn is_lance_dataset(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    // Check the type name directly
    if let Ok(type_name) = value.get_type().repr() {
        let type_str = type_name.to_string();
        let is_lance = type_str.contains("lance.dataset.LanceDataset");
        return Ok(is_lance);
    }

    // Fallback: check for uri (which we know Lance datasets have)
    value.hasattr("uri")
}

// Convert Lance dataset to RecordBatch using alternative methods that avoid GIL issues
fn lance_dataset_to_record_batch(dataset: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    // Try the scanner() approach that's used elsewhere in the Lance codebase
    if let Ok(scanner) = dataset.call_method0("scanner") {
        if let Ok(py_reader) = scanner.call_method0("to_pyarrow") {
            return python_any_to_record_batch(&py_reader);
        }
    }

    // Method 2: Use the count_rows + take approach to get data without to_table()
    if dataset.hasattr("count_rows")? && dataset.hasattr("take")? {
        let count = dataset.call_method0("count_rows")?;
        let count_int: usize = count.extract()?;
        if count_int > 0 {
            // Take a range of rows (limit to 10000 for performance)
            let take_count = std::cmp::min(count_int, 10000);

            // Create a Python list of indices
            let py = dataset.py();
            let range_list = pyo3::types::PyList::empty(py);
            for i in 0..take_count {
                range_list.append(i)?;
            }

            if let Ok(table) = dataset.call_method1("take", (range_list,)) {
                return python_any_to_record_batch(&table);
            }
        }
    }

    // Fallback: Use to_table() (this might still cause GIL issues but is last resort)
    let table = dataset.call_method0("to_table")?;
    python_any_to_record_batch(&table)
}

fn python_any_to_record_batch(value: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    use arrow::pyarrow::FromPyArrow;

    if let Ok(batch) = RecordBatch::from_pyarrow_bound(value) {
        return Ok(batch);
    }

    let mut reader = ArrowArrayStreamReader::from_pyarrow_bound(value)?;
    let schema = reader.schema();
    let mut batches = Vec::new();
    while let Some(batch) = reader
        .next()
        .transpose()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read Arrow batch: {}", e)))?
    {
        batches.push(batch);
    }

    if batches.is_empty() {
        return Err(PyRuntimeError::new_err("Table has no data"));
    }

    concat_batches(&schema, &batches)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to concatenate batches: {}", e)))
}

fn record_batch_to_python_table(
    py: Python,
    batch: &arrow_array::RecordBatch,
) -> PyResult<PyObject> {
    use arrow::pyarrow::ToPyArrow;
    use pyo3::types::PyList;

    // Convert RecordBatch -> PyArrow.RecordBatch
    let py_rb = batch.to_pyarrow(py)?;

    // Build pyarrow.Table from batches
    let pa = py.import("pyarrow")?;
    let table_cls = pa.getattr("Table")?;
    let batches = PyList::new(py, [py_rb])?;
    let table = table_cls.call_method1("from_batches", (batches,))?;
    Ok(table.unbind())
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
