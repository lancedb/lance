use arrow::pyarrow::PyArrowType;
use arrow_array::RecordBatch;
use arrow_schema::Schema;
use lance_datagen::{BatchCount, ByteCount};
use pyo3::{pyfunction, types::PyModule, wrap_pyfunction, PyResult, Python};

const DEFAULT_BATCH_SIZE_BYTES: u64 = 32 * 1024;
const DEFAULT_BATCH_COUNT: u32 = 4;

#[pyfunction]
pub fn is_datagen_supported() -> bool {
    true
}

#[pyfunction]
pub fn rand_batches(
    schema: PyArrowType<Schema>,
    batch_count: Option<u32>,
    bytes_in_batch: Option<u64>,
) -> PyResult<Vec<PyArrowType<RecordBatch>>> {
    lance_datagen::rand(&schema.0)
        .into_reader_bytes(
            ByteCount::from(bytes_in_batch.unwrap_or(DEFAULT_BATCH_SIZE_BYTES)),
            BatchCount::from(batch_count.unwrap_or(DEFAULT_BATCH_COUNT)),
            lance_datagen::RoundingBehavior::RoundUp,
        )
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to generate batches: {}", e))
        })?
        .map(|item| {
            item.map(PyArrowType::from).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Failed to generate batch: {}", e))
            })
        })
        .collect::<PyResult<Vec<PyArrowType<RecordBatch>>>>()
}

pub fn register_datagen(py: Python, m: &PyModule) -> PyResult<()> {
    let datagen = PyModule::new(py, "datagen")?;
    datagen.add_wrapped(wrap_pyfunction!(is_datagen_supported))?;
    datagen.add_wrapped(wrap_pyfunction!(rand_batches))?;
    m.add_submodule(datagen)?;
    Ok(())
}
