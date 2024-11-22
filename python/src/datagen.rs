use arrow::pyarrow::PyArrowType;
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::Schema;
use lance_datagen::{BatchCount, ByteCount, RowCount};
use pyo3::{pyfunction, types::PyModule, wrap_pyfunction, PyResult, Python};

const DEFAULT_BATCH_SIZE_BYTES: u64 = 32 * 1024;
const DEFAULT_BATCH_COUNT: u32 = 4;

#[pyfunction]
pub fn is_datagen_supported() -> bool {
    true
}

pub fn rand_reader_internal(
    schema: PyArrowType<Schema>,
    batch_count: Option<u32>,
    bytes_in_batch: Option<u64>,
    rows_in_batch: Option<u64>,
) -> PyResult<Box<dyn RecordBatchReader + Send>> {
    let batch_count = BatchCount::from(batch_count.unwrap_or(DEFAULT_BATCH_COUNT));
    let gen = lance_datagen::rand(&schema.0);
    Ok(match (bytes_in_batch, rows_in_batch) {
        (Some(_), Some(_)) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Only one of bytes_in_batch or rows_in_batch can be specified",
            ))
        }
        (None, None) => Box::new(
            gen.into_reader_bytes(
                ByteCount::from(DEFAULT_BATCH_SIZE_BYTES),
                batch_count,
                lance_datagen::RoundingBehavior::RoundUp,
            )
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Failed to generate batches: {}",
                    e
                ))
            })?,
        ),
        (Some(bytes_in_batch), None) => Box::new(
            gen.into_reader_bytes(
                ByteCount::from(bytes_in_batch),
                batch_count,
                lance_datagen::RoundingBehavior::RoundUp,
            )
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Failed to generate batches: {}",
                    e
                ))
            })?,
        ),
        (None, Some(rows_in_batch)) => {
            Box::new(gen.into_reader_rows(RowCount::from(rows_in_batch), batch_count))
        }
    })
}

#[pyfunction]
pub fn rand_batches(
    schema: PyArrowType<Schema>,
    batch_count: Option<u32>,
    bytes_in_batch: Option<u64>,
    rows_in_batch: Option<u64>,
) -> PyResult<Vec<PyArrowType<RecordBatch>>> {
    let reader = rand_reader_internal(schema, batch_count, bytes_in_batch, rows_in_batch)?;
    reader
        .map(|item| {
            item.map(PyArrowType::from).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Failed to generate batch: {}", e))
            })
        })
        .collect::<PyResult<Vec<PyArrowType<RecordBatch>>>>()
}

#[pyfunction]
pub fn rand_reader(
    schema: PyArrowType<Schema>,
    batch_count: Option<u32>,
    bytes_in_batch: Option<u64>,
    rows_in_batch: Option<u64>,
) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
    let reader = rand_reader_internal(schema, batch_count, bytes_in_batch, rows_in_batch)?;
    Ok(PyArrowType(Box::new(reader)))
}

pub fn register_datagen(py: Python, m: &PyModule) -> PyResult<()> {
    let datagen = PyModule::new(py, "datagen")?;
    datagen.add_wrapped(wrap_pyfunction!(is_datagen_supported))?;
    datagen.add_wrapped(wrap_pyfunction!(rand_batches))?;
    datagen.add_wrapped(wrap_pyfunction!(rand_reader))?;
    m.add_submodule(datagen)?;
    Ok(())
}
