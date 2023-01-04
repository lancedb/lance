use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};

/// Dataset Scanner
pub struct Scanner {}

impl Scanner {
    pub fn new() -> Self {
        Self {}
    }
}

impl RecordBatchReader for Scanner {
    fn schema(&self) -> SchemaRef {
        todo!()
    }
}

impl Iterator for Scanner {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}
