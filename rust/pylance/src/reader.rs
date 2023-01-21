// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
use futures::stream::StreamExt;
use std::sync::Arc;
use tokio::runtime::Runtime;

use ::lance::dataset::scanner::{Scanner as LanceScanner, ScannerStream};
use ::lance::error::Error;
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};

/// Lance's RecordBatchReader
/// This implements Arrow's RecordBatchReader trait
/// which is then used for FFI to turn this into
/// an ArrowArrayStream in the Arrow C Data Interface
pub struct LanceReader {
    schema: SchemaRef,
    stream: ScannerStream,
    rt: Arc<Runtime>,
}

impl LanceReader {
    pub fn new(scanner: LanceScanner, rt: Arc<Runtime>) -> Self {
        Self {
            schema: scanner.schema(),
            stream: scanner.into_stream(),
            rt,
        }
    }
}

impl Iterator for LanceReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let stream = &mut self.stream;
        self.rt.block_on(async {
            stream.next().await.map(|rs| {
                rs.map_err(|err| {
                    match err {
                        Error::Arrow(err) => ArrowError::IoError(err), // we lose the error type converting to LanceError
                        Error::IO(err) => ArrowError::IoError(err),
                        Error::Schema(err) => ArrowError::SchemaError(err),
                    }
                })
            })
        })
    }
}

impl RecordBatchReader for LanceReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
