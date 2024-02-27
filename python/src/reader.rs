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
use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use futures::lock::Mutex;
use futures::stream::StreamExt;

use lance::dataset::scanner::{DatasetRecordBatchStream, Scanner as LanceScanner};
use lance_io::stream::RecordBatchStream;

use crate::RT;

/// Lance's RecordBatchReader
/// This implements Arrow's RecordBatchReader trait
/// which is then used for FFI to turn this into
/// an ArrowArrayStream in the Arrow C Data Interface
pub struct LanceReader {
    schema: SchemaRef,
    /// We wrap stream in a mutex so we can call `next` in the background
    /// executor while we still have a reference to the stream on the main thread.
    stream: Arc<Mutex<DatasetRecordBatchStream>>,
}

impl LanceReader {
    pub async fn try_new(scanner: Arc<LanceScanner>) -> ::lance::error::Result<Self> {
        Ok(Self {
            schema: scanner.schema().await?,
            stream: Arc::new(Mutex::new(scanner.try_into_stream().await?)), // needs tokio Runtime
        })
    }

    pub fn from_stream(stream: DatasetRecordBatchStream) -> Self {
        Self {
            schema: stream.schema(),
            stream: Arc::new(Mutex::new(stream)),
        }
    }
}

impl Iterator for LanceReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let stream = self.stream.clone();
        RT.spawn(None, async move {
            let mut stream = stream.lock().await;
            stream.next().await
        })
        .transpose()
        .map(|rs| match rs {
            Ok(Ok(batch)) => Ok(batch),
            Ok(Err(err)) => Err(ArrowError::from(err)),
            Err(err) => Err(ArrowError::ExternalError(Box::new(err))),
        })
    }
}

impl RecordBatchReader for LanceReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
