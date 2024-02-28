// Copyright 2023 Lance Developers.
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

use arrow::array::RecordBatchReader;

use lance::dataset::{Dataset, WriteParams};

use crate::Result;
use crate::RT;

pub struct BlockingDataset {
    inner: Dataset,
}

impl BlockingDataset {
    pub fn write(
        reader: impl RecordBatchReader + Send + 'static,
        uri: &str,
        params: Option<WriteParams>,
    ) -> Result<Self> {
        let inner = RT.block_on(Dataset::write(reader, uri, params))?;
        Ok(Self { inner })
    }

    pub fn open(uri: &str) -> Result<Self> {
        let inner = RT.block_on(Dataset::open(uri))?;
        Ok(Self { inner })
    }

    pub fn count_rows(&self) -> Result<usize> {
        Ok(RT.block_on(self.inner.count_rows())?)
    }

    pub fn close(&self) {}
}
