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

use arrow::ffi_stream::ArrowArrayStreamReader;
use tokio::runtime::Runtime;

use lance::Dataset;
use lance::Result;

pub struct BlockingDataset {
    inner: Dataset,
    rt: Runtime,
}

impl BlockingDataset {
    pub fn write(reader: ArrowArrayStreamReader, uri: &str) -> Result<Self> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;

        let inner = rt.block_on(
            Dataset::write(reader, uri, Option::None)
        )?;
        Ok(BlockingDataset { inner, rt })
    }
    pub fn open(uri: &str) -> Result<Self> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;

        let inner = rt.block_on(Dataset::open(uri))?;
        Ok(BlockingDataset { inner, rt })
    }

    pub fn count_rows(&self) -> Result<usize> {
        self.rt.block_on(self.inner.count_rows())
    }
    pub fn close(&self) {}
}
