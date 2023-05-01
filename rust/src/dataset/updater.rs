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

use arrow_array::RecordBatch;

use super::fragment::FragmentReader;
use crate::Result;

/// Update or insert a new column.
pub struct Updater {
    /// The reader over the [`Fragment`]
    reader: FragmentReader,

    last_input: Option<RecordBatch>,
}

impl Updater {
    /// Create a new updater.
    fn new(reader: FragmentReader) -> Self {
        Self {
            reader,
            last_input: None,
        }
    }

    /// Returns the next [`RecordBatch`] as input for updater.
    pub async fn next(&mut self) -> Result<Option<RecordBatch>> {
        todo!()
    }
}