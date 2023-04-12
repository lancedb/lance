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
use arrow_schema::SchemaRef;
use futures::Stream;

use crate::Result;

/// RecordBatch Stream trait.
pub trait RecordBatchStream: Stream<Item = Result<RecordBatch>> + Send + 'static {
    /// Returns the schema of the stream.
    fn schema(&self) -> SchemaRef;
}
