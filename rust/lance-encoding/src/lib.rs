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

use std::ops::Range;

use bytes::Bytes;
use futures::future::BoxFuture;

use lance_core::Result;

pub mod decoder;
pub mod encoder;
pub mod encodings;
pub mod format;
#[cfg(test)]
pub mod testing;

/// A trait for an I/O service
///
/// This represents the I/O API that the encoders and decoders need in order to operate.
/// We specify this as a trait so that lance-encodings does not need to depend on lance-io
///
/// In general, it is assumed that this trait will be implemented by some kind of "file reader"
/// or "file scheduler".  The encodings here are all limited to accessing a single file.
pub trait EncodingsIo: Send + Sync {
    /// Submit an I/O request
    ///
    /// The response must contain a `Bytes` object for each range requested even if the underlying
    /// I/O was coalesced into fewer actual requests.
    ///
    /// # Arguments
    ///
    /// * `ranges` - the byte ranges to request
    fn submit_request(&self, range: Vec<Range<u64>>) -> BoxFuture<'static, Result<Vec<Bytes>>>;
}
