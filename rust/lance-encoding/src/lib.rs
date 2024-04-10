// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

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
