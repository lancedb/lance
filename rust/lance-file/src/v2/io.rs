// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use futures::{future::BoxFuture, FutureExt};
use lance_encoding::EncodingsIo;
use lance_io::scheduler::FileScheduler;

#[derive(Debug)]
pub struct LanceEncodingsIo(pub FileScheduler);

impl EncodingsIo for LanceEncodingsIo {
    fn submit_request(
        &self,
        range: Vec<std::ops::Range<u64>>,
    ) -> BoxFuture<'static, lance_core::Result<Vec<bytes::Bytes>>> {
        self.0.submit_request(range).boxed()
    }
}
