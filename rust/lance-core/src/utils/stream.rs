// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use futures::{
    stream::{BoxStream, TryStream, TryStreamExt},
    StreamExt, TryFuture,
};

pub trait LanceTryStreamExt: TryStream {
    fn try_buffered_with_ordering<'a>(
        self,
        buffer_size: usize,
        ordered: bool,
    ) -> BoxStream<'a, Result<<Self::Ok as TryFuture>::Ok, Self::Error>>
    where
        Self::Ok: TryFuture<Error = Self::Error> + Send,
        Self: Sized + 'a + Send,
        <Self as TryStream>::Error: Send,
        <Self as TryStream>::Ok: Send,
        <<Self as TryStream>::Ok as TryFuture>::Ok: Send,
    {
        if ordered {
            self.try_buffered(buffer_size).boxed()
        } else {
            self.try_buffer_unordered(buffer_size).boxed()
        }
    }
}

impl<S: ?Sized + TryStream> LanceTryStreamExt for S {}
