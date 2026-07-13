// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;

use async_trait::async_trait;
use bytes::Bytes;
use futures::{StreamExt, future::BoxFuture, stream::BoxStream};
use lance_core::deepsize::DeepSizeOf;
use object_store::path::Path;
use prost::Message;
use tokio::io::{AsyncWrite, AsyncWriteExt};

use lance_core::{Error, Result};

use crate::object_writer::WriteResult;

pub trait ProtoStruct {
    type Proto: Message;
}

pub type ByteStream = BoxStream<'static, object_store::Result<Bytes>>;

fn checked_protobuf_wire_len(len: usize) -> Result<u32> {
    u32::try_from(len).map_err(|_| {
        Error::format_capacity_exceeded(format!(
            "protobuf payload is {len} bytes but the Lance framing limit is {} bytes",
            u32::MAX
        ))
    })
}

/// A trait for writing to a file on local file system or object store.
#[async_trait]
pub trait Writer: AsyncWrite + Unpin + Send {
    /// Tell the current offset.
    async fn tell(&mut self) -> Result<usize>;

    /// Flush all buffered data and finalize the write, returning metadata about
    /// the written object.
    async fn shutdown(&mut self) -> Result<WriteResult>;
}

#[async_trait]
impl Writer for Box<dyn Writer> {
    async fn tell(&mut self) -> Result<usize> {
        self.as_mut().tell().await
    }

    async fn shutdown(&mut self) -> Result<WriteResult> {
        self.as_mut().shutdown().await
    }
}

/// Lance Write Extension.
#[async_trait]
pub trait WriteExt {
    /// Write a Protobuf message to the [Writer], and returns the file position
    /// where the protobuf is written.
    async fn write_protobuf(&mut self, msg: &impl Message) -> Result<usize>;

    async fn write_struct<
        'b,
        M: Message + From<&'b T>,
        T: ProtoStruct<Proto = M> + Send + Sync + 'b,
    >(
        &mut self,
        obj: &'b T,
    ) -> Result<usize> {
        let msg: M = M::from(obj);
        self.write_protobuf(&msg).await
    }
    /// Write magics to the tail of a file before closing the file.
    async fn write_magics(
        &mut self,
        pos: usize,
        major_version: i16,
        minor_version: i16,
        magic: &[u8],
    ) -> Result<()>;

    async fn copy_from_reader(&mut self, reader: &dyn Reader) -> Result<usize>;

    async fn copy_range_from_reader(
        &mut self,
        reader: &dyn Reader,
        range: Range<usize>,
    ) -> Result<usize>;
}

#[async_trait]
impl<W: Writer + ?Sized> WriteExt for W {
    async fn write_protobuf(&mut self, msg: &impl Message) -> Result<usize> {
        let offset = self.tell().await?;

        let len = msg.encoded_len();
        let wire_len = checked_protobuf_wire_len(len)?;

        self.write_u32_le(wire_len).await?;
        self.write_all(&msg.encode_to_vec()).await?;

        Ok(offset)
    }

    async fn write_magics(
        &mut self,
        pos: usize,
        major_version: i16,
        minor_version: i16,
        magic: &[u8],
    ) -> Result<()> {
        self.write_i64_le(pos as i64).await?;
        self.write_i16_le(major_version).await?;
        self.write_i16_le(minor_version).await?;
        self.write_all(magic).await?;
        Ok(())
    }

    async fn copy_from_reader(&mut self, reader: &dyn Reader) -> Result<usize> {
        let mut stream = reader.get_stream().await?;
        let mut copied = 0usize;
        while let Some(chunk) = stream.next().await {
            let bytes = chunk?;
            copied += bytes.len();
            self.write_all(&bytes).await?;
        }
        Ok(copied)
    }

    async fn copy_range_from_reader(
        &mut self,
        reader: &dyn Reader,
        range: Range<usize>,
    ) -> Result<usize> {
        let mut stream = reader.get_range_stream(range).await?;
        let mut copied = 0usize;
        while let Some(chunk) = stream.next().await {
            let bytes = chunk?;
            copied += bytes.len();
            self.write_all(&bytes).await?;
        }
        Ok(copied)
    }
}

pub trait Reader: std::fmt::Debug + Send + Sync + DeepSizeOf {
    fn path(&self) -> &Path;

    /// Suggest optimal I/O size per storage device.
    fn block_size(&self) -> usize;

    /// Suggest optimal I/O parallelism per storage device.
    fn io_parallelism(&self) -> usize;

    /// Object/File Size.
    fn size(&self) -> BoxFuture<'_, object_store::Result<usize>>;

    /// Read a range of bytes from the object.
    ///
    /// TODO: change to read_at()?
    fn get_range(&self, range: Range<usize>) -> BoxFuture<'static, object_store::Result<Bytes>>;

    /// Read all bytes from the object.
    ///
    /// By default this reads the size in a separate IOP but some implementations
    /// may not need the size beforehand.
    fn get_all(&self) -> BoxFuture<'_, object_store::Result<Bytes>>;

    /// Read the entire object as a byte stream.
    fn get_stream(&self) -> BoxFuture<'_, object_store::Result<ByteStream>> {
        Box::pin(async move {
            let bytes = self.get_all().await?;
            Ok(futures::stream::once(async move { Ok(bytes) }).boxed())
        })
    }

    /// Read a byte range as a byte stream.
    fn get_range_stream(
        &self,
        range: Range<usize>,
    ) -> BoxFuture<'_, object_store::Result<ByteStream>> {
        Box::pin(async move {
            let bytes = self.get_range(range).await?;
            Ok(futures::stream::once(async move { Ok(bytes) }).boxed())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::checked_protobuf_wire_len;
    use lance_core::error::FormatCapacityExceeded;

    #[test]
    fn protobuf_wire_length_rejects_values_above_u32() {
        assert_eq!(
            checked_protobuf_wire_len(u32::MAX as usize).unwrap(),
            u32::MAX
        );
        if let Ok(too_large) = usize::try_from(u64::from(u32::MAX) + 1) {
            let error = checked_protobuf_wire_len(too_large).unwrap_err();
            assert!(error.to_string().contains("FormatCapacityExceeded"));
            let lance_core::Error::InvalidInput { source, .. } = error else {
                panic!("structural capacity must use the InvalidInput boundary")
            };
            assert!(source.downcast_ref::<FormatCapacityExceeded>().is_some());
        }
    }
}
