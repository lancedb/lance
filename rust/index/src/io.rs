//! I/Os

use prost::Message;
use std::io::Result;
use tokio::io::{AsyncWrite, AsyncWriteExt};
use async_trait::async_trait;


#[async_trait]
pub trait AsyncWriteProtoExt {
    async fn write_pb(&mut self, msg: impl Message) -> Result<u64>;

    /// Write footer with the offset to the root metadata block.
    async fn write_footer(&mut self, offset: u64) -> Result<()>;
}

#[async_trait]
impl<T: AsyncWrite + Unpin + std::marker::Send> AsyncWriteProtoExt for T {
    async fn write_pb(&mut self, msg: impl Message) -> Result<u64> {
        let len = msg.encoded_len();

        self.write_u32_le(len as u32).await?;
        self.write_all(&msg.encode_to_vec()).await?;
        Ok(0)
    }

    async fn write_footer(&mut self, offset: u64) -> Result<()> {
        self.write_u64_le(offset).await?;
        self.write_all(b"LANCEIDX").await?;
        Ok(())
    }
}
