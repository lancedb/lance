//! I/Os

use prost::Message;
use std::io::Result;
use tokio::io::{AsyncWrite, AsyncWriteExt};

/// Write protobuf to an open writer.
pub async fn write_proto(msg: &impl Message, writer: &mut (impl AsyncWrite + Unpin)) -> Result<()> {
    let len = msg.encoded_len();

    writer.write_u32_le(len as u32).await?;
    writer.write_all(msg.encode_to_vec().as_slice()).await?;

    Ok(())
}
