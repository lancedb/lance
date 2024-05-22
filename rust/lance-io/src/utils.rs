// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::cmp::min;

use arrow_array::{
    types::{BinaryType, LargeBinaryType, LargeUtf8Type, Utf8Type},
    ArrayRef,
};
use arrow_schema::DataType;
use byteorder::{ByteOrder, LittleEndian};
use bytes::Bytes;
use lance_arrow::*;
use prost::Message;
use snafu::{location, Location};

use crate::{
    encodings::{binary::BinaryDecoder, plain::PlainDecoder, AsyncIndex, Decoder},
    traits::ProtoStruct,
};
use crate::{traits::Reader, ReadBatchParams};
use lance_core::{Error, Result};

/// Read a binary array from a [Reader].
///
pub async fn read_binary_array(
    reader: &dyn Reader,
    data_type: &DataType,
    nullable: bool,
    position: usize,
    length: usize,
    params: impl Into<ReadBatchParams>,
) -> Result<ArrayRef> {
    use arrow_schema::DataType::*;
    let decoder: Box<dyn Decoder<Output = Result<ArrayRef>> + Send> = match data_type {
        Utf8 => Box::new(BinaryDecoder::<Utf8Type>::new(
            reader, position, length, nullable,
        )),
        Binary => Box::new(BinaryDecoder::<BinaryType>::new(
            reader, position, length, nullable,
        )),
        LargeUtf8 => Box::new(BinaryDecoder::<LargeUtf8Type>::new(
            reader, position, length, nullable,
        )),
        LargeBinary => Box::new(BinaryDecoder::<LargeBinaryType>::new(
            reader, position, length, nullable,
        )),
        _ => {
            return Err(Error::io(
                format!("Unsupported binary type: {}", data_type),
                location!(),
            ));
        }
    };
    let fut = decoder.as_ref().get(params.into());
    fut.await
}

/// Read a fixed stride array from disk.
///
pub async fn read_fixed_stride_array(
    reader: &dyn Reader,
    data_type: &DataType,
    position: usize,
    length: usize,
    params: impl Into<ReadBatchParams>,
) -> Result<ArrayRef> {
    if !data_type.is_fixed_stride() {
        return Err(Error::Schema {
            message: format!("{data_type} is not a fixed stride type"),
            location: location!(),
        });
    }
    // TODO: support more than plain encoding here.
    let decoder = PlainDecoder::new(reader, data_type, position, length)?;
    decoder.get(params.into()).await
}

/// Read a protobuf message at file position 'pos'.
// TODO: pub(crate)
pub async fn read_message<M: Message + Default>(reader: &dyn Reader, pos: usize) -> Result<M> {
    let file_size = reader.size().await?;
    if pos > file_size {
        return Err(Error::io("file size is too small".to_string(), location!()));
    }

    let range = pos..min(pos + 4096, file_size);
    let buf = reader.get_range(range.clone()).await?;
    let msg_len = LittleEndian::read_u32(&buf) as usize;

    if msg_len + 4 > buf.len() {
        let remaining_range = range.end..min(4 + pos + msg_len, file_size);
        let remaining_bytes = reader.get_range(remaining_range).await?;
        let buf = [buf, remaining_bytes].concat();
        assert!(buf.len() >= msg_len + 4);
        Ok(M::decode(&buf[4..4 + msg_len])?)
    } else {
        Ok(M::decode(&buf[4..4 + msg_len])?)
    }
}

/// Read a Protobuf-backed struct at file position: `pos`.
// TODO: pub(crate)
pub async fn read_struct<
    'm,
    M: Message + Default + 'static,
    T: ProtoStruct<Proto = M> + TryFrom<M, Error = Error>,
>(
    reader: &dyn Reader,
    pos: usize,
) -> Result<T> {
    let msg = read_message::<M>(reader, pos).await?;
    T::try_from(msg)
}

pub async fn read_last_block(reader: &dyn Reader) -> Result<Bytes> {
    let file_size = reader.size().await?;
    let block_size = reader.block_size();
    let begin = if file_size < block_size {
        0
    } else {
        file_size - block_size
    };
    reader.get_range(begin..file_size).await
}

pub fn read_metadata_offset(bytes: &Bytes) -> Result<usize> {
    let len = bytes.len();
    if len < 16 {
        return Err(Error::io(
            format!(
                "does not have sufficient data, len: {}, bytes: {:?}",
                len, bytes
            ),
            location!(),
        ));
    }
    let offset_bytes = bytes.slice(len - 16..len - 8);
    Ok(LittleEndian::read_u64(offset_bytes.as_ref()) as usize)
}

/// Read the version from the footer bytes
pub fn read_version(bytes: &Bytes) -> Result<(u16, u16)> {
    let len = bytes.len();
    if len < 8 {
        return Err(Error::io(
            format!(
                "does not have sufficient data, len: {}, bytes: {:?}",
                len, bytes
            ),
            location!(),
        ));
    }

    let major_version = LittleEndian::read_u16(bytes.slice(len - 8..len - 6).as_ref());
    let minor_version = LittleEndian::read_u16(bytes.slice(len - 6..len - 4).as_ref());
    Ok((major_version, minor_version))
}

/// Read protobuf from a buffer.
pub fn read_message_from_buf<M: Message + Default>(buf: &Bytes) -> Result<M> {
    let msg_len = LittleEndian::read_u32(buf) as usize;
    Ok(M::decode(&buf[4..4 + msg_len])?)
}

/// Read a Protobuf-backed struct from a buffer.
pub fn read_struct_from_buf<
    M: Message + Default,
    T: ProtoStruct<Proto = M> + TryFrom<M, Error = Error>,
>(
    buf: &Bytes,
) -> Result<T> {
    let msg: M = read_message_from_buf(buf)?;
    T::try_from(msg)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use object_store::{memory::InMemory, path::Path};

    use crate::{
        object_reader::CloudObjectReader,
        object_writer::ObjectWriter,
        traits::{ProtoStruct, WriteExt, Writer},
        utils::read_struct,
        Error, Result,
    };

    // Bytes is a prost::Message, since we don't have any .proto files in this crate we
    // can use it to simulate a real message object.
    #[derive(Debug, PartialEq)]
    struct BytesWrapper(Bytes);

    impl ProtoStruct for BytesWrapper {
        type Proto = Bytes;
    }

    impl From<&BytesWrapper> for Bytes {
        fn from(value: &BytesWrapper) -> Self {
            value.0.clone()
        }
    }

    impl TryFrom<Bytes> for BytesWrapper {
        type Error = Error;
        fn try_from(value: Bytes) -> Result<Self> {
            Ok(Self(value))
        }
    }

    #[tokio::test]
    async fn test_write_proto_structs() {
        let store = InMemory::new();
        let path = Path::from("/foo");

        let mut object_writer = ObjectWriter::new(&store, &path).await.unwrap();
        assert_eq!(object_writer.tell().await.unwrap(), 0);

        let some_message = BytesWrapper(Bytes::from(vec![10, 20, 30]));

        let pos = object_writer.write_struct(&some_message).await.unwrap();
        assert_eq!(pos, 0);
        object_writer.shutdown().await.unwrap();

        let object_reader = CloudObjectReader::new(Arc::new(store), path, 1024).unwrap();
        let actual: BytesWrapper = read_struct(&object_reader, pos).await.unwrap();
        assert_eq!(some_message, actual);
    }
}
