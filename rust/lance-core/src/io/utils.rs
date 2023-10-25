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

use std::cmp::min;

use arrow_array::{
    types::{BinaryType, LargeBinaryType, LargeUtf8Type, Utf8Type},
    ArrayRef,
};
use arrow_schema::DataType;
use byteorder::{ByteOrder, LittleEndian};
use lance_arrow::*;
use prost::Message;
use snafu::{location, Location};

use crate::encodings::{
    binary::{BinaryDecoder, BinaryEncoder},
    plain::{PlainDecoder, PlainEncoder},
    AsyncIndex, Decoder, Encoder,
};
use crate::format::{pb, Index, Manifest, ProtoStruct};
use crate::io::{ReadBatchParams, Reader, WriteExt, Writer};
use crate::{Error, Result};

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
            return Err(Error::IO {
                message: format!("Unsupported binary type: {data_type}",),
                location: location!(),
            })
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
        return Err(Error::IO {
            message: "file size is too small".to_string(),
            location: location!(),
        });
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
    T: ProtoStruct<Proto = M> + From<M>,
>(
    reader: &dyn Reader,
    pos: usize,
) -> Result<T> {
    let msg = read_message::<M>(reader, pos).await?;
    let obj = T::from(msg);
    Ok(obj)
}

/// Write manifest to an open file.
pub async fn write_manifest(
    writer: &mut dyn Writer,
    manifest: &mut Manifest,
    indices: Option<Vec<Index>>,
) -> Result<usize> {
    // Write dictionary values.
    let max_field_id = manifest.schema.max_field_id().unwrap_or(-1);
    for field_id in 0..max_field_id + 1 {
        if let Some(field) = manifest.schema.mut_field_by_id(field_id) {
            if field.data_type().is_dictionary() {
                let dict_info = field.dictionary.as_mut().ok_or_else(|| Error::IO {
                    message: format!("Lance field {} misses dictionary info", field.name),
                    location: location!(),
                })?;

                let value_arr = dict_info.values.as_ref().ok_or_else(|| Error::IO {
                    message: format!(
                        "Lance field {} is dictionary type, but misses the dictionary value array",
                        field.name
                    ),
                    location: location!(),
                })?;

                let data_type = value_arr.data_type();
                let pos = match data_type {
                    dt if dt.is_numeric() => {
                        let mut encoder = PlainEncoder::new(writer, dt);
                        encoder.encode(&[value_arr]).await?
                    }
                    dt if dt.is_binary_like() => {
                        let mut encoder = BinaryEncoder::new(writer);
                        encoder.encode(&[value_arr]).await?
                    }
                    _ => {
                        return Err(Error::IO {
                            message: format!(
                                "Does not support {} as dictionary value type",
                                value_arr.data_type()
                            ),
                            location: location!(),
                        });
                    }
                };
                dict_info.offset = pos;
                dict_info.length = value_arr.len();
            }
        }
    }

    // Write indices if presented.
    if let Some(indices) = indices.as_ref() {
        let section = pb::IndexSection {
            indices: indices.iter().map(|i| i.into()).collect(),
        };
        let pos = writer.write_protobuf(&section).await?;
        manifest.index_section = Some(pos);
    }

    writer.write_struct(manifest).await
}
