use std::ops::Range;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BooleanArray};
use arrow_array::cast::as_boolean_array;
use arrow_buffer::{bit_util, Buffer};
use arrow_data::ArrayDataBuilder;
use arrow_schema::DataType;
use async_trait::async_trait;
use object_store::path::Path;
use tokio::io::AsyncWriteExt;
use crate::encodings::Decoder;

use crate::error::Result;
use crate::io::object_reader::ObjectReader;
use crate::io::object_writer::ObjectWriter;
use crate::io::ObjectStore;

// Because Boolean type is special, it cannot be a PlainEncoder/PlainDecoder

pub struct BooleanEncoder<'a> {
    writer: &'a mut ObjectWriter<'a>,
}

impl<'a> BooleanEncoder<'a> {
    pub fn new(writer: &'a mut ObjectWriter<'a>) -> BooleanEncoder<'a> {
        BooleanEncoder {
            writer,
        }
    }

    /// Encode an array of a batch.
    /// Returns the offset of the metadata
    pub async fn encode(&mut self, array: &dyn Array) -> Result<usize> {
        let offset = self.writer.tell() as usize;

        let data = array.data().buffers()[0].as_slice();
        self.writer.write_all(data).await?;

        Ok(offset)
    }
}

/// Decoder for plain encoding.
pub struct BooleanDecoder<'a> {
    reader: &'a ObjectReader<'a>,
    /// The start position of the batch in the file.
    position: usize,
    /// Number of the rows in this batch.
    length: usize,
}

impl<'a> BooleanDecoder<'a> {
    pub fn new(
        reader: &'a ObjectReader,
        position: usize,
        length: usize,
    ) -> Result<BooleanDecoder<'a>> {
        Ok(BooleanDecoder {
            reader,
            position,
            length
        })
    }

    pub async fn at(&self, _idx: usize) -> Result<Option<bool>> {
        todo!()
    }
}

#[async_trait]
impl<'a> Decoder for BooleanDecoder<'a> {
    async fn decode(&self) -> Result<ArrayRef> {
        // The Rust bit_util is different. In C++ there's (bits >> 3) + ((bits & 7) != 0)
        let array_bytes = bit_util::ceil(self.length, 8);
        let range = Range {
            start: self.position,
            end: self.position + array_bytes,
        };

        let data = self.reader.get_range(range).await?;
        let buf: Buffer = data.into();
        let array_data = ArrayDataBuilder::new(DataType::Boolean)
            .len(self.length)
            .null_count(0)
            .add_buffer(buf)
            .build()?;
        Ok(Arc::new(BooleanArray::from(array_data)))
    }
}


#[tokio::test]
async fn test_encode_decode_bool_array() {
    let store = ObjectStore::new(":memory:").unwrap();
    let path = Path::from("/foo");
    let (_, mut writer) = store.inner.put_multipart(&path).await.unwrap();

    let arr = BooleanArray::from(vec![true, false].repeat(100));
    {
        let mut object_writer = ObjectWriter::new(writer.as_mut());
        let mut encoder = BooleanEncoder::new(&mut object_writer);

        assert_eq!(encoder.encode(&arr).await.unwrap(), 0);
    }
    writer.shutdown().await.unwrap();

    let mut reader = store.open(&path).await.unwrap();
    assert!(reader.size().await.unwrap() > 0);
    let decoder = BooleanDecoder::new(&reader, 0, arr.len()).unwrap();
    let read_arr = decoder.decode().await.unwrap();
    let expect_arr = as_boolean_array(read_arr.as_ref());
    assert_eq!(expect_arr, &arr);
}