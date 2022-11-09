//  Copyright 2022 Lance Authors
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::io::{Cursor, Error, ErrorKind, Read, Result, Seek, SeekFrom};
use arrow2::array::{Array, DictionaryKey};
use arrow2::datatypes::DataType;
use arrow2::scalar::{Scalar, StructScalar, Utf8Scalar};
use arrow2::types::Offset;
use arrow2::types::Index;

use byteorder::{LittleEndian, ReadBytesExt};

use crate::metadata::Metadata;
use crate::page_table::PageTable;
use crate::schema::{Field, Schema};

static MAGIC_NUMBER: &str = "LANC";

/// Lance File Reader.
pub struct FileReader<R: Read + Seek> {
    file: R,
    schema: Schema,
    metadata: Metadata,
    page_table: PageTable,
}

trait ProtoReader<P: prost::Message + Default> {
    fn read<R: Read + Seek>(file: &mut R, pos: i64) -> Result<P>;
}

struct ArrayParams {
    offset: u32,
    len: Option<u32>,
}

struct ProtoParser;

impl<P: prost::Message + Default> ProtoReader<P> for ProtoParser {
    fn read<R: Read + Seek>(file: &mut R, pos: i64) -> Result<P> {
        let mut size_buf = [0; 4];
        file.seek(SeekFrom::Start(pos as u64))?;
        file.read_exact(&mut size_buf)?;
        let pb_size = Cursor::new(size_buf).read_i32::<LittleEndian>()?;
        let mut buf = vec![0; pb_size as usize];
        file.read_exact(&mut buf)?;
        match P::decode(&buf[..]) {
            Ok(m) => Ok(m),
            Err(e) => Err(Error::new(
                ErrorKind::InvalidData,
                "Invalid metadata: ".to_owned() + &e.to_string(),
            )),
        }
    }
}

fn read_footer<R: Read + Seek>(file: &mut R) -> Result<i64> {
    file.seek(SeekFrom::End(-16))?;
    let mut buf = [0; 16];
    let nbytes = file.read(&mut buf)?;
    if nbytes < 16 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Not a lance file: size < 16 bytes",
        ));
    }
    let s = match std::str::from_utf8(&buf[12..16]) {
        Ok(s) => s,
        Err(_) => return Err(Error::new(ErrorKind::InvalidData, "Not a lance file")),
    };
    if !s.eq(MAGIC_NUMBER) {
        return Err(Error::new(ErrorKind::InvalidData, "Not a lance file"));
    }
    // TODO: check version

    Cursor::new(&buf[0..8]).read_i64::<LittleEndian>()
}

impl<R: Read + Seek> FileReader<R> {
    pub fn new(file: R) -> Result<Self> {
        let mut f = file;
        let metadata_pos = read_footer(&mut f)?;
        let pb_meta: crate::format::pb::Metadata = ProtoParser::read(&mut f, metadata_pos)?;
        let manifest: crate::format::pb::Manifest =
            ProtoParser::read(&mut f, pb_meta.manifest_position as i64)?;
        let num_columns = manifest.fields.len();
        let num_batches = pb_meta.batch_offsets.len() - 1;
        let page_table = PageTable::make(&mut f, pb_meta.page_table_position, num_columns, num_batches);
        let metadata = Metadata::make(pb_meta);
        Ok(FileReader {
            file: f,
            schema: Schema::from_proto(&manifest.fields),
            metadata,
            page_table,
        })
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    pub fn num_chunks(&self) -> usize {
        self.metadata.num_chunks()
    }

    pub fn get(&self, idx: u32) -> Vec<Box<dyn Scalar>> {
        // TODO usize for idiomatic ? or i32 for protobuf compatibility?
        // FileReader::Get
        let schema = self.schema();
        let (batch_id, idx_in_batch) = self.metadata.locate_batch(idx as i32).unwrap();
        let mut res = Vec::new();
        for field in &schema.fields {
            let num_batches = self.metadata.num_batches();
            let v = self.get_scalar(field, batch_id, idx_in_batch);
            res.push(v)
        }
        return res;
    }


    fn get_array(field: &Field, batch_id: usize, array_params: ArrayParams) -> Box<dyn Array> {
        // TODO
        // let d_type = field.data_type();
        // let storage_type = field.storage_type();
        // let storage_array: Box<dyn Array> = match storage_type {
        //     DataType::List(_) => { get_list_array(field, batch_id, &array_params) }
        //     DataType::Struct(_) => { get_struct_array(field, batch_id, &array_params) }
        //     DataType::Dictionary(_, _, _) => { get_dictionary_array(field, batch_id, &array_params) }
        //     _ => {
        //         get_primitive_array(field, batch_id, &array_params)
        //     }
        // };
        //
        // storage_array
        todo!()
    }

    fn get_list_array(field: &Field, batch_id: usize, array_params: &ArrayParams) -> Box<dyn Array> {
        todo!()
    }

    fn get_struct_array(field: &Field, batch_id: usize, array_params: &ArrayParams) -> Box<dyn Array> {
        todo!()
    }

    fn get_dictionary_array(field: &Field, batch_id: usize, array_params: &ArrayParams) -> Box<dyn Array> {
        todo!()
    }

    fn get_primitive_array(&self, field: &Field, batch_id: usize, array_params: &ArrayParams) -> Box<dyn Array> {
        let field_id = field.id.to_usize();
        let page_info = self.page_table.get_page_info(field_id, batch_id);
        // field.get_decoder(&self.file);
        todo!()
    }

    fn get_scalar(&self, field: &Field, batch_id: i32, idx_in_batch: i32) -> Box<dyn arrow2::scalar::Scalar> {
        //FileReader::GetScalar
        match field.data_type() {
            DataType::Struct(_) => {
                Box::new(Utf8Scalar::<i32>::new(Some("struct not support yet")))
            }
            DataType::List(_) => {
                Box::new(Utf8Scalar::<i32>::new(Some("list not support yet")))
            }
            DataType::Extension(t, _, Some(x)) if t == "not_supported_yet" => {
                Box::new(Utf8Scalar::<i32>::new(Some(format!("data_type {:} not supported yet", x))))
            }
            x => {
                self.get_primitive_scalar(field, batch_id, idx_in_batch)
            }
        }
    }

    fn get_struct_scalar(&self, field: &Field, batch_id: i32, idx_in_batch: i32) -> Box<dyn Scalar> {
        //FileReader::GetStructScalar
        let mut values = Vec::new();
        for child in field.fields() {
            let scalar = self.get_scalar(child, batch_id, idx_in_batch);
            values.push(scalar)
        }
        return Box::new(StructScalar::new(field.data_type(), values.into()));
    }

    fn get_list_scalar(&self, field: &Field, batch_id: i32, idx_in_batch: i32) -> Box<dyn Scalar> {
        //FileReader::GetListScalar
        // auto field_id = field->id();
        let field_id = field.id;
        let page_info = self.page_table.get_page_info(field_id as usize, batch_id as usize);
        // ARROW_ASSIGN_OR_RAISE(auto page, GetPageInfo(field_id, batch_id));
        // auto [pos, length] = page;
        // decoder->Reset(pos, length);
        // ARROW_ASSIGN_OR_RAISE(auto decoder, field->GetDecoder(file_));
        let mut decoder = field.get_decoder(&self.file, page_info);

        let val = decoder.decode(idx_in_batch, &Some(2)).unwrap();
        // ARROW_ASSIGN_OR_RAISE(auto offsets_arr, decoder->ToArray(idx, 2));
        // auto offsets = std::static_pointer_cast<::arrow::Int32Array>(offsets_arr);
        // if (offsets->Value(0) == offsets->Value(1)) {
        //     return std::make_shared<::arrow::NullScalar>();
        // }
        // ARROW_ASSIGN_OR_RAISE(
        //     auto values,
        //     GetArray(field->fields()[0],
        //              batch_id,
        //              ArrayReadParams(offsets->Value(0), offsets->Value(1) - offsets->Value(0))));
        // return std::make_shared<::arrow::ListScalar>(values);
        todo!()
    }

    fn get_primitive_scalar(&self, field: &Field, batch_id: i32, idx_in_batch: i32) -> Box<dyn Scalar> {
        //FileReader::GetPrimitiveScalar
        let data_type = field.data_type();
        let field_id = field.id;
        let page_info = self.page_table.get_page_info(field_id as usize, batch_id as usize);

        let mut decoder = field.get_decoder(&self.file, page_info);
        return decoder.value(idx_in_batch as usize).unwrap();
    }
}
