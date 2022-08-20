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

use std::io::{Cursor, Error, ErrorKind, Read, Result, Seek, SeekFrom};

use byteorder::{LittleEndian, ReadBytesExt};

use crate::format::pb;
use crate::schema::Schema;

static MAGIC_NUMBER: &str = "LANC";

/// Lance File Reader.
pub struct FileReader<R: Read + Seek> {
    file: R,
    schema: Schema,
    // TODO: impl a Metadata
    metadata: pb::Metadata,
}

trait ProtoReader<P: prost::Message + Default> {
    fn read<R: Read + Seek>(file: &mut R, pos: i64) -> Result<P>;
}

struct ProtoParser;

impl<P: prost::Message + Default> ProtoReader<P> for ProtoParser {
    fn read<R: Read + Seek>(file: &mut R, pos: i64) -> Result<P> {
        let mut size_buf = [0; 4];
        file.seek(SeekFrom::Start(pos as u64))?;
        file.read(&mut size_buf)?;
        let pb_size = Cursor::new(size_buf).read_i32::<LittleEndian>()?;
        let mut buf = Vec::with_capacity(pb_size as usize);
        buf.resize(pb_size as usize, 0);
        file.read(&mut buf)?;
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
        let metadata: crate::format::pb::Metadata = ProtoParser::read(&mut f, metadata_pos)?;
        let manifest: crate::format::pb::Manifest =
            ProtoParser::read(&mut f, metadata.manifest_position as i64)?;
        Ok(FileReader {
            file: f,
            schema: Schema::new(&manifest.fields),
            metadata,
        })
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    pub fn num_chunks(&self) -> i32 {
        self.metadata.batch_offsets.len() as i32 - 1
    }
}
