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

use std::io::{Error, ErrorKind, Read, Result, Seek, SeekFrom};

use crate::schema::Schema;

static MAGIC_NUMBER: &str = "LANC";

/// Lance File Reader.
pub struct FileReader<R: Read + Seek> {
    file: R,
    schema: Schema,
}

impl<R: Read + Seek> FileReader<R> {
    pub fn new(file: R) -> Result<Self> {
        let mut reader = FileReader {
            file,
            schema: Schema {},
        };
        let metadata_pos = reader.read_footer()?;

        match reader.open() {
            Ok(_) => Ok(reader),
            Err(e) => Err(e),
        }
    }

    // Open the file reader
    fn open(&mut self) -> Result<()> {
        Ok(())
    }

    fn read_footer(&mut self) -> Result<i64> {
        self.file.seek(SeekFrom::End(-16))?;
        let mut buf = [0; 16];
        let nbytes = self.file.read(&mut buf)?;
        if nbytes < 16 {
            return Err(Error::new(ErrorKind::InvalidData, "Not a lance file: size < 16 bytes"));
        }
        let s = match std::str::from_utf8(&buf[12..16]) {
            Ok(s) => s,
            Err(_) => return Err(Error::new(ErrorKind::InvalidData, "Not a lance file")),
        };
        if !s.eq(MAGIC_NUMBER) {
            return Err(Error::new(ErrorKind::InvalidData, "Not a lance file"));
        }
        Ok(0)
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}
