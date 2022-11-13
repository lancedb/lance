use std::io::{Read, Seek, SeekFrom};

use arrow2::datatypes::DataType;
use byteorder::{LittleEndian, ReadBytesExt};

#[derive(Debug, Clone)]
pub struct PageInfo {
    pub position: i64,
    pub length: i64,
}

#[derive(Debug, Clone)]
pub struct PageTable {
    page_info_map: Vec<Vec<PageInfo>>,
}

impl PageTable {
    pub fn new<R: Read + Seek>(
        file: &mut R,
        page_table_position: u64,
        num_columns: usize,
        num_batches: usize,
    ) -> PageTable {
        file.seek(SeekFrom::Start(page_table_position)).unwrap();

        let mut vec = vec![0i64; num_columns * num_batches * 2];
        file.read_i64_into::<LittleEndian>(&mut vec).unwrap(); //TODO is it right?
        let buffer = arrow2::buffer::Buffer::from(vec);
        let arr = arrow2::array::Int64Array::new(DataType::Int64, buffer, None);
        let mut lt = PageTable {
            page_info_map: Vec::new(),
        };

        //a replacement of PageTable::SetPageInfo in C++
        for col in 0..num_columns {
            let mut a_col = Vec::new();
            for batch in 0..num_batches {
                let idx = col * num_batches + batch;
                let position = arr.value(idx * 2);
                let length = arr.value(idx * 2 + 1);
                a_col.push(PageInfo { position, length });
            }
            lt.page_info_map.push(a_col);
        }
        lt
    }

    pub fn get_page_info(&self, column_id: usize, batch_id: usize) -> PageInfo {
        self.page_info_map[column_id][batch_id].clone()
    }
}
