use crate::format::pb;
use arrow2::error::Error;

pub struct Metadata {
    pb: pb::Metadata,
}

impl Metadata {
    pub fn make(pb: pb::Metadata) -> Self {
        Self {
            pb
        }
    }

    pub fn num_chunks(&self) -> usize {
        self.pb.batch_offsets.len() - 1
    }

    pub fn num_batches(&self) -> usize {
        self.pb.batch_offsets.len() - 1
    }

    pub fn length(&self) -> i32 {
        // Metadata::length
        // TODO need a link to the design doc
        match self.pb.batch_offsets.last() {
            None => { 0 }
            Some(v) => *v
        }
    }

    pub fn locate_batch(&self, row_index: i32) -> Result<(i32, i32), Error> {

        // Metadata::LocateBatch
        let len = self.length();

        if row_index < 0 || row_index >= len {
            return Err(Error::InvalidArgumentError(format!("Row index out of range: {} of {}", row_index, len)));
        }

        let bound_idx = self.pb.batch_offsets.partition_point(|x| { x <= &row_index }) as i32 - 1;

        if bound_idx <= 0 {
            return Err(Error::InvalidArgumentError(format!("metadata is not valid {:?}", self.pb.batch_offsets)));
        }

        let offset = row_index - self.pb.batch_offsets[bound_idx as usize];

        Ok((bound_idx, offset))
    }
}