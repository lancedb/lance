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
        // int64_t len = length();

        if row_index < 0 || row_index >= len {
            return Err(Error::InvalidArgumentError(format!("Row index out of range: {} of {}", row_index, len)));
        }
        // if (row_index < 0 || row_index >= len) {
        //     return ::arrow::Status::IndexError(fmt::format("Row index out of range: {} of {}", row_index, len));
        // }

        // auto it = std::upper_bound(pb_.batch_offsets().begin(), pb_.batch_offsets().end(), row_index);
        // if (it == pb_.batch_offsets().end()) {
        //     return ::arrow::Status::IndexError("Row index out of range {} of {}", row_index, len);
        // }
        // int32_t bound_idx = std::distance(pb_.batch_offsets().begin(), it);
        // assert(bound_idx >= 0);
        // bound_idx = std::max(0, bound_idx - 1);
        // // Offset within the batch.
        // int32_t offset = row_index - pb_.batch_offsets(bound_idx);
        // return std::tuple(bound_idx, offset);

        todo!()
    }
}