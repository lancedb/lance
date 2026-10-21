// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

// Arrow only implements AsRef<[u8]> for borrowed Buffer. Keep ownership while
// exposing its bytes to Bytes::from_owner, so query views need no payload copy.
struct PositionBuffer(Buffer);

impl AsRef<[u8]> for PositionBuffer {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

/// Compressed position rows retained without a cache entry or allocation per term.
#[derive(Debug, Clone)]
pub(in super::super) struct PositionGroup {
    pub(in super::super) batch: RecordBatch,
    pub(in super::super) codec: PositionStreamCodec,
    bytes: bytes::Bytes,
    offsets: ScalarBuffer<u32>,
}

impl PositionGroup {
    pub(in super::super) fn new(batch: RecordBatch, codec: PositionStreamCodec) -> Result<Self> {
        let payload = batch
            .column_by_name(COMPRESSED_POSITION_COL)
            .and_then(|column| column.as_binary_opt::<i64>())
            .ok_or_else(|| Error::index("position group requires LargeBinary payloads"))?;
        let offsets = batch
            .column_by_name(POSITION_BLOCK_OFFSET_COL)
            .and_then(|column| column.as_list_opt::<i32>())
            .ok_or_else(|| Error::index("position group requires List<UInt32> offsets"))?;
        let values = offsets
            .values()
            .as_primitive_opt::<UInt32Type>()
            .ok_or_else(|| Error::index("position group offsets must contain UInt32 values"))?;
        if payload.null_count() != 0 || offsets.null_count() != 0 || values.null_count() != 0 {
            return Err(Error::index("position group must not contain nulls"));
        }
        for slot in 0..batch.num_rows() {
            let start = offsets.value_offsets()[slot] as usize;
            let end = offsets.value_offsets()[slot + 1] as usize;
            let row_offsets = &values.values()[start..end];
            let payload_len = payload.value_length(slot) as usize;
            if row_offsets.first().is_some_and(|offset| *offset != 0)
                || row_offsets.windows(2).any(|pair| pair[0] > pair[1])
                || row_offsets
                    .last()
                    .is_some_and(|offset| *offset as usize > payload_len)
                || (row_offsets.is_empty() && payload_len != 0)
            {
                return Err(Error::index(format!(
                    "invalid position group offsets for slot {slot}, payload length {payload_len}"
                )));
            }
        }
        let bytes = bytes::Bytes::from_owner(PositionBuffer(payload.values().clone()));
        let offsets = values.values().clone();
        Ok(Self {
            batch,
            codec,
            bytes,
            offsets,
        })
    }

    pub(super) fn position(&self, slot: usize) -> Result<CompressedPositionStorage> {
        if slot >= self.batch.num_rows() {
            return Err(Error::index(format!(
                "position slot {slot} exceeds group length {}",
                self.batch.num_rows()
            )));
        }
        let payload = self.batch[COMPRESSED_POSITION_COL].as_binary::<i64>();
        let offsets = self.batch[POSITION_BLOCK_OFFSET_COL].as_list::<i32>();
        let byte_start = payload.value_offsets()[slot] as usize;
        let byte_end = payload.value_offsets()[slot + 1] as usize;
        let offset_start = offsets.value_offsets()[slot] as usize;
        let offset_end = offsets.value_offsets()[slot + 1] as usize;
        Ok(CompressedPositionStorage::SharedStream(
            SharedPositionStream::from_buffers(
                self.codec,
                self.offsets.slice(offset_start, offset_end - offset_start),
                self.bytes.slice(byte_start..byte_end),
            ),
        ))
    }
}

impl DeepSizeOf for PositionGroup {
    fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
        // The query views and cached handles share these same allocations.
        self.batch
            .columns()
            .iter()
            .map(|array| array.get_array_memory_size())
            .sum()
    }
}

#[derive(Debug, Clone)]
pub(super) struct PositionGroupKey {
    pub start: u32,
    pub end: u32,
}

impl CacheKey for PositionGroupKey {
    type ValueType = PositionGroup;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("position-group-{}-{}", self.start, self.end).into()
    }

    fn type_name() -> &'static str {
        "PositionGroup"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.scalar.inverted.position-group-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u32(self.start);
        builder.write_u32(self.end);
    }

    fn codec() -> Option<CacheCodec> {
        Some(CacheCodec::from_impl::<PositionGroup>())
    }
}
