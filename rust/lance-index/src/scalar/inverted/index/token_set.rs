// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::hash::{Hash, Hasher};

use arrow::buffer::{Buffer, OffsetBuffer, ScalarBuffer};
use arrow_array::{StringArray, UInt32Array};
use deepsize::DeepSizeOf;
use fst::Streamer;

/// A token set that can be appended to.
#[derive(Debug, Clone)]
pub struct AppendableTokenSet {
    // We use a single buffer to store all tokens.
    data: String,
    // Offsets of each token in the data buffer, similar to Arrow's LargeBinaryArray.
    offsets: Vec<i32>,
    /// Token ids
    token_ids: Vec<u32>,
    /// A hash table mapping from token hash to token id.
    hash_table: hashbrown::hash_table::HashTable<u32>,
}

impl DeepSizeOf for AppendableTokenSet {
    fn deep_size_of_children(&self, ctx: &mut deepsize::Context) -> usize {
        let mut size = 0;
        size += self.data.deep_size_of_children(ctx);
        size += self.offsets.deep_size_of_children(ctx);
        size += self.token_ids.deep_size_of_children(ctx);
        size +=
            self.hash_table.capacity() * (std::mem::size_of::<u32>() + std::mem::size_of::<u64>());
        size
    }
}

impl AppendableTokenSet {
    pub fn new() -> Self {
        Self {
            data: String::new(),
            offsets: vec![0],
            hash_table: hashbrown::hash_table::HashTable::new(),
            token_ids: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let average_token_length = 4; // This is conservative.
        let mut offsets = Vec::with_capacity(capacity + 1);
        offsets.push(0);
        Self {
            data: String::with_capacity(capacity * average_token_length),
            offsets,
            hash_table: hashbrown::hash_table::HashTable::with_capacity(capacity),
            token_ids: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of tokens in the set.
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    pub fn get(&self, token: &str) -> Option<u32> {
        let hash = Self::hash(token);
        let offset = self
            .hash_table
            .find(hash, |i| {
                let start = self.offsets[*i as usize] as usize;
                let end = self.offsets[*i as usize + 1] as usize;
                &self.data[start..end] == token
            })
            .copied();
        offset.map(|i| self.token_ids[i as usize])
    }

    fn hash(token: &str) -> u64 {
        let mut hasher = std::hash::DefaultHasher::new();
        token.hash(&mut hasher);
        hasher.finish()
    }

    /// Insert a new token into the set without checking for duplicates.
    ///
    /// # Safety
    /// The caller must ensure that the token does not already exist in the set.
    pub unsafe fn insert_unchecked(&mut self, token: &str, token_id: u32) {
        let hash = Self::hash(token);

        debug_assert!(
            !self
                .hash_table
                .find(hash, |i| {
                    let start = self.offsets[*i as usize] as usize;
                    let end = self.offsets[*i as usize + 1] as usize;
                    &self.data[start..end] == token
                })
                .is_some(),
            "Token already exists in the set"
        );

        self.hash_table
            .insert_unique(hash, dbg!(self.offsets.len() as u32), |offset| {
                let start = self.offsets[*offset as usize] as usize;
                let end = self.offsets[*offset as usize + 1] as usize;
                Self::hash(&self.data[start..end])
            });
        self.data.push_str(token);
        self.offsets.push(self.data.len() as i32);
        self.token_ids.push(token_id);
    }

    pub fn iter(&self) -> AppendableTokenSetIterator<'_> {
        AppendableTokenSetIterator {
            token_set: self,
            index: 0,
        }
    }

    pub fn into_arrow(self) -> (StringArray, UInt32Array) {
        let token_ids =
            UInt32Array::from_iter_values((0..(self.offsets.len() - 1)).map(|i| i as u32));

        let offsets = OffsetBuffer::new(ScalarBuffer::from(self.offsets));
        let values = Buffer::from(self.data.into_bytes());

        // Safety: offsets were constructed correctly.
        let tokens = unsafe { StringArray::new_unchecked(offsets, values, None) };

        (tokens, token_ids)
    }
}

impl From<AppendableTokenSet> for fst::Map<Vec<u8>> {
    fn from(token_set: AppendableTokenSet) -> Self {
        // We need to insert in sorted order for fst::MapBuilder.
        let mut tokens = token_set.iter().collect::<Vec<_>>();
        tokens.sort_unstable_by_key(|(token, _)| *token);

        let mut builder = fst::MapBuilder::memory();
        for (token, token_id) in tokens {
            builder.insert(token, token_id as u64).unwrap();
        }
        let bytes = builder.into_inner().unwrap();
        fst::Map::new(bytes).unwrap()
    }
}

impl From<fst::Map<Vec<u8>>> for AppendableTokenSet {
    fn from(fst_map: fst::Map<Vec<u8>>) -> Self {
        let mut token_set = AppendableTokenSet::new();
        let mut stream = fst_map.stream();
        while let Some((token, token_id)) = stream.next() {
            let token_str = String::from_utf8_lossy(token);
            // Safety: fst::Map does not allow duplicate keys.
            unsafe { token_set.insert_unchecked(&token_str, token_id as u32) };
        }
        token_set
    }
}

pub struct AppendableTokenSetIterator<'a> {
    token_set: &'a AppendableTokenSet,
    index: usize,
}

impl<'a> Iterator for AppendableTokenSetIterator<'a> {
    type Item = (&'a str, u32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.token_set.offsets.len() - 1 {
            return None;
        }
        let start = self.token_set.offsets[self.index] as usize;
        let end = self.token_set.offsets[self.index + 1] as usize;
        let token = &self.token_set.data[start..end];
        let token_id = self.index as u32;
        self.index += 1;
        Some((token, token_id))
    }
}
