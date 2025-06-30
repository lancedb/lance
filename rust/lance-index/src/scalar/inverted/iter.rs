// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::hash_map;

use arrow::array::AsArray;
use arrow_array::{Array, LargeBinaryArray, ListArray};
use fst::Streamer;

use super::{
    builder::BLOCK_SIZE,
    encoding::{decompress_positions, decompress_posting_block, decompress_posting_remainder},
    PostingList,
};

pub enum TokenSource<'a> {
    HashMap(hash_map::Iter<'a, String, u32>),
    Fst(fst::map::Stream<'a>),
}
pub struct TokenIterator<'a> {
    source: TokenSource<'a>,
}

impl<'a> TokenIterator<'a> {
    pub(crate) fn new(source: TokenSource<'a>) -> Self {
        Self { source }
    }
}

impl Iterator for TokenIterator<'_> {
    type Item = (String, u32);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.source {
            TokenSource::HashMap(iter) => iter
                .next()
                .map(|(token, token_id)| (token.clone(), *token_id)),
            TokenSource::Fst(iter) => iter.next().map(|(token, token_id)| {
                (String::from_utf8_lossy(token).into_owned(), token_id as u32)
            }),
        }
    }
}

pub enum PostingListIterator<'a> {
    Plain(PlainPostingListIterator<'a>),
    Compressed(Box<CompressedPostingListIterator>),
}

impl<'a> PostingListIterator<'a> {
    pub fn new(posting: &'a PostingList) -> Self {
        match posting {
            PostingList::Plain(posting) => Self::Plain(posting.iter()),
            PostingList::Compressed(posting) => {
                Self::Compressed(Box::new(CompressedPostingListIterator::new(
                    posting.length as usize,
                    posting.blocks.clone(),
                    posting.positions.clone(),
                )))
            }
        }
    }
}

impl<'a> Iterator for PostingListIterator<'a> {
    type Item = (u64, u32, Option<Box<dyn Iterator<Item = u32> + 'a>>);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            PostingListIterator::Plain(iter) => iter
                .next()
                .map(|(doc_id, freq, pos)| (doc_id, freq as u32, pos)),
            PostingListIterator::Compressed(iter) => iter
                .as_mut()
                .next()
                .map(|(doc_id, freq, pos)| (doc_id as u64, freq, pos)),
        }
    }
}

pub type PlainPostingListIterator<'a> =
    Box<dyn Iterator<Item = (u64, f32, Option<Box<dyn Iterator<Item = u32> + 'a>>)> + 'a>;

pub struct CompressedPostingListIterator {
    remainder: usize,
    blocks: LargeBinaryArray,
    next_block_idx: usize,
    positions: Option<ListArray>,
    idx: u32,
    iter: InnerIterator,
    buffer: [u32; BLOCK_SIZE],
}

type InnerIterator = std::iter::Zip<std::vec::IntoIter<u32>, std::vec::IntoIter<u32>>;

impl CompressedPostingListIterator {
    pub fn new(length: usize, blocks: LargeBinaryArray, positions: Option<ListArray>) -> Self {
        debug_assert!(length > 0);
        debug_assert_eq!(blocks.len(), length.div_ceil(BLOCK_SIZE));

        Self {
            remainder: length % BLOCK_SIZE,
            blocks,
            next_block_idx: 0,
            positions,
            idx: 0,
            iter: std::iter::zip(Vec::new(), Vec::new()),
            buffer: [0; BLOCK_SIZE],
        }
    }
}

impl Iterator for CompressedPostingListIterator {
    type Item = (u32, u32, Option<Box<dyn Iterator<Item = u32>>>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((doc_id, freq)) = self.iter.next() {
            let positions = self.positions.as_ref().map(|p| {
                let compressed = p.value(self.idx as usize);
                let positions = decompress_positions(compressed.as_binary());
                Box::new(positions.into_iter()) as _
            });
            self.idx += 1;
            return Some((doc_id, freq, positions));
        }

        // move to the next block
        if self.next_block_idx >= self.blocks.len() {
            return None;
        }
        let compressed = self.blocks.value(self.next_block_idx);
        self.next_block_idx += 1;

        let mut doc_ids = Vec::with_capacity(BLOCK_SIZE);
        let mut frequencies = Vec::with_capacity(BLOCK_SIZE);
        if self.next_block_idx == self.blocks.len() && self.remainder > 0 {
            decompress_posting_remainder(
                compressed,
                self.remainder,
                &mut doc_ids,
                &mut frequencies,
            );
        } else {
            decompress_posting_block(compressed, &mut self.buffer, &mut doc_ids, &mut frequencies);
        }
        self.iter = std::iter::zip(doc_ids, frequencies);
        self.next()
    }
}

pub fn take_fst_keys<'f, I, S>(s: I, dst: &mut Vec<String>, max_expansion: usize)
where
    I: for<'a> fst::IntoStreamer<'a, Into = S, Item = (&'a [u8], u64)>,
    S: 'f + for<'a> fst::Streamer<'a, Item = (&'a [u8], u64)>,
{
    let mut stream = s.into_stream();
    while let Some((token, _)) = stream.next() {
        dst.push(String::from_utf8_lossy(token).into_owned());
        if dst.len() >= max_expansion {
            break;
        }
    }
}
