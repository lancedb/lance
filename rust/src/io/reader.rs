//! Lance File Reader

use std::io::{Error, ErrorKind, Result};
use std::ops::Range;

use byteorder::ByteOrder;
use byteorder::LittleEndian;
use object_store::path::Path;
use prost::Message;

use super::object_store::ObjectStore;
use crate::format::pb;
use crate::format::Manifest;

/// Read Manifest on URI.
pub async fn read_manifest(object_store: &ObjectStore, path: &Path) -> Result<Manifest> {
    let file_size = object_store.inner.head(path).await?.size;
    const PREFETCH_SIZE: usize = 64 * 1024;
    let range = Range {
        start: std::cmp::max(file_size as i64 - PREFETCH_SIZE as i64, 0) as usize,
        end: file_size,
    };
    let buf = object_store.inner.get_range(path, range).await?;
    if buf.len() < 16 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Invalid format: file size is smaller than 8 bytes",
        ));
    }
    if !buf.ends_with(super::MAGIC) {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Invalid format: magic number does not match",
        ));
    }
    let manifest_pos = LittleEndian::read_i64(&buf[buf.len() - 16..buf.len() - 8]) as usize;
    assert!(file_size - manifest_pos < buf.len());
    let proto =
        pb::Manifest::decode(&buf[buf.len() - (file_size - manifest_pos) + 4..buf.len() - 16])?;
    Ok(Manifest::from(&proto))
}
