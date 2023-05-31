use std::collections::HashSet;

use arrow_array::UInt32Array;
use roaring::bitmap::RoaringBitmap;

use crate::format::{Fragment, DeletionFile, DeletionFileType};


/// Threshold for when a DeletionVector::Set should be promoted to a DeletionVector::Bitmap.
const BITMAP_THRESDHOLD: u32 = 5_000;
// TODO: Benchmark to find a better value.


/// Represents a set of deleted row ids in a single fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
enum DeletionVector {
    NoDeletions,
    Set(HashSet<u32>),
    Bitmap(RoaringBitmap),
}

// TODO: impl methods for DeletionVector
/// impl DeletionVector {
///     pub fn get(i: u32) -> bool { ... }
/// }
/// impl BitAnd for DeletionVector { ... }

impl IntoIterator for DeletionVector {
    type Item = Box<dyn Iterator<Item = u32>>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            DeletionVector::NoDeletions => Box::new(std::iter::empty()),
            DeletionVector::Set(set) => Box::new(set.into_iter()),
            DeletionVector::Bitmap(bitmap) => Box::new(bitmap.into_iter()),
        }
    }
}

impl FromIterator for DeletionVector {
    fn from_iter<T: IntoIterator<Item = u32>>(iter: T) -> Self {
        
        let iter = iter.into_iter();
        match iter.size_hint() {
            (_, Some(0)) => DeletionVector::NoDeletions,
            (lower, Some(upper)) if lower > BITMAP_THRESDHOLD => {
                let bitmap = iter.collect::<RoaringBitmap>();
                DeletionVector::Bitmap(bitmap)
            }
            (lower, Some(upper)) if upper < BITMAP_THRESDHOLD => {
                let set = iter.collect::<HashSet<_>>();
                DeletionVector::Set(set)
            },
            _ => {
                // We don't know the size, so just try as a set and move to bitmap
                // if it ends up being big.
                let set = iter.collect::<HashSet<_>>();
                if set.len() > BITMAP_THRESDHOLD {
                    let bitmap = set.into_iter().collect::<RoaringBitmap>();
                    DeletionVector::Bitmap(bitmap)
                } else {
                    DeletionVector::Set(set)
                }
            },
        }
    }
}

pub async fn write_deletion_file(fragment_id: i64, read_version: i64, removed_rows: DeletionVector) -> Result<Option<DeletionFile>> {
    match removed_rows {
        DeletionVector::NoDeletions => { Ok(None) },
        DeletionVector::Set(set) => {
            let id = rand::thread_rng().gen::<i64>();
            let path = format!("_deletions/{fragment_id}-{read_version}-{id}.arrow");

            let array = UInt32Array::from_iter(removed_rows);

            todo!("Write array as file");

            Ok(Some(DeletionFile {
                read_version,
                id,
                file_type: DeletionFileType::Array,
            }))
        },
        DeletionVector::Bitmap(bitmap) => {
            let id = rand::thread_rng().gen::<i64>();
            let path = format!("_deletions/{fragment_id}-{read_version}-{id}.bin");

            todo!("write bitmap to file");

            Ok(Some(DeletionFile {
                read_version,
                id,
                file_type: DeletionFileType::Bitmap,
            }))
        },
    }
}

pub async fn read_deletion_file(fragment: Fragment) -> Result<Option<DeletionVector>> {
    let deletion_file = fragment.deletion_file?;

    match deletion_file.file_type {
        DeletionFileType::Array => {
            let path = format!("_deletions/{fragment.id}-{deletion_file.read_version}-{deletion_file.id}.arrow");
            let array: UInt32Array = todo!("Read array from file");

            Ok(Some(DeletionVector::from_iter(array)))
        },
        DeletionFileType::Bitmap => {
            let path = format!("_deletions/{fragment.id}-{deletion_file.read_version}-{deletion_file.id}.bin");
            let bitmap: RoaringBitmap = todo!("Read bitmap from file");

            Ok(Some(DeletionVector::Bitmap(bitmap)))
        },
    }
}