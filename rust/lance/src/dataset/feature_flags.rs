// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Feature flags
use crate::format::Manifest;

/// Flag to indicate that the dataset contains deletion files.
pub const FLAG_DELETION_FILES: u64 = 1;
/// Flag to indicate that the dataset contains FixedSizeList fields that have
/// child fields saved as part of the schema.
pub const FLAG_FSL_FIELDS: u64 = 2;

/// Set the reader and writer feature flags in the manifest based on the contents of the manifest.
pub fn apply_feature_flags(manifest: &mut Manifest) {
    // Reset flags
    manifest.reader_feature_flags = 0;
    manifest.writer_feature_flags = 0;

    let has_deletion_files = manifest
        .fragments
        .iter()
        .any(|frag| frag.deletion_file.is_some());
    if has_deletion_files {
        // Both readers and writers need to be able to read deletion files
        manifest.reader_feature_flags |= FLAG_DELETION_FILES;
        manifest.writer_feature_flags |= FLAG_DELETION_FILES;
    }

    if manifest.schema.contains_fsl_fields() {
        // Both readers and writers need to be able to read FSL fields
        manifest.reader_feature_flags |= FLAG_FSL_FIELDS;
        manifest.writer_feature_flags |= FLAG_FSL_FIELDS;
    }
}

pub fn can_read_dataset(reader_flags: u64) -> bool {
    reader_flags <= FLAG_FSL_FIELDS
}

pub fn can_write_dataset(writer_flags: u64) -> bool {
    writer_flags <= FLAG_FSL_FIELDS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_check() {
        assert!(can_read_dataset(0));
        assert!(can_read_dataset(super::FLAG_DELETION_FILES));
        assert!(!can_read_dataset(super::FLAG_DELETION_FILES + 1));
    }

    #[test]
    fn test_write_check() {
        assert!(can_write_dataset(0));
        assert!(can_write_dataset(super::FLAG_DELETION_FILES));
        assert!(!can_write_dataset(super::FLAG_DELETION_FILES + 1));
    }
}
