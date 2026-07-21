// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt::{Display, Formatter};

use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::{Error, Result};

pub use lance_encoding::version::{
    LEGACY_FORMAT_VERSION, LanceFileVersion, V2_FORMAT_2_0, V2_FORMAT_2_1, V2_FORMAT_2_2,
    V2_FORMAT_2_3,
};

/// The exact persisted identity of a Lance file format.
///
/// Unlike [`LanceFileVersion`], this type cannot represent release selectors such as
/// `stable` or `next`. Exact versions deliberately have no ordering because format
/// capabilities are not implied by release order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LanceFileFormat {
    /// The legacy v1 file format.
    V1,
    /// The v2.0 file format.
    V2_0,
    /// The v2.1 file format.
    V2_1,
    /// The v2.2 file format.
    V2_2,
    /// The v2.3 file format.
    V2_3,
}

impl DeepSizeOf for LanceFileFormat {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        0
    }
}

impl LanceFileFormat {
    /// Decode the exact version string stored in a dataset manifest.
    ///
    /// Public selector aliases such as `legacy`, `0.3`, `stable`, and `next` are
    /// intentionally rejected because manifests only store canonical exact versions.
    pub fn from_manifest_string(value: &str) -> Result<Self> {
        match value {
            LEGACY_FORMAT_VERSION => Ok(Self::V1),
            V2_FORMAT_2_0 => Ok(Self::V2_0),
            V2_FORMAT_2_1 => Ok(Self::V2_1),
            V2_FORMAT_2_2 => Ok(Self::V2_2),
            V2_FORMAT_2_3 => Ok(Self::V2_3),
            _ => Err(unknown_version(value)),
        }
    }

    /// Encode this exact version as the canonical string stored in a dataset manifest.
    pub const fn to_manifest_string(self) -> &'static str {
        match self {
            Self::V1 => LEGACY_FORMAT_VERSION,
            Self::V2_0 => V2_FORMAT_2_0,
            Self::V2_1 => V2_FORMAT_2_1,
            Self::V2_2 => V2_FORMAT_2_2,
            Self::V2_3 => V2_FORMAT_2_3,
        }
    }

    /// Decode the major/minor version stored in `DataFile` metadata.
    ///
    /// Legacy manifests may omit these fields and decode to `(0, 0)`, so all legacy
    /// v1 number pairs accepted by the historical decoder remain valid inputs. The
    /// historical generic decoder also accepted the standard v2.0 footer pair `(0, 3)`;
    /// decoding retains that compatibility while encoding always emits `(2, 0)`.
    pub fn from_data_file_numbers(major: u32, minor: u32) -> Result<Self> {
        match (major, minor) {
            (0, 0..=2) => Ok(Self::V1),
            (0, 3) | (2, 0) => Ok(Self::V2_0),
            (2, 1) => Ok(Self::V2_1),
            (2, 2) => Ok(Self::V2_2),
            (2, 3) => Ok(Self::V2_3),
            _ => Err(unknown_version(format_args!("{}.{}", major, minor))),
        }
    }

    /// Encode the canonical major/minor pair stored in `DataFile` metadata.
    pub const fn to_data_file_numbers(self) -> (u32, u32) {
        match self {
            Self::V1 => (0, 2),
            Self::V2_0 => (2, 0),
            Self::V2_1 => (2, 1),
            Self::V2_2 => (2, 2),
            Self::V2_3 => (2, 3),
        }
    }

    /// Decode the major/minor version stored in a Lance file footer.
    ///
    /// V2.0 has two accepted representations: `(0, 3)` from the standard file writer
    /// and `(2, 0)` from self-described and mini-lance writers.
    pub fn from_footer_numbers(major: u16, minor: u16) -> Result<Self> {
        match (major, minor) {
            (0, 0..=2) => Ok(Self::V1),
            (0, 3) | (2, 0) => Ok(Self::V2_0),
            (2, 1) => Ok(Self::V2_1),
            (2, 2) => Ok(Self::V2_2),
            (2, 3) => Ok(Self::V2_3),
            _ => Err(unknown_version(format_args!("{}.{}", major, minor))),
        }
    }

    /// Encode the footer numbers emitted by the standard Lance file writer.
    pub const fn to_standard_footer_numbers(self) -> (u16, u16) {
        match self {
            Self::V1 => (0, 2),
            Self::V2_0 => (0, 3),
            Self::V2_1 => (2, 1),
            Self::V2_2 => (2, 2),
            Self::V2_3 => (2, 3),
        }
    }

    /// Encode the footer numbers emitted by self-described and mini-lance writers.
    pub const fn to_embedded_footer_numbers(self) -> (u16, u16) {
        match self {
            Self::V1 => (0, 2),
            Self::V2_0 => (2, 0),
            Self::V2_1 => (2, 1),
            Self::V2_2 => (2, 2),
            Self::V2_3 => (2, 3),
        }
    }
}

impl Display for LanceFileFormat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.to_manifest_string())
    }
}

impl From<LanceFileFormat> for LanceFileVersion {
    fn from(value: LanceFileFormat) -> Self {
        match value {
            LanceFileFormat::V1 => Self::Legacy,
            LanceFileFormat::V2_0 => Self::V2_0,
            LanceFileFormat::V2_1 => Self::V2_1,
            LanceFileFormat::V2_2 => Self::V2_2,
            LanceFileFormat::V2_3 => Self::V2_3,
        }
    }
}

impl From<LanceFileVersion> for LanceFileFormat {
    fn from(value: LanceFileVersion) -> Self {
        match value.resolve() {
            LanceFileVersion::Legacy => Self::V1,
            LanceFileVersion::V2_0 => Self::V2_0,
            LanceFileVersion::V2_1 => Self::V2_1,
            LanceFileVersion::V2_2 => Self::V2_2,
            LanceFileVersion::V2_3 => Self::V2_3,
            LanceFileVersion::Stable | LanceFileVersion::Next => {
                unreachable!("resolved file-version selector must be exact")
            }
        }
    }
}

fn unknown_version(value: impl Display) -> Error {
    Error::invalid_input_source(format!("Unknown Lance storage version: {}", value).into())
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use lance_io::object_store::ObjectStore;
    use object_store::path::Path;

    use super::*;

    const EXACT_VERSIONS: [LanceFileFormat; 5] = [
        LanceFileFormat::V1,
        LanceFileFormat::V2_0,
        LanceFileFormat::V2_1,
        LanceFileFormat::V2_2,
        LanceFileFormat::V2_3,
    ];

    #[test]
    fn selector_resolution_is_exact() {
        let cases = [
            (LanceFileVersion::Legacy, LanceFileFormat::V1),
            (LanceFileVersion::V2_0, LanceFileFormat::V2_0),
            (LanceFileVersion::V2_1, LanceFileFormat::V2_1),
            (LanceFileVersion::Stable, LanceFileFormat::V2_1),
            (LanceFileVersion::V2_2, LanceFileFormat::V2_2),
            (LanceFileVersion::Next, LanceFileFormat::V2_3),
            (LanceFileVersion::V2_3, LanceFileFormat::V2_3),
        ];

        for (selector, expected) in cases {
            assert_eq!(LanceFileFormat::from(selector), expected);
            assert_eq!(LanceFileVersion::from(expected), selector.resolve());
        }
    }

    #[test]
    fn public_selector_aliases_remain_unchanged() {
        let cases = [
            ("0.1", LanceFileVersion::Legacy),
            ("legacy", LanceFileVersion::Legacy),
            ("2.0", LanceFileVersion::V2_0),
            ("0.3", LanceFileVersion::V2_0),
            ("2.1", LanceFileVersion::V2_1),
            ("stable", LanceFileVersion::Stable),
            ("2.2", LanceFileVersion::V2_2),
            ("next", LanceFileVersion::Next),
            ("2.3", LanceFileVersion::V2_3),
        ];

        for (value, expected) in cases {
            assert_eq!(LanceFileVersion::from_str(value).unwrap(), expected);
        }
    }

    #[test]
    fn manifest_codec_only_accepts_canonical_exact_versions() {
        for version in EXACT_VERSIONS {
            let encoded = version.to_manifest_string();
            assert_eq!(
                LanceFileFormat::from_manifest_string(encoded).unwrap(),
                version
            );
        }

        for selector_or_alias in ["legacy", "0.3", "stable", "next"] {
            assert!(LanceFileFormat::from_manifest_string(selector_or_alias).is_err());
        }
    }

    #[test]
    fn data_file_codec_preserves_wire_numbers() {
        let cases = [
            (LanceFileFormat::V1, (0, 2)),
            (LanceFileFormat::V2_0, (2, 0)),
            (LanceFileFormat::V2_1, (2, 1)),
            (LanceFileFormat::V2_2, (2, 2)),
            (LanceFileFormat::V2_3, (2, 3)),
        ];

        for (version, encoded) in cases {
            assert_eq!(version.to_data_file_numbers(), encoded);
            assert_eq!(
                LanceFileFormat::from_data_file_numbers(encoded.0, encoded.1).unwrap(),
                version
            );
        }
        for minor in 0..=2 {
            assert_eq!(
                LanceFileFormat::from_data_file_numbers(0, minor).unwrap(),
                LanceFileFormat::V1
            );
        }
        assert_eq!(
            LanceFileFormat::from_data_file_numbers(0, 3).unwrap(),
            LanceFileFormat::V2_0
        );
    }

    #[test]
    fn footer_codec_preserves_both_v2_0_writer_representations() {
        let standard_cases = [
            (LanceFileFormat::V1, (0, 2)),
            (LanceFileFormat::V2_0, (0, 3)),
            (LanceFileFormat::V2_1, (2, 1)),
            (LanceFileFormat::V2_2, (2, 2)),
            (LanceFileFormat::V2_3, (2, 3)),
        ];
        let embedded_cases = [
            (LanceFileFormat::V1, (0, 2)),
            (LanceFileFormat::V2_0, (2, 0)),
            (LanceFileFormat::V2_1, (2, 1)),
            (LanceFileFormat::V2_2, (2, 2)),
            (LanceFileFormat::V2_3, (2, 3)),
        ];

        for (version, encoded) in standard_cases {
            assert_eq!(version.to_standard_footer_numbers(), encoded);
            assert_eq!(
                LanceFileFormat::from_footer_numbers(encoded.0, encoded.1).unwrap(),
                version
            );
        }
        for (version, encoded) in embedded_cases {
            assert_eq!(version.to_embedded_footer_numbers(), encoded);
            assert_eq!(
                LanceFileFormat::from_footer_numbers(encoded.0, encoded.1).unwrap(),
                version
            );
        }
        for minor in 0..=2 {
            assert_eq!(
                LanceFileFormat::from_footer_numbers(0, minor).unwrap(),
                LanceFileFormat::V1
            );
        }
    }

    #[tokio::test]
    async fn file_version_detection_accepts_all_legacy_footer_aliases() {
        let object_store = ObjectStore::memory();
        for minor in 0u16..=2 {
            let path = Path::from(format!("legacy-{minor}.lance"));
            let mut footer = Vec::with_capacity(8);
            footer.extend_from_slice(&0u16.to_le_bytes());
            footer.extend_from_slice(&minor.to_le_bytes());
            footer.extend_from_slice(crate::format::MAGIC);
            object_store.put(&path, &footer).await.unwrap();

            assert_eq!(
                crate::determine_file_version(&object_store, &path, Some(footer.len()))
                    .await
                    .unwrap(),
                LanceFileVersion::Legacy
            );
        }
    }
}
