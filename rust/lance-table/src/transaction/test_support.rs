// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fixtures shared between the tests of several submodules.

use crate::format::overlay::{DataOverlayFile, OverlayCoverage};
use crate::format::{
    DataFile, DataStorageFormat, Fragment, IndexMetadata, Manifest, ManifestBuildConfig,
};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use chrono::Utc;
use lance_core::datatypes::Schema as LanceSchema;
use lance_file::version::LanceFileVersion;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// The build config that `lance`'s `ManifestWriteConfig::default()` resolves to.
pub fn default_build_config() -> ManifestBuildConfig {
    ManifestBuildConfig {
        auto_set_feature_flags: true,
        timestamp_nanos: std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        use_stable_row_ids: false,
        use_legacy_format: None,
        storage_format: None,
        disable_transaction_file: false,
    }
}

pub fn sample_manifest() -> Manifest {
    let schema = ArrowSchema::new(vec![ArrowField::new("id", DataType::Int32, false)]);
    Manifest::new(
        LanceSchema::try_from(&schema).unwrap(),
        Arc::new(vec![Fragment::new(0)]),
        DataStorageFormat::new(LanceFileVersion::V2_0),
        HashMap::new(),
    )
}

pub fn sample_index_metadata(name: &str) -> IndexMetadata {
    IndexMetadata {
        uuid: Uuid::new_v4(),
        fields: vec![0],
        name: name.to_string(),
        dataset_version: 0,
        fragment_bitmap: Some([0].into_iter().collect()),
        index_details: None,
        index_version: 1,
        created_at: Some(Utc::now()),
        base_id: None,
        files: None,
    }
}

pub fn overlay_with_field(field: i32, committed_version: u64) -> DataOverlayFile {
    DataOverlayFile {
        data_file: DataFile::new_legacy_from_fields("o.lance", vec![field], None),
        coverage: OverlayCoverage::dense(roaring::RoaringBitmap::from_iter([0u32])),
        committed_version,
    }
}
