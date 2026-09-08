// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
#[cfg(unix)]
use std::fs;
use std::sync::Arc;

use arrow_array::{Float64Array, RecordBatch, RecordBatchOptions, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance_function::{
    Artifact, BlobDescriptor, Digest, Error, ErrorCode, ExtensionTypes, ImageRoot, Platform,
    Result, Schemas, read_schema,
};
use rstest::rstest;
use serde_json::{Value, json};

const MANIFEST: &[u8] = include_bytes!("fixtures/py312-bookworm/manifest.json");
const CONFIG: &[u8] = include_bytes!("fixtures/py312-bookworm/config.json");
const INDEX: &[u8] = include_bytes!("fixtures/py312-bookworm/index.json");
const VERSION: &str = "sha256:7e22f815b6648e14f093a3979a8e5a2082fa773ebe1ec84b135cae7e84d6f8e6";

fn platform() -> Platform {
    Platform {
        os: "linux".into(),
        architecture: "amd64".into(),
        variant: None,
    }
}

fn fetch_fixture(blob: &BlobDescriptor) -> Result<Vec<u8>> {
    if blob.digest() == &Digest::of(MANIFEST) {
        Ok(MANIFEST.to_vec())
    } else if blob.digest() == &Digest::of(CONFIG) {
        Ok(CONFIG.to_vec())
    } else {
        Err(Error::new(
            ErrorCode::Incompatible,
            "test attempted to fetch an unexpected blob",
        ))
    }
}

#[rstest]
#[case::manifest(MANIFEST)]
#[case::layout_index(INDEX)]
fn reads_independently_built_image(#[case] bytes: &[u8]) {
    let artifact = Artifact::resolve(bytes, &platform(), fetch_fixture).unwrap();
    assert_eq!(artifact.version().as_str(), VERSION);
    assert_eq!(artifact.platform(), &platform());
    assert_eq!(artifact.descriptor().python().version, "3.12.14");
    assert_eq!(artifact.descriptor().entrypoint(), "app.function:create");
    assert_eq!(artifact.layers().len(), artifact.diff_ids().len());
    assert_eq!(artifact.layers().len(), 9);
    assert_eq!(
        artifact.image_config()["config"]["Entrypoint"][0],
        "/this/image/entrypoint/must/not/run"
    );
}

#[test]
fn version_uses_exact_manifest_bytes() {
    let mut spaced = MANIFEST.to_vec();
    spaced.push(b'\n');
    let artifact = Artifact::resolve(&spaced, &platform(), fetch_fixture).unwrap();
    assert_ne!(artifact.version().as_str(), VERSION);
    assert_eq!(artifact.version(), &Digest::of(&spaced));
    assert_eq!(
        artifact.descriptor(),
        Artifact::resolve(MANIFEST, &platform(), fetch_fixture)
            .unwrap()
            .descriptor()
    );
}

fn with_config(config: Value) -> (Vec<u8>, Vec<u8>) {
    let config = serde_json::to_vec(&config).unwrap();
    let mut manifest: Value = serde_json::from_slice(MANIFEST).unwrap();
    manifest["config"]["digest"] = json!(Digest::of(&config).as_str());
    manifest["config"]["size"] = json!(config.len());
    (serde_json::to_vec(&manifest).unwrap(), config)
}

#[test]
fn checks_selected_platform_against_config() {
    let mut config: Value = serde_json::from_slice(CONFIG).unwrap();
    config["architecture"] = json!("arm64");
    let (manifest, config) = with_config(config);
    let index = json!({"schemaVersion":2, "manifests":[{
        "mediaType":"application/vnd.oci.image.manifest.v1+json",
        "digest":Digest::of(&manifest).as_str(), "size":manifest.len(),
        "platform":{"os":"linux", "architecture":"amd64"}
    }]});
    let error = Artifact::resolve(index.to_string().as_bytes(), &platform(), |blob| {
        if blob.digest() == &Digest::of(&manifest) {
            Ok(manifest.clone())
        } else {
            Ok(config.clone())
        }
    })
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::Incompatible);
    assert!(error.message.contains("contradicts config"));
}

#[test]
fn nested_indices_skip_unrelated_platforms_and_keep_manifest_identity() {
    let mut nested: Value = serde_json::from_slice(INDEX).unwrap();
    nested["manifests"].as_array_mut().unwrap().insert(
        0,
        json!({
            "mediaType":"application/vnd.oci.image.manifest.v1+json",
            "digest":Digest::of(b"unavailable other platform").as_str(), "size":20,
            "platform":{"os":"linux", "architecture":"arm64"}
        }),
    );
    let nested = serde_json::to_vec(&nested).unwrap();
    let outer = json!({"schemaVersion":2, "mediaType":"application/vnd.oci.image.index.v1+json", "manifests":[{
        "mediaType":"application/vnd.oci.image.index.v1+json",
        "digest":Digest::of(&nested).as_str(), "size":nested.len()
    }]});
    let mut fetched = 0;
    let artifact = Artifact::resolve(outer.to_string().as_bytes(), &platform(), |blob| {
        fetched += 1;
        if blob.digest() == &Digest::of(&nested) {
            Ok(nested.clone())
        } else {
            fetch_fixture(blob)
        }
    })
    .unwrap();
    assert_eq!(fetched, 3);
    assert_eq!(artifact.version().as_str(), VERSION);
}

#[rstest]
#[case::size(false, "size")]
#[case::digest(true, "digest")]
fn rejects_config_integrity_failure(#[case] keep_size: bool, #[case] reason: &str) {
    let mut corrupt = CONFIG.to_vec();
    if keep_size {
        corrupt[0] = b' ';
    } else {
        corrupt.push(b' ');
    }
    let error = Artifact::resolve(MANIFEST, &platform(), |_| Ok(corrupt.clone())).unwrap_err();
    assert_eq!(error.code, ErrorCode::Incompatible);
    assert!(error.message.contains(reason));
}

#[rstest]
#[case::schema_version("/schemaVersion", json!(1), "schemaVersion")]
#[case::no_layers("/layers", json!([]), "layer")]
#[case::layer_media("/layers/0/mediaType", json!("application/unknown"), "layer media")]
#[case::config_media("/config/mediaType", json!("application/unknown"), "image config")]
#[case::negative_size("/config/size", json!(-1), "OCI manifest")]
#[case::digest_case("/config/digest", json!(format!("sha256:{}", "A".repeat(64))), "lowercase")]
fn rejects_manifest_contract_violations(
    #[case] pointer: &str,
    #[case] value: Value,
    #[case] reason: &str,
) {
    let mut manifest: Value = serde_json::from_slice(MANIFEST).unwrap();
    *manifest.pointer_mut(pointer).unwrap() = value;
    let error =
        Artifact::resolve(manifest.to_string().as_bytes(), &platform(), fetch_fixture).unwrap_err();
    assert_eq!(error.code, ErrorCode::Incompatible);
    assert!(error.message.contains(reason), "{error}");
}

#[rstest]
#[case::diffids("/rootfs/diff_ids", json!([]), "DiffID")]
#[case::rootfs_type("/rootfs/type", json!("unknown"), "DiffID")]
#[case::label_type("/config/Labels/lance.function.v1", json!({}), "string label")]
#[case::label_fields("/config/Labels/lance.function.v1", json!("{}"), "missing field")]
fn rejects_config_contract_violations(
    #[case] pointer: &str,
    #[case] value: Value,
    #[case] reason: &str,
) {
    let mut config: Value = serde_json::from_slice(CONFIG).unwrap();
    *config.pointer_mut(pointer).unwrap() = value;
    let (manifest, config) = with_config(config);
    let error = Artifact::resolve(&manifest, &platform(), |_| Ok(config.clone())).unwrap_err();
    assert_eq!(error.code, ErrorCode::Incompatible);
    assert!(error.message.contains(reason), "{error}");
}

#[test]
fn preserves_additional_oci_properties_but_rejects_duplicate_keys() {
    let mut config: Value = serde_json::from_slice(CONFIG).unwrap();
    config["future_property"] = json!({"example":true});
    let (manifest, config) = with_config(config);
    let artifact = Artifact::resolve(&manifest, &platform(), |_| Ok(config.clone())).unwrap();
    assert_eq!(artifact.image_config()["future_property"]["example"], true);

    let duplicate = String::from_utf8(MANIFEST.to_vec()).unwrap().replacen(
        "\"schemaVersion\":2",
        "\"schemaVersion\":2,\"schemaVersion\":2",
        1,
    );
    let error = Artifact::resolve(duplicate.as_bytes(), &platform(), fetch_fixture).unwrap_err();
    assert_eq!(error.code, ErrorCode::Incompatible);
    assert!(error.message.contains("duplicate JSON key"));
}

#[test]
fn rejects_no_matching_platform_without_fetching() {
    let index = json!({"schemaVersion":2, "manifests":[{
        "mediaType":"application/vnd.oci.image.manifest.v1+json",
        "digest":Digest::of(MANIFEST).as_str(), "size":MANIFEST.len(),
        "platform":{"os":"linux", "architecture":"arm64"}
    }]});
    let error = Artifact::resolve(index.to_string().as_bytes(), &platform(), |_| {
        Err(Error::new(ErrorCode::ExecutionFailed, "must not fetch"))
    })
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::Incompatible);
    assert!(error.message.contains("no Function image"));
}

#[test]
fn reads_pyarrow_schemas_from_ipc() {
    let schemas = Schemas::from_ipc(
        include_bytes!("fixtures/py312-bookworm/input.arrow"),
        include_bytes!("fixtures/py312-bookworm/output.arrow"),
        include_bytes!("fixtures/py312-bookworm/initialization.arrow"),
        &ExtensionTypes::default(),
    )
    .unwrap();
    assert_pyarrow_schemas(&schemas);
}

#[cfg(not(unix))]
#[test]
fn rejects_image_roots_on_non_posix_hosts() {
    let root = tempfile::tempdir().unwrap();
    let error = ImageRoot::new(root.path()).unwrap_err();
    assert_eq!(error.code, ErrorCode::Incompatible);
    assert!(error.message.contains("POSIX host filesystem"));
}

#[cfg(unix)]
#[rstest]
#[case::exact_limit(None, 0)]
#[case::input_one_byte_over(Some("input.arrow"), 1)]
#[case::output_one_byte_over(Some("output.arrow"), 1)]
#[case::initialization_one_byte_over(Some("initialization.arrow"), 1)]
#[case::oversized_sparse_file(Some("input.arrow"), 64 * 1024 * 1024)]
fn reads_pyarrow_schemas_through_the_artifact_entrypoint(
    #[case] oversized: Option<&str>,
    #[case] extra_bytes: u64,
) {
    let artifact = Artifact::resolve(INDEX, &platform(), fetch_fixture).unwrap();
    let root = tempfile::tempdir().unwrap();
    let schemas_path = root.path().join("opt/function/schemas");
    fs::create_dir_all(&schemas_path).unwrap();
    let fixtures = [
        (
            "input.arrow",
            include_bytes!("fixtures/py312-bookworm/input.arrow").as_slice(),
        ),
        (
            "output.arrow",
            include_bytes!("fixtures/py312-bookworm/output.arrow").as_slice(),
        ),
        (
            "initialization.arrow",
            include_bytes!("fixtures/py312-bookworm/initialization.arrow").as_slice(),
        ),
    ];
    let max_schema_bytes = fixtures.iter().map(|(_, bytes)| bytes.len()).max().unwrap() as u64;
    for (name, bytes) in fixtures {
        fs::write(schemas_path.join(name), bytes).unwrap();
    }
    if let Some(name) = oversized {
        fs::File::options()
            .write(true)
            .open(schemas_path.join(name))
            .unwrap()
            .set_len(max_schema_bytes + extra_bytes)
            .unwrap();
    }
    let image = ImageRoot::new(root.path()).unwrap();
    let result = Schemas::from_image(
        artifact.descriptor(),
        &image,
        &ExtensionTypes::default(),
        max_schema_bytes,
    );
    if let Some(name) = oversized {
        let error = result.unwrap_err();
        assert_eq!(error.code, ErrorCode::Incompatible);
        assert_eq!(
            error.message,
            format!(
                "schema /opt/function/schemas/{name} has {} bytes; limit is {max_schema_bytes} bytes",
                max_schema_bytes + extra_bytes,
            )
        );
    } else {
        assert_pyarrow_schemas(&result.unwrap());
    }
}

fn assert_pyarrow_schemas(schemas: &Schemas) {
    let expected_input = Schema::new_with_metadata(
        vec![Field::new("x", DataType::Float64, true)],
        HashMap::from([("fixture".into(), "scalar-v1".into())]),
    );
    assert_eq!(schemas.input().as_ref(), &expected_input);
    assert_eq!(
        schemas.output().as_ref(),
        &Schema::new(vec![Field::new("value", DataType::Float64, true)])
    );
    let initialization = RecordBatch::try_new(
        schemas.initialization().clone(),
        vec![
            Arc::new(Float64Array::from(vec![2.0])),
            Arc::new(StringArray::from(vec!["normal"])),
        ],
    )
    .unwrap();
    schemas.validate_initialization(&initialization).unwrap();
    let mut changed = expected_input.metadata().clone();
    changed.insert("fixture".into(), "different".into());
    assert_ne!(
        schemas.input().as_ref(),
        &expected_input.with_metadata(changed)
    );
}

#[test]
fn reads_independent_empty_and_nested_schemas() {
    let extensions = ExtensionTypes::default();
    let nested = read_schema(
        include_bytes!("fixtures/py312-bookworm/nested.arrow"),
        &extensions,
    )
    .unwrap();
    let expected = Schema::new_with_metadata(
        vec![Field::new(
            "nested",
            DataType::Struct(
                vec![
                    Field::new(
                        "vector",
                        DataType::FixedSizeList(
                            Arc::new(Field::new("item", DataType::Float32, true)),
                            3,
                        ),
                        false,
                    ),
                    Field::new(
                        "labels",
                        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                        true,
                    ),
                ]
                .into(),
            ),
            true,
        )],
        HashMap::from([("fixture".into(), "nested-vector".into())]),
    );
    assert_eq!(nested.as_ref(), &expected);
    let schemas = Schemas::from_ipc(
        include_bytes!("fixtures/py312-bookworm/input.arrow"),
        include_bytes!("fixtures/py312-bookworm/output.arrow"),
        include_bytes!("fixtures/py312-bookworm/empty_initialization.arrow"),
        &extensions,
    )
    .unwrap();
    assert!(schemas.initialization().fields().is_empty());
    let initialization = RecordBatch::try_new_with_options(
        schemas.initialization().clone(),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(1)),
    )
    .unwrap();
    schemas.validate_initialization(&initialization).unwrap();
}
