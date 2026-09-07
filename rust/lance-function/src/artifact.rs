// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::str::FromStr;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as _, Sha256};

use crate::{Descriptor, Error, Result, json};

const MANIFEST_MEDIA_TYPE: &str = "application/vnd.oci.image.manifest.v1+json";
const INDEX_MEDIA_TYPE: &str = "application/vnd.oci.image.index.v1+json";
const CONFIG_MEDIA_TYPE: &str = "application/vnd.oci.image.config.v1+json";
const FUNCTION_LABEL: &str = "lance.function.v1";

/// A canonical SHA-256 identity, independent of names, registries, and cache paths.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Digest(String);

impl Digest {
    /// Hash the exact bytes; never parse or reserialize before computing identity.
    pub fn of(bytes: &[u8]) -> Self {
        Self(format!("sha256:{}", hex::encode(Sha256::digest(bytes))))
    }

    /// `sha256:` followed by 64 lowercase hexadecimal digits.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for Digest {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        let hash = value.strip_prefix("sha256:").ok_or_else(|| {
            Error::incompatible(format!("unsupported digest {value:?}; expected sha256"))
        })?;
        if hash.len() != 64
            || !hash
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(Error::incompatible(format!(
                "digest {value:?} requires 64 lowercase hex digits"
            )));
        }
        Ok(Self(value.to_owned()))
    }
}

impl TryFrom<String> for Digest {
    type Error = Error;
    fn try_from(value: String) -> Result<Self> {
        value.parse()
    }
}

impl From<Digest> for String {
    fn from(value: Digest) -> Self {
        value.0
    }
}

/// The size and identity of a referenced OCI object.
///
/// Unknown OCI descriptor properties are accepted; the Function descriptor has
/// a separate strict unknown-field policy. Fetchers must verify returned bytes.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BlobDescriptor {
    media_type: String,
    digest: Digest,
    size: u64,
}

impl BlobDescriptor {
    /// Media type declared by the referring OCI object.
    pub fn media_type(&self) -> &str {
        &self.media_type
    }

    /// Content identity used by the registry/layout fetcher.
    pub fn digest(&self) -> &Digest {
        &self.digest
    }

    /// Expected byte length before decompression.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Check both size and exact bytes before parsing or decompressing an object.
    pub fn verify(&self, bytes: &[u8]) -> Result<()> {
        if bytes.len() as u64 != self.size {
            return Err(Error::incompatible(format!(
                "blob {} size {} differs from declared {}",
                self.digest.as_str(),
                bytes.len(),
                self.size
            )));
        }
        let actual = Digest::of(bytes);
        if actual != self.digest {
            return Err(Error::incompatible(format!(
                "blob digest {} differs from declared {}",
                actual.as_str(),
                self.digest.as_str()
            )));
        }
        Ok(())
    }
}

/// OCI image platform. Architecture names follow OCI/Go conventions, e.g. `amd64`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct Platform {
    /// Function v1 requires `linux`.
    pub os: String,
    /// Required processor architecture, such as `amd64` or `arm64`.
    pub architecture: String,
    /// Optional architecture variant; `None` means no variant was specified.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
}

impl Platform {
    fn validate(&self) -> Result<()> {
        if self.os != "linux"
            || self.architecture.is_empty()
            || self.variant.as_ref().is_some_and(String::is_empty)
        {
            return Err(Error::incompatible(format!(
                "unsupported Function platform {self:?}"
            )));
        }
        Ok(())
    }

    fn matches(&self, actual: &Self) -> bool {
        self.os == actual.os
            && self.architecture == actual.architecture
            && self
                .variant
                .as_ref()
                .is_none_or(|variant| Some(variant) == actual.variant.as_ref())
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Manifest {
    schema_version: u32,
    config: BlobDescriptor,
    layers: Vec<BlobDescriptor>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Index {
    schema_version: u32,
    manifests: Vec<IndexEntry>,
}

#[derive(Deserialize)]
struct IndexEntry {
    #[serde(flatten)]
    blob: BlobDescriptor,
    platform: Option<Platform>,
}

#[derive(Deserialize)]
struct Rootfs {
    #[serde(rename = "type")]
    kind: String,
    diff_ids: Vec<Digest>,
}

/// Validated OCI metadata for one concrete platform Function image.
///
/// Layers are described but not fetched or unpacked. Before execution the caller
/// must verify every layer and DiffID and supply a complete immutable rootfs.
/// Image runtime properties remain available without being rewritten by this crate.
#[derive(Debug, Clone)]
pub struct Artifact {
    version: Digest,
    platform: Platform,
    descriptor: Descriptor,
    config: Value,
    layers: Vec<BlobDescriptor>,
    diff_ids: Vec<Digest>,
}

impl Artifact {
    /// Resolve raw manifest or index bytes to one concrete image.
    ///
    /// `fetch` supplies referenced manifests and configs from a registry, layout,
    /// or test store. Every fetched object's size and digest are verified here.
    /// If a root reference supplied a digest, verify it against `document` first.
    /// Index order breaks ties, as recommended by OCI. Nested indices are supported
    /// up to 32 levels; unrelated platforms are skipped without fetching them.
    /// A matching entry whose platform contradicts its config is rejected.
    ///
    /// The selected manifest's exact bytes determine [`Self::version`], including
    /// when the caller started from a mutable tag or a multi-platform index.
    pub fn resolve(
        document: &[u8],
        target: &Platform,
        mut fetch: impl FnMut(&BlobDescriptor) -> Result<Vec<u8>>,
    ) -> Result<Self> {
        target.validate()?;
        Self::resolve_document(document, target, &mut fetch, &[], 0)?.ok_or_else(|| {
            Error::incompatible(format!("OCI index has no Function image for {target:?}"))
        })
    }

    fn resolve_document(
        bytes: &[u8],
        target: &Platform,
        fetch: &mut impl FnMut(&BlobDescriptor) -> Result<Vec<u8>>,
        declarations: &[Platform],
        depth: usize,
    ) -> Result<Option<Self>> {
        const MAX_INDEX_DEPTH: usize = 32;
        if depth > MAX_INDEX_DEPTH {
            return Err(Error::incompatible("OCI index nesting exceeds 32 levels"));
        }
        let value = json::parse(bytes, "OCI document")?;
        let media_type = document_media_type(&value)?;
        if media_type == INDEX_MEDIA_TYPE {
            let index: Index = json::decode(value, "OCI index")?;
            if index.schema_version != 2 {
                return Err(Error::incompatible(format!(
                    "OCI index schemaVersion {} is not 2",
                    index.schema_version
                )));
            }
            for entry in index.manifests {
                if entry.platform.as_ref().is_some_and(|platform| {
                    platform.os != target.os
                        || platform.architecture != target.architecture
                        || platform
                            .variant
                            .as_ref()
                            .zip(target.variant.as_ref())
                            .is_some_and(|(left, right)| left != right)
                }) {
                    continue;
                }
                if !matches!(
                    entry.blob.media_type.as_str(),
                    MANIFEST_MEDIA_TYPE | INDEX_MEDIA_TYPE
                ) {
                    continue;
                }
                let child_bytes = fetch(&entry.blob)?;
                entry.blob.verify(&child_bytes)?;
                let child = json::parse(&child_bytes, "OCI index child")?;
                if document_media_type(&child)? != entry.blob.media_type {
                    return Err(Error::incompatible(
                        "OCI child media type differs from its index descriptor",
                    ));
                }
                let mut claimed = declarations.to_vec();
                if let Some(platform) = entry.platform {
                    platform.validate()?;
                    claimed.push(platform);
                }
                if let Some(artifact) =
                    Self::resolve_document(&child_bytes, target, fetch, &claimed, depth + 1)?
                {
                    return Ok(Some(artifact));
                }
            }
            return Ok(None);
        }

        let manifest: Manifest = json::decode(value, "OCI manifest")?;
        if manifest.schema_version != 2
            || manifest.config.media_type != CONFIG_MEDIA_TYPE
            || manifest.layers.is_empty()
        {
            return Err(Error::incompatible(
                "Function requires OCI schemaVersion 2, image config, and at least one layer",
            ));
        }
        for layer in &manifest.layers {
            if !matches!(
                layer.media_type.as_str(),
                "application/vnd.oci.image.layer.v1.tar"
                    | "application/vnd.oci.image.layer.v1.tar+gzip"
                    | "application/vnd.oci.image.layer.v1.tar+zstd"
            ) {
                return Err(Error::incompatible(format!(
                    "unsupported Function layer media type {:?}",
                    layer.media_type
                )));
            }
        }
        let config_bytes = fetch(&manifest.config)?;
        manifest.config.verify(&config_bytes)?;
        let config = json::parse(&config_bytes, "OCI image config")?;
        let platform: Platform = json::decode(config.clone(), "OCI config platform")?;
        platform.validate()?;
        for declaration in declarations {
            if !declaration.matches(&platform) {
                return Err(Error::incompatible(format!(
                    "OCI index platform {declaration:?} contradicts config {platform:?}"
                )));
            }
        }
        if !target.matches(&platform) {
            return Ok(None);
        }
        let rootfs: Rootfs = json::decode(
            config
                .get("rootfs")
                .cloned()
                .ok_or_else(|| Error::incompatible("OCI config missing rootfs"))?,
            "OCI rootfs",
        )?;
        if rootfs.kind != "layers" || rootfs.diff_ids.len() != manifest.layers.len() {
            return Err(Error::incompatible(
                "OCI rootfs must contain one ordered DiffID for each manifest layer",
            ));
        }
        let label = config
            .pointer("/config/Labels")
            .and_then(|labels| labels.get(FUNCTION_LABEL))
            .and_then(Value::as_str)
            .ok_or_else(|| {
                Error::incompatible("OCI config requires string label lance.function.v1")
            })?;
        let descriptor = Descriptor::from_json(label)?;
        Ok(Some(Self {
            version: Digest::of(bytes),
            platform,
            descriptor,
            config,
            layers: manifest.layers,
            diff_ids: rootfs.diff_ids,
        }))
    }

    /// Exact platform manifest digest; an index digest is never an execution identity.
    pub fn version(&self) -> &Digest {
        &self.version
    }

    /// Authoritative platform read from the verified image config.
    pub fn platform(&self) -> &Platform {
        &self.platform
    }

    /// Parsed Function requirements and interface, without catalog state.
    pub fn descriptor(&self) -> &Descriptor {
        &self.descriptor
    }

    /// Complete image config, including OCI fields that this parser does not interpret.
    ///
    /// The launcher must override Entrypoint/Cmd and construct authorized runtime
    /// state; retaining these fields does not authorize executing them.
    pub fn image_config(&self) -> &Value {
        &self.config
    }

    /// Ordered compressed layer identities for subsequent rootfs preparation.
    pub fn layers(&self) -> &[BlobDescriptor] {
        &self.layers
    }

    /// Ordered uncompressed layer identities, to verify during layer application.
    pub fn diff_ids(&self) -> &[Digest] {
        &self.diff_ids
    }
}

fn document_media_type(value: &Value) -> Result<&str> {
    let media_type = match value.get("mediaType") {
        Some(value) => value
            .as_str()
            .ok_or_else(|| Error::incompatible("OCI mediaType must be a string"))?,
        None if value.get("manifests").is_some() => INDEX_MEDIA_TYPE,
        None if value.get("config").is_some() => MANIFEST_MEDIA_TYPE,
        None => {
            return Err(Error::incompatible(
                "OCI document is neither a manifest nor an index",
            ));
        }
    };
    if !matches!(media_type, MANIFEST_MEDIA_TYPE | INDEX_MEDIA_TYPE) {
        return Err(Error::incompatible(format!(
            "unsupported OCI document media type {media_type:?}"
        )));
    }
    Ok(media_type)
}
