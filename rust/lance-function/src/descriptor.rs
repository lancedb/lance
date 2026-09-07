// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::{ContainerPath, Error, ImageRoot, PathKind, Result, json};

macro_rules! version_one {
    ($name:ident, $description:literal) => {
        #[doc = $description]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
        #[serde(try_from = "u32", into = "u32")]
        pub enum $name {
            /// The v1 contract; unsupported wire versions are rejected.
            V1,
        }

        impl TryFrom<u32> for $name {
            type Error = Error;

            fn try_from(version: u32) -> Result<Self> {
                match version {
                    1 => Ok(Self::V1),
                    _ => Err(Error::incompatible(format!(
                        "unsupported {} version {version}",
                        stringify!($name)
                    ))),
                }
            }
        }

        impl From<$name> for u32 {
            fn from(version: $name) -> Self {
                match version {
                    $name::V1 => 1,
                }
            }
        }
    };
}

version_one!(FormatVersion, "Descriptor layout and image loading rules.");
version_one!(
    PythonApi,
    "Python factory, scalar instance, and context API."
);
version_one!(
    ScalarVersion,
    "Row-aligned scalar semantics, independent of the Python API version."
);
version_one!(
    CudaVersion,
    "CUDA capability declaration semantics, independent of the toolkit version."
);

/// Image-owned interpreter and ordered code search paths.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Python {
    /// Python implementation; Function v1 supports `cpython`.
    pub implementation: String,
    /// Full image interpreter version, independent of the Function Python API.
    pub version: String,
    /// Native packaging ABI tag; native adapter compatibility must be checked too.
    pub abi_tag: String,
    /// Interpreter inside the image, including any image-owned symlink target.
    pub executable: ContainerPath,
    /// Unique code directories, searched in order after stdlib and before site-packages.
    pub import_paths: Vec<ContainerPath>,
}

/// Versioned scalar semantics, independent of descriptor and Python API versions.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Interface {
    /// Function v1 supports `lance.scalar`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Semantic version; this implementation accepts version 1.
    pub version: ScalarVersion,
}

/// Image files containing individual V5 Arrow IPC Schema messages.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SchemaPaths {
    /// Ordered fields for each input RecordBatch.
    pub input: ContainerPath,
    /// Exactly one field describing the returned Array.
    pub output: ContainerPath,
    /// Fields for the single initialization row; an empty schema is valid.
    pub initialization: ContainerPath,
}

/// Conditions under which the same row produces a stable result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResultStability {
    /// Stable when input, initialization, and all result-relevant context are fixed.
    InputAndContext,
    /// May vary between calls, for example through a mutable external service.
    PerCall,
}

/// Side effects across import, initialization, application, and cleanup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SideEffects {
    /// No externally observable side effects; execution retry may be allowed.
    None,
    /// Retry additionally requires a real external idempotency mechanism.
    Idempotent,
    /// Automatic re-execution is forbidden.
    NonIdempotent,
}

/// Producer declarations; they do not implement caching or exactly-once effects.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Behavior {
    /// Result reproducibility under fixed input and relevant context.
    pub result_stability: ResultStability,
    /// Effects of every lifecycle phase, including import and close.
    pub side_effects: SideEffects,
}

/// Required CUDA host interface; user-space toolkit libraries belong to the image.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CudaRequirement {
    /// CUDA capability contract; this implementation accepts version 1.
    pub version: CudaVersion,
    /// Image toolkit major.minor version; this is not a host toolkit request.
    pub cuda_toolkit: String,
    /// Minimum host driver major.minor.patch version, compared numerically.
    pub driver_min: String,
    /// Exact supported GPU architecture set, expressed as major.minor strings.
    pub compute_capabilities: Vec<String>,
    /// Positive number of required devices.
    pub device_count: u32,
}

/// An additional required capability; unknown types are incompatible.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum Capability {
    /// CUDA driver/device interfaces must be allocated and checked by the worker.
    #[serde(rename = "lance.cuda")]
    Cuda(CudaRequirement),
}

/// Image requirements that can be checked before importing user code.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Requirements {
    /// Linux major.minor.patch minimum; distro suffixes are not part of the format.
    pub kernel_min: String,
    /// Empty for a CPU function without additional required capabilities.
    pub capabilities: Vec<Capability>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Fields {
    format_version: FormatVersion,
    python: Python,
    python_api: PythonApi,
    entrypoint: String,
    interface: Interface,
    schemas: SchemaPaths,
    requires: Requirements,
    behavior: Behavior,
}

/// A structurally validated Function v1 descriptor from `lance.function.v1`.
///
/// Construction rejects duplicate JSON keys, unknown fields and versions, invalid
/// image paths, and unknown required capabilities. It does not load Python.
/// Runtime compatibility and image content still require explicit checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Descriptor(Fields);

impl Descriptor {
    /// Parse the JSON string stored in the image config label.
    pub fn from_json(bytes: impl AsRef<[u8]>) -> Result<Self> {
        let fields: Fields = json::decode(
            json::parse(bytes.as_ref(), "Function descriptor")?,
            "Function descriptor",
        )?;
        if fields.interface.kind != "lance.scalar" {
            return Err(Error::incompatible(format!(
                "unsupported Function interface {:?}",
                fields.interface.kind
            )));
        }
        if fields.python.implementation != "cpython" {
            return Err(Error::incompatible(format!(
                "unsupported Python implementation {:?}",
                fields.python.implementation
            )));
        }
        numeric_version::<3>(&fields.python.version, "python.version")?;
        if fields.python.abi_tag.is_empty()
            || !fields
                .python
                .abi_tag
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(Error::incompatible(format!(
                "invalid python.abi_tag {:?}",
                fields.python.abi_tag
            )));
        }
        let mut paths = HashSet::new();
        for path in &fields.python.import_paths {
            if !paths.insert(path) {
                return Err(Error::incompatible(format!(
                    "duplicate python.import_paths {:?}",
                    path.as_str()
                )));
            }
        }
        let Some((module, factory)) = fields.entrypoint.split_once(':') else {
            return Err(Error::incompatible("entrypoint must be module:factory"));
        };
        if module.split('.').any(str::is_empty)
            || factory.is_empty()
            || fields.entrypoint.chars().any(|character| {
                character.is_whitespace() || matches!(character, '/' | '\\' | '\0')
            })
            || factory.contains([':', '.'])
        {
            return Err(Error::incompatible(format!(
                "invalid module:factory entrypoint {:?}",
                fields.entrypoint
            )));
        }
        numeric_version::<3>(&fields.requires.kernel_min, "requires.kernel_min")?;
        for capability in &fields.requires.capabilities {
            match capability {
                Capability::Cuda(cuda) => {
                    if cuda.device_count == 0 || cuda.compute_capabilities.is_empty() {
                        return Err(Error::incompatible(
                            "lance.cuda requires version 1, positive device_count, and nonempty compute_capabilities",
                        ));
                    }
                    numeric_version::<2>(&cuda.cuda_toolkit, "cuda_toolkit")?;
                    numeric_version::<3>(&cuda.driver_min, "driver_min")?;
                    let mut architectures = HashSet::new();
                    for architecture in &cuda.compute_capabilities {
                        let version = numeric_version::<2>(architecture, "compute_capabilities")?;
                        if !architectures.insert(version) {
                            return Err(Error::incompatible(format!(
                                "duplicate CUDA compute capability {architecture:?}"
                            )));
                        }
                    }
                }
            }
        }
        Ok(Self(fields))
    }

    /// Serialize the validated descriptor for an image config label.
    ///
    /// Artifact identity must always hash manifest bytes, never this serialization.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(&self.0)
            .map_err(|error| Error::incompatible(format!("descriptor serialization: {error}")))
    }

    /// Image interpreter and import search order.
    pub fn python(&self) -> &Python {
        &self.0.python
    }

    /// Version governing descriptor structure and image loading rules.
    pub fn format_version(&self) -> FormatVersion {
        self.0.format_version
    }

    /// Version governing Python calls and lifecycle, independent of native ABI.
    pub fn python_api(&self) -> PythonApi {
        self.0.python_api
    }

    /// Import module and top-level factory attribute in `module:factory` form.
    pub fn entrypoint(&self) -> &str {
        &self.0.entrypoint
    }

    /// Versioned scalar interface.
    pub fn interface(&self) -> &Interface {
        &self.0.interface
    }

    /// Container paths of the three schema files.
    pub fn schemas(&self) -> &SchemaPaths {
        &self.0.schemas
    }

    /// Required host kernel and additional capabilities.
    pub fn requires(&self) -> &Requirements {
        &self.0.requires
    }

    /// Producer-declared result and side-effect behavior.
    pub fn behavior(&self) -> &Behavior {
        &self.0.behavior
    }

    /// Check kernel, interpreter identity and allocated CUDA capabilities.
    ///
    /// The worker must supply observed interpreter facts and must additionally
    /// preflight the actual adapter's libc/native dependencies. Matching a Python
    /// ABI tag alone cannot establish native compatibility.
    pub fn check_runtime(&self, support: &RuntimeSupport) -> Result<()> {
        if support.kernel
            < numeric_version::<3>(&self.0.requires.kernel_min, "requires.kernel_min")?
        {
            return Err(Error::incompatible(format!(
                "kernel {:?} is below {}",
                support.kernel, self.0.requires.kernel_min
            )));
        }
        let python = &self.0.python;
        if support.python_implementation != python.implementation
            || support.python_version != python.version
            || support.python_abi_tag != python.abi_tag
        {
            return Err(Error::incompatible(format!(
                "image Python does not match declared {}/{}/{}",
                python.implementation, python.version, python.abi_tag
            )));
        }
        for capability in &self.0.requires.capabilities {
            match capability {
                Capability::Cuda(requirement) => {
                    let available = support.cuda.as_ref().ok_or_else(|| {
                        Error::incompatible("required lance.cuda capability is unavailable")
                    })?;
                    if available.driver
                        < numeric_version::<3>(&requirement.driver_min, "driver_min")?
                    {
                        return Err(Error::incompatible(format!(
                            "CUDA driver {:?} is below {}",
                            available.driver, requirement.driver_min
                        )));
                    }
                    if available.compute_capabilities.len() < requirement.device_count as usize {
                        return Err(Error::incompatible(format!(
                            "CUDA requires {} devices, allocated {}",
                            requirement.device_count,
                            available.compute_capabilities.len()
                        )));
                    }
                    let architectures = requirement
                        .compute_capabilities
                        .iter()
                        .map(|value| numeric_version::<2>(value, "compute_capabilities"))
                        .collect::<Result<HashSet<_>>>()?;
                    if available
                        .compute_capabilities
                        .iter()
                        .any(|architecture| !architectures.contains(architecture))
                    {
                        return Err(Error::incompatible(
                            "allocated CUDA device architecture is outside the declared exact set",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Check declared paths against a complete immutable rootfs, without importing code.
    ///
    /// The launcher must also prevent its writable/device/config mounts from
    /// shadowing these paths or their resolved targets. Permission and executable
    /// loading checks use the actual runtime identity in the later launcher stage.
    pub fn check_image_paths(&self, image: &ImageRoot) -> Result<()> {
        image.resolve(&self.0.python.executable, PathKind::File)?;
        for path in &self.0.python.import_paths {
            image.resolve(path, PathKind::Directory)?;
        }
        for path in [
            &self.0.schemas.input,
            &self.0.schemas.output,
            &self.0.schemas.initialization,
        ] {
            image.resolve(path, PathKind::File)?;
        }
        Ok(())
    }
}

/// Worker-observed platform facts used for pre-import compatibility checking.
#[derive(Debug, Clone)]
pub struct RuntimeSupport {
    /// Parsed Linux major.minor.patch; normalize any host distro suffix first.
    pub kernel: [u64; 3],
    /// Observed interpreter implementation, normally `cpython`.
    pub python_implementation: String,
    /// Observed full interpreter version.
    pub python_version: String,
    /// Observed native packaging ABI tag.
    pub python_abi_tag: String,
    /// Allocated CUDA capability; `None` means unavailable or not granted.
    pub cuda: Option<CudaSupport>,
}

/// Observed driver and the architectures of all devices granted to this instance.
#[derive(Debug, Clone)]
pub struct CudaSupport {
    /// Parsed host driver version; image toolkit libraries are not substituted.
    pub driver: [u64; 3],
    /// One exact major.minor architecture for each allocated device.
    pub compute_capabilities: Vec<[u64; 2]>,
}

fn numeric_version<const N: usize>(value: &str, property: &str) -> Result<[u64; N]> {
    let mut result = [0; N];
    let mut parts = value.split('.');
    for number in &mut result {
        let part = parts.next().ok_or_else(|| {
            Error::incompatible(format!(
                "{property} {value:?} requires {N} numeric components"
            ))
        })?;
        if part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(Error::incompatible(format!(
                "{property} {value:?} requires {N} nonnegative numeric components"
            )));
        }
        *number = part
            .parse()
            .map_err(|error| Error::incompatible(format!("{property} {value:?}: {error}")))?;
    }
    if parts.next().is_some() {
        return Err(Error::incompatible(format!(
            "{property} {value:?} requires {N} numeric components"
        )));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorCode;
    use rstest::rstest;
    use serde_json::{Value, json};

    fn descriptor() -> Value {
        json!({
            "format_version": 1,
            "python": {"implementation": "cpython", "version": "3.12.11", "abi_tag": "cp312", "executable": "/opt/python/bin/python3", "import_paths": ["/opt/function/code"]},
            "python_api": 1,
            "entrypoint": "app.scale:create",
            "interface": {"type": "lance.scalar", "version": 1},
            "schemas": {"input": "/opt/function/input.arrow", "output": "/opt/function/output.arrow", "initialization": "/opt/function/initialization.arrow"},
            "requires": {"kernel_min": "4.18.0", "capabilities": []},
            "behavior": {"result_stability": "input_and_context", "side_effects": "none"}
        })
    }

    #[test]
    fn roundtrip_and_runtime_preflight() {
        let descriptor = Descriptor::from_json(descriptor().to_string()).unwrap();
        assert_eq!(
            Descriptor::from_json(descriptor.to_json().unwrap()).unwrap(),
            descriptor
        );
        let mut runtime = RuntimeSupport {
            kernel: [6, 1, 0],
            python_implementation: "cpython".into(),
            python_version: "3.12.11".into(),
            python_abi_tag: "cp312".into(),
            cuda: None,
        };
        descriptor.check_runtime(&runtime).unwrap();
        runtime.kernel = [4, 9, 200];
        let error = descriptor.check_runtime(&runtime).unwrap_err();
        assert_eq!(error.code, ErrorCode::Incompatible);
        assert!(error.message.contains("kernel"));
        runtime.kernel = [6, 1, 0];
        runtime.python_abi_tag = "cp311".into();
        assert!(
            descriptor
                .check_runtime(&runtime)
                .unwrap_err()
                .message
                .contains("Python")
        );
    }

    #[rstest]
    #[case::format("/format_version", json!(2))]
    #[case::api("/python_api", json!(2))]
    #[case::interface("/interface/version", json!(2))]
    #[case::unknown_interface("/interface/type", json!("lance.udtf"))]
    #[case::implementation("/python/implementation", json!("pypy"))]
    #[case::python_version("/python/version", json!("3.12"))]
    #[case::abi("/python/abi_tag", json!("cp312.bad"))]
    #[case::relative("/python/executable", json!("python3"))]
    #[case::path_alias("/python/import_paths", json!(["/opt/code", "/opt/./code/"]))]
    #[case::entrypoint("/entrypoint", json!("app:create:other"))]
    #[case::kernel("/requires/kernel_min", json!("4.18.0-custom"))]
    #[case::unknown_capability("/requires/capabilities", json!([{"type":"future.cpu", "version":1}]))]
    #[case::behavior("/behavior/side_effects", json!("unknown"))]
    fn rejects_invalid_fields(#[case] pointer: &str, #[case] value: Value) {
        let mut input = descriptor();
        *input.pointer_mut(pointer).unwrap() = value;
        let error = Descriptor::from_json(input.to_string()).unwrap_err();
        assert_eq!(error.code, ErrorCode::Incompatible);
        assert!(!error.message.is_empty());
    }

    #[rstest]
    #[case::root("")]
    #[case::python("/python")]
    #[case::interface("/interface")]
    #[case::schemas("/schemas")]
    #[case::requires("/requires")]
    #[case::behavior("/behavior")]
    fn rejects_unknown_fields(#[case] pointer: &str) {
        let mut input = descriptor();
        input
            .pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("unknown".into(), json!(true));
        let error = Descriptor::from_json(input.to_string()).unwrap_err();
        assert_eq!(error.code, ErrorCode::Incompatible);
        assert!(error.message.contains("unknown field"));
    }

    #[test]
    fn rejects_duplicate_escaped_keys() {
        let input = descriptor().to_string().replacen(
            "\"format_version\":1",
            "\"format_version\":1,\"format_\\u0076ersion\":1",
            1,
        );
        let error = Descriptor::from_json(input).unwrap_err();
        assert_eq!(error.code, ErrorCode::Incompatible);
        assert!(error.message.contains("duplicate JSON key"));
    }

    #[test]
    fn cuda_requires_allocated_compatible_devices() {
        let mut input = descriptor();
        input["requires"]["capabilities"] = json!([{"type":"lance.cuda", "version":1, "cuda_toolkit":"12.4", "driver_min":"550.54.14", "compute_capabilities":["8.0", "8.6"], "device_count":2}]);
        let descriptor = Descriptor::from_json(input.to_string()).unwrap();
        let mut runtime = RuntimeSupport {
            kernel: [6, 1, 0],
            python_implementation: "cpython".into(),
            python_version: "3.12.11".into(),
            python_abi_tag: "cp312".into(),
            cuda: None,
        };
        assert!(
            descriptor
                .check_runtime(&runtime)
                .unwrap_err()
                .message
                .contains("unavailable")
        );
        runtime.cuda = Some(CudaSupport {
            driver: [550, 54, 14],
            compute_capabilities: vec![[8, 0], [8, 6]],
        });
        descriptor.check_runtime(&runtime).unwrap();
        runtime.cuda.as_mut().unwrap().compute_capabilities[0] = [9, 0];
        assert!(
            descriptor
                .check_runtime(&runtime)
                .unwrap_err()
                .message
                .contains("architecture")
        );
        input["requires"]["capabilities"][0]["extra"] = json!(1);
        assert!(
            Descriptor::from_json(input.to_string())
                .unwrap_err()
                .message
                .contains("unknown field")
        );
    }
}
