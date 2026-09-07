// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Parse and validate Lance Function v1 artifacts without importing user code.
//!
//! [`Artifact`] resolves content-addressed OCI metadata; [`Descriptor`] defines
//! the image-owned Python interface; [`Schemas`] preserves the Arrow contract.
//! Registry access, layer application, isolation, and execution belong to the
//! caller. Validation never launches Python or installs image dependencies.
//!
//! ```
//! use lance_function::{Digest, ErrorCode};
//!
//! let version = Digest::of(b"exact manifest bytes");
//! assert!(version.as_str().starts_with("sha256:"));
//! assert_eq!(ErrorCode::Incompatible.as_str(), "incompatible");
//! ```

#![doc = include_str!("../README.md")]

mod artifact;
mod descriptor;
mod error;
mod json;
mod path;
mod schema;

pub use artifact::{Artifact, BlobDescriptor, Digest, Platform};
pub use descriptor::{
    Behavior, Capability, CudaRequirement, CudaSupport, CudaVersion, Descriptor, FormatVersion,
    Interface, Python, PythonApi, Requirements, ResultStability, RuntimeSupport, ScalarVersion,
    SchemaPaths, SideEffects,
};
pub use error::{Error, ErrorCode, Result};
pub use path::{ContainerPath, ImageRoot, PathKind};
pub use schema::{ExtensionTypes, Schemas, read_schema};
