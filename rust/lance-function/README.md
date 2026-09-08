# Lance Function contracts

`lance-function` validates the metadata and Arrow interface of a Linux Function
v1 image. It is independent of catalogs, job registries, Python runtimes, and
network transports. An artifact's identity is the SHA-256 of the exact selected
platform manifest bytes, not a hash of a parsed or normalized descriptor.

```rust,no_run
use lance_function::{Artifact, Error, ErrorCode, ExtensionTypes, ImageRoot, Platform, Schemas};

let document = std::fs::read("image/index.json")?;
let platform = Platform {
    os: "linux".into(),
    architecture: "amd64".into(),
    variant: None,
};
let artifact = Artifact::resolve(&document, &platform, |blob| {
    let hash = blob.digest().as_str().trim_start_matches("sha256:");
    std::fs::read(format!("image/blobs/sha256/{hash}"))
        .map_err(|error| Error::new(ErrorCode::Incompatible, error.to_string()))
})?;

// The caller verifies and unpacks every layer before exposing this immutable root.
let root = ImageRoot::new("cache/complete-rootfs")?;
artifact.descriptor().check_image_paths(&root)?;
let max_schema_bytes = 1024 * 1024; // This caller allows up to 1 MiB per schema file.
let schemas = Schemas::from_image(
    artifact.descriptor(), &root, &ExtensionTypes::default(), max_schema_bytes,
)?;
assert_eq!(schemas.output().fields().len(), 1);
# Ok::<(), Box<dyn std::error::Error>>(())
```

The parser verifies fetched manifest/config sizes and digests, preserves config
properties, resolves nested OCI indices in declaration order, and checks index
platform claims against image config. Unknown OCI properties are accepted;
unknown Function descriptor properties, required capabilities, and versions are
rejected. Duplicate JSON keys are rejected before conversion to maps.

Schema files contain exactly one encapsulated Arrow IPC Schema message with V5
metadata. The reader checks raw schema and field metadata for duplicate keys and
checks semantic type validity before Arrow conversion. No unknown extension type
is accepted by default. Workers register validators for supported extension
storage types and metadata. Scalar output has one field; initialization data has
one row, including the zero-field case. Validation performs no coercions.

`Schemas::from_image` requires a caller-selected byte budget for each schema file
and rejects oversized files before reading their contents. It reads and parses
files one at a time. The budget limits serialized input, not decoded Arrow memory;
it is a caller resource policy, not a Function v1 format limit. Callers supplying
bytes through `Schemas::from_ipc` are responsible for bounding their input.

`ImageRoot` resolves POSIX image paths with container-root symlink semantics.
The caller must keep the rootfs immutable during validation and use; the resolver
is not a defense against concurrent filesystem mutation. The launcher must check
that runtime mounts do not shadow declared paths or their resolved targets and
must perform interpreter/native adapter preflight under the execution identity.

This crate does not fetch layers, unpack archives, install dependencies, launch
containers, import user modules, retry execution, or publish results. Those are
worker responsibilities. Runtime requirements are checked against caller-supplied
observed facts, including allocated CUDA devices; a matching Python ABI tag does
not establish libc or native dependency compatibility.

OCI structures follow [image-spec v1.1.1](https://github.com/opencontainers/image-spec/tree/v1.1.1).
The Function v1 contract is under development and has not been declared stable.
