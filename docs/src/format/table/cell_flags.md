# Cell Flags

Cell Flags are schema-registered, field-scoped Boolean state attached to dataset rows. A definition has a dataset-stable `flag_id`, the stable `field_id` of its owning field, and a user-visible `name` that is unique within that field. The format does not assign meaning to names.

An unrecorded flag value is `false`. Flag state is independent of Arrow values and validity: writing NULL, a non-NULL value, or the same value does not change a flag unless the mutation carries an explicit flag change.

## Manifest registry

Each manifest snapshot contains the complete current registry and the next ID to allocate. The metadata is encoded under the reserved manifest config key `lance.cell_flags.v1`. Its value is the prefix `protobuf-base64:` followed by unpadded RFC 4648 base64 of:

```protobuf
message CellFlagDefinition {
  uint32 flag_id = 1;
  int32 field_id = 2;
  string name = 3;
}

message CellFlagManifest {
  repeated CellFlagDefinition definitions = 1;
  repeated CellFlagState states = 2;
  uint32 next_flag_id = 3;
  string dataset_id = 4;
}
```

Using the existing config extension slot preserves the public generated `Manifest` API and lets readers that do not understand Cell Flags ignore the metadata while reading ordinary fields. Cell Flag-aware readers remove this reserved entry from user-visible dataset config and reject attempts to write it through the public config API.

Definitions and state descriptors are sorted by `flag_id`. IDs are never reused, including after a definition is dropped. `dataset_id` is a deterministic UUIDv8-shaped identity derived from the founding snapshot. It identifies the dataset incarnation and prevents an uncommitted transaction from being replayed into a dataset recreated at the same URI. Renaming a field preserves its definitions because they bind to `field_id`. Dropping a field removes its definitions from the new snapshot; historical snapshots retain their own registry and state.

Registering a flag initializes all current live rows explicitly to `false` by default or to a caller-supplied Boolean value. A mutation cannot create an unregistered name.

## Immutable state objects

Only flags with at least one `true` row need a `CellFlagState` descriptor. Each descriptor points to one immutable root:

```text
_cell_flags/
|-- roots/<flag-id>/<uuid>.root
`-- bitmaps/<flag-id>/<fragment-id>/<uuid>.rbm
```

```protobuf
message CellFlagState {
  uint32 flag_id = 1;
  CellFlagFile root = 2;
}

message CellFlagFile {
  string path = 1;
  uint64 size_bytes = 2;
  optional uint32 base_id = 3;
  optional bytes inline_bytes = 4;
  uint64 memory_size_bytes = 5;
}

message CellFlagRoot {
  repeated CellFlagFragment fragments = 1;
}

message CellFlagFragment {
  uint64 fragment_id = 1;
  uint64 physical_rows = 2;

  oneof state {
    bool all_set = 3;
    CellFlagFile partial = 4;
    bytes inline_partial = 5;
  }
}
```

A missing fragment entry is `Empty`; `all_set=true` is `Full`; and either partial representation is an `LCF1` adaptive bitmap of physical row offsets. Partial bitmaps must be non-empty, non-full, and within `physical_rows`. Small bitmaps may be embedded in a root. A small root may be stored inline-only in its manifest descriptor: `path` is empty, `size_bytes` is zero, and `inline_bytes` is authoritative. An optional `base_id` supplies the default base for external bitmap objects referenced inside an inline root, which lets shallow clones retain source ownership without a root object. Larger roots use an immutable object path; an external descriptor may also carry an exact inline cache whose length equals `size_bytes`.

All integers in the following binary envelopes are little-endian.

### LCF1 bitmap envelope

An `LCF1` object starts with the four ASCII bytes `LCF1`, a one-byte encoding discriminator, an eight-byte retained-memory upper bound, and the encoding payload. The retained-memory bound must not exceed 30 MiB. An external bitmap's `CellFlagFile.memory_size_bytes` must equal the envelope value.

The encoding discriminators and payloads are:

- `0`: a portable serialized 32-bit Roaring bitmap. The retained-memory value is the payload length.
- `1`: an LSB0 dense bitset. Bit `i % 8` of byte `i / 8` represents offset `i`. For a payload of `n` bytes, the retained-memory value is `16 * n + 32 * ceil(n / 8192)`.
- `2`: an eight-byte decoded length followed by a Zstandard-compressed portable Roaring bitmap. The decoded length equals the retained-memory value.
- `3`: an eight-byte decoded bitset length followed by a Zstandard-compressed LSB0 bitset. The retained-memory value uses the dense-bitset formula above with the decoded length.
- `4`: three `uint32` values `(start, step, count)`. It represents `start + i * step` for `0 <= i < count`. `step` and `count` are non-zero and the final value must fit in `uint32`. The retained-memory value is an upper bound for the reconstructed Roaring bitmap; representation normalization may make the decoded bitmap smaller than the declaration.

Writers choose the smallest available payload. Query-bound inline bitmaps use only discriminator `0` or `2` so planning retains the direct Roaring decode path.

### LCG1 root envelope

An immutable root object starts with the four ASCII bytes `LCG1`, a one-byte encoding discriminator, an eight-byte decoded protobuf length, and the payload. Discriminator `0` stores the `CellFlagRoot` protobuf bytes directly and requires the payload length to equal the decoded length. Discriminator `1` stores a single Zstandard-compressed frame whose output length must equal the decoded length. The decoded protobuf length must not exceed 32 MiB.

`CellFlagFile.memory_size_bytes` is not the protobuf length. It is the following platform-independent upper bound on the encoded input, protobuf decode, and protobuf-to-materialized-root conversion peak:

```text
encoded LCG1 length
+ decoded protobuf length
+ 512 * fragment count
+ 2 * dynamic byte length
+ 256
```

Dynamic byte length is the sum of every partial file path, partial file inline copy, and inline partial bitmap in the root. The bound must not exceed 64 MiB. Readers recompute it from the decoded root and require exact equality with the manifest descriptor.

Unchanged snapshots reuse roots and bitmap objects. A sparse flag change rewrites only the affected fragment pages and that flag's root. Queries that do not reference Cell Flags do not load these objects.

## Query contract

`cell_flag(field, name)` is a non-null Boolean Lance expression. Planning resolves the direct field reference and string name against the input snapshot to a stable `flag_id`. Unknown fields, unknown names, non-field first arguments, and non-string names are planning errors.

The expression supports projection, filters, ordering, aggregation, and ordinary Boolean composition. The planner may turn it into an exact row-address mask and combine it with deletion masks or scalar-index results. If an expression cannot be pushed down, the scan evaluates the same Boolean values as a fallback. Substrait uses the Lance function extension URN `urn:lance:extension:functions`.

## Transaction contract

Append, update, merge, and merge-insert operations may carry explicit flag changes. Each change names a registered flag and applies to the affected row set of that operation or merge action. Flag-only mutations are valid. Field writes and flag changes become visible atomically.

- Existing rows preserve every unmentioned flag.
- New rows receive `false` for every unmentioned flag.
- Matched and inserted merge-insert actions have independent flag changes.
- No value, NULL, omission, overlay coverage, or data-file rewrite infers state.

The transaction protobuf carries one typed `cell_flag_transaction` sidecar. It records the dataset incarnation, registry changes, existing-row address changes, exact state for newly written fragments, field-ID transfers used by schema casts, the operation's complete existing-row rewrite or deletion set when rebasing needs it, and a fixed-size commitment to the public operation fields. The sidecar is internal replay and conflict-detection material, is hidden by language bindings, and does not introduce arbitrary per-row keys or policy callbacks. The source-compatible native Rust carrier uses a reserved in-memory property entry; protobuf and language-binding conversion extracts it into the typed sidecar and excludes it from application transaction properties. Callers that replace application properties or regenerate a UUID on a returned native transaction must use the carrier-preserving `Transaction` methods rather than reconstructing opaque internal entries.

New-fragment partial states use the `LCF1` envelope. Existing-row changes use the `LCFR` envelope. `LCFR` starts with the four ASCII bytes `LCFR`, followed by a one-byte discriminator and its payload:

- `0`: a portable serialized 64-bit Roaring treemap. The high 32 bits of each address are the fragment ID and the low 32 bits are the physical row offset.
- `1`: a `uint32` fragment count followed by that many entries. Each entry is a strictly increasing `uint32` fragment ID, a `uint32` byte length, and one `LCF1` bitmap of local row offsets. No trailing bytes are permitted.
- `2`: an eight-byte decoded length followed by one Zstandard-compressed discriminator-`1` payload. The decoded payload must not exceed 64 MiB.

Writers compare the portable treemap and fragmented representation, choose the smaller one, and apply discriminator `2` only when Zstandard further reduces the fragmented payload.

Concurrent registry edits conflict. Row changes use the existing mutation conflict machinery and the operation's read snapshot. Atomicity does not establish application-level freshness; systems that need freshness must validate their own source revision or read set.

## Physical lifecycle

Deletes mask rows through the dataset deletion state. Compaction, reclustering, row rewrites, and other operations that change physical addresses must preserve or remap every registered flag with the same row mapping used for values. Column-only rewrites preserve flag roots directly.

Time travel and restore select the registry and roots of the chosen snapshot. Shallow clone may reference immutable flag objects through a base ID; deep clone copies them. Cleanup retains objects reachable from every retained manifest and removes only unreferenced objects.

## Compatibility

Cell Flags use writer feature bit `1 << 8`. It is set once a dataset allocates a flag ID and remains set after all current definitions are dropped so the monotonic allocator cannot be reset by an older writer.

The bit is writer-only: older readers can continue reading ordinary field values, but cannot plan `cell_flag`. A writer that does not understand the bit must reject the dataset before mutation.

This contract requires a two-stage rollout because a feature bit cannot retrofit a historical writer whose mutation paths did not all check unknown writer bits. First deploy a gate-only release that enforces the writer check at every mutation, clone, and restore entry point. After every possible writer has been upgraded, deploy or enable Cell Flag registration. Pre-gate writers are not part of the supported mixed-version set and must be removed before the first flag is registered. In the second stage, gate-aware older readers can still read ordinary fields while gate-aware writers that lack Cell Flag support reject mutation.

The first registration is disabled by default. Operators may set `LANCE_ASSUME_CELL_FLAG_WRITER_GATE_DEPLOYED` only after every writer has been upgraded to the gate-only release. Once a dataset has allocated a flag ID, subsequent registrations do not require the process assertion because the dataset has already crossed the compatibility boundary.
