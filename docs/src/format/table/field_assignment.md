# Field Assignment

Field assignment is optional snapshot metadata that records whether each live row has a current logical assignment for a tracked field. It is independent of Arrow validity: an explicitly written NULL is assigned, while an omitted or invalidated value is unassigned. Descriptors are keyed by stable field ID, so renaming a field does not change its state.

The manifest stores one descriptor per tracked field:

```protobuf
message FieldAssignmentFile {
  string path = 1;
  uint64 size_bytes = 2;
  optional uint32 base_id = 3;
}

message FieldAssignmentState {
  int32 field_id = 1;
  FieldAssignmentFile root = 2;
}
```

`FieldAssignmentFile.path` is relative to the owning dataset root. `base_id`, when present, resolves through the manifest base-path map and allows shallow clones to reuse immutable assignment objects.

## Root and bitmap objects

Each descriptor points to an immutable protobuf root under `_field_assignments/roots/<field-id>/`. The root contains sorted, unique fragment entries:

```protobuf
message FieldAssignmentRoot {
  repeated FieldAssignmentFragment fragments = 1;
}

message FieldAssignmentFragment {
  uint64 fragment_id = 1;
  uint64 physical_rows = 2;

  oneof state {
    bool all_assigned = 3;
    FieldAssignmentFile partial = 4;
  }
}
```

An absent fragment entry means no physical rows are assigned. `all_assigned=true` means every physical row is assigned; `all_assigned=false` is invalid. A partial entry references a portable serialized 32-bit Roaring bitmap under `_field_assignments/bitmaps/<field-id>/<fragment-id>/`. Bitmap values are physical row offsets, must be within `physical_rows`, and must represent a non-empty, non-full set.

Roots and bitmap pages are immutable. A new snapshot reuses every unchanged object and writes only the roots and pages needed for changed fragment states. Cleanup retains objects referenced by any retained manifest or descendant branch and removes objects that are no longer reachable.

## Snapshot and mutation rules

- Enabling tracking requires an explicit all-assigned or all-unassigned state. Readers never infer state from NULL values or data-file layout.
- A successful logical field write assigns exactly the supplied rows, including rows for which the supplied Arrow value is NULL. An append that omits a tracked field leaves its new rows unassigned.
- Explicit invalidation clears assignment in the same transaction as any accompanying value writes. Writing and invalidating the same field in one mutation is invalid.
- Deletion files mask live rows without rewriting assignment objects. Physical rewrites preserve membership; a rewrite that changes row addresses must remap every affected partial state from the old physical offsets to the new offsets.
- Renames retain the stable field ID. A schema cast that replaces a stable field ID transfers the descriptor to the replacement ID. Dropping a field drops its descriptor. Time travel, restore, and clone use the assignment descriptors of the selected snapshot.
- Data Overlay `FieldCoverage` remains the mutation-local set of cells supplied by an overlay file. Committing an overlay assigns those covered cells, but `FieldCoverage` is not persisted or queried as snapshot-level field assignment state.

## Query contract

`is_assigned(field)` resolves its single direct field reference to a stable field ID while planning. It returns a non-null Boolean value for every output row. Unknown fields, untracked fields, and non-field arguments are planning errors.

A fallback execution reads the field root and required partial pages, combines physical row address fragment IDs and offsets with the assignment state, and emits an Arrow BooleanArray. Ordinary projection continues to return stored Arrow values even for unassigned rows. A plan that does not contain `is_assigned` must not read assignment roots or bitmaps.

The logical function is encoded in Substrait with extension URN `urn:lance:extension:functions`. Consumers must register the Lance `is_assigned` extension before accepting such a plan.

## Compatibility

Manifest writer feature bit `1 << 8` indicates that field assignment state is present. The bit is writer-only: older readers may ignore the descriptors and perform ordinary Arrow projection, but a writer that does not understand the bit must refuse every mutation because it cannot preserve assignment state.
