# Fragment Metadata Tree

!!! warning "Experimental"

    This layout is unstable. Readers and writers may change it without
    keeping compatibility with earlier unstable revisions. Flat manifests
    remain the default.

**This page specifies a proposed on-disk format. Support for creating or
reading tree tables is not yet available in released Lance versions.**

!!! note "Tree tables require feature flag 512 (fragment metadata tree)"

    A reader or writer that does not understand this layout must refuse the
    dataset. The flat `fragments` list is empty on a tree table, so a reader
    that ignored the flag would see an empty table.

A flat Version Manifest carries every `DataFragment`. Publishing a version and
opening a dataset then cost work proportional to the whole fragment list. The
fragment metadata tree keeps complete fragment records in immutable Lance
leaves, routes by fragment id, and buffers validated changes above the leaves.
A small commit publishes its changes without rewriting every fragment.

This page defines the storage contract. The protobuf messages are in
`protos/table.proto`. Operation semantics, validation, and conflict rules stay
with [transactions](transaction.md). A storage action is the result of a
successful native transaction, never a second implementation of its rules.

## Snapshot authority

The Version Manifest is the only snapshot authority. A tree table sets
`Manifest.fragment_metadata`, leaves `Manifest.fragments` empty, and sets flag
512 in both `reader_feature_flags` and `writer_feature_flags`. A manifest with
any of these out of agreement is invalid.

`FragmentMetadata.layout` selects exactly one layout. A descriptor with no layout
is invalid.

`FragmentMetadataTree` selects exactly one base.

- `inline_root` carries the complete `FragmentMetadataRoot` in the manifest and
  must have an empty `suffix`.
- `root_path` names an immutable checkpoint object, and `suffix` carries every
  retained change since that checkpoint.

The suffix is cumulative within one manifest. It never refers to another
manifest or another suffix. Opening a named version reads its manifest and, when
external, that one root object. It never lists objects, walks earlier manifests,
replays raw transactions, or depends on later versions.

## Objects and paths

Every object the tree references is immutable and lives under `_bt/` inside the
dataset root. Paths are stored relative to the dataset root and must begin with
`_bt/`. A reader rejects an absolute path or a path outside that prefix.

```
{dataset_root}/
    _bt/
        base/{uuid}.root     -- FragmentMetadataRoot protobuf, one per checkpoint
        node/{uuid}.node     -- FragmentMetadataNode protobuf, one per interior
        leaf/{uuid}.lance    -- Lance file holding complete fragment records
```

A `.root` file is the bare protobuf encoding of `FragmentMetadataRoot`. A `.node`
file is the bare protobuf encoding of `FragmentMetadataNode`. Neither has a
header or footer beyond the message.

## Leaf schema

A leaf is a Lance file, format version 2.1, with this schema.

```python
import pyarrow as pa

leaf_schema = pa.schema([
    pa.field("row_kind", pa.uint8(), nullable=False),
    pa.field("frag_id", pa.uint64(), nullable=False),
    pa.field("fragment_meta", pa.binary(), nullable=True),
    pa.field("path", pa.utf8(), nullable=False),
    pa.field("field_ids", pa.list_(pa.field("item", pa.int32(), nullable=False)), nullable=False),
    pa.field("column_indices", pa.list_(pa.field("item", pa.int32(), nullable=False)), nullable=False),
    pa.field("major_version", pa.uint32(), nullable=False),
    pa.field("minor_version", pa.uint32(), nullable=False),
    pa.field("file_size_bytes", pa.uint64(), nullable=False),
    pa.field("base_id", pa.uint32(), nullable=True),
])
```

`row_kind` is 0 for a FRAGMENT row and 1 for a DATA_FILE row.

A fragment is one FRAGMENT row followed by one DATA_FILE row per data file, in
the fragment's file order. A fragment with no files is a single FRAGMENT row.
Fragment ids are strictly increasing through the leaf. A fragment is never split
across leaves.

A FRAGMENT row carries `fragment_meta`, the `DataFragment` protobuf for that
fragment with `files` cleared. Its other columns hold fixed sentinels: `path`
is the empty string, `field_ids` and `column_indices` are empty lists, the
version and size columns are 0, and `base_id` is null. A DATA_FILE row carries
one `DataFile` spread across the remaining columns and a null `fragment_meta`.
`file_size_bytes` of 0 means unknown. A null `base_id` means the file lives
under the dataset's own root. Otherwise it indexes `Manifest.base_paths`, as it
does on a flat `DataFile`.

A reader rejects a leaf whose schema differs, whose encoded size differs from
the parent's `object_size`, whose non nullable columns or list elements hold
nulls, whose `row_kind` is unknown, whose fragment ids are out of order or
repeated, whose DATA_FILE row does not follow a FRAGMENT row for the same id,
whose FRAGMENT row holds inline files, or whose FRAGMENT row sentinel columns
differ from the values above. After decoding, the reader recomputes the child
summary and rejects the leaf if it differs from the parent's
`FragmentMetadataChild`.

Counts in a leaf are always known. A `physical_rows` of 0 is zero rows, not
unknown. A deletion file must carry its `num_deleted_rows`, and it may not exceed
`physical_rows`.

## Routing

Children of a root or interior are sorted by `min_key`. `min_key` is the
inclusive start of the range that child owns. The next sibling's `min_key` is
the exclusive end. The root's first child has `min_key` 0. An interior node's
first child has the `min_key` its parent assigned to that node, and its
remaining children partition that range by increasing `min_key`. Every
fragment id therefore routes to exactly one leaf. Routing a key picks the
rightmost child whose `min_key` is at or below it.

`max_key` is the largest fragment id stored in the subtree. It is a summary. It
never routes and never prunes, because a buffered insert may sit past it.

Ranges never overlap. Each buffered mutation belongs to exactly one child range.
Every child in one list has the same `height`. A leaf has `height` 0,
`num_children` 0, and `num_keys` above 0. An interior has at least two children
and a `height` one above its children. An empty child is removed from its parent
rather than kept.

`num_keys`, `total_rows`, and `visible_rows` on a child include every mutation
buffered inside that subtree. A leaf summary is recomputed from the leaf's
records on read. An interior or root summary is the sum of its children plus
its own buffered deltas. `visible_rows` never exceeds `total_rows`.

## Mutations and replay

`FragmentMetadataMutation` wraps one `FragmentAction` with a sequence number and
three count deltas. The deltas are exact. `fragment_count_delta` is 1 when the
action creates a record, minus 1 when it removes one, and 0 otherwise.
`total_rows_delta` and `visible_rows_delta` are the record after the action
minus the record before it, with an absent record counting as zero.

A reader that resolves a fragment has the record before the action and the
action itself. It recomputes the deltas and rejects a mutation whose stored
deltas differ. Summaries above a leaf are derived from these deltas, so they
are exact for a conforming writer and are checked wherever a fragment is
resolved. Materialization recomputes leaf summaries from records. Once the
rewritten leaf and its updated path are decoded, a summary mismatch is
corruption. A writer must not publish a materialization whose recomputed
summaries disagree with the snapshot totals.

The actions are applied to one fragment as follows.

| Action | Effect |
|---|---|
| `add_fragment` | Install the complete record, replacing anything at that id |
| `remove_fragment` | Remove the record at that id |
| `add_data_file` | Append the file to the end of the ordered file list |
| `remove_data_file` | Remove every file whose path matches, keeping survivor order |
| `add_deletion_file` | Set the deletion file |
| `clear_deletion_file` | Clear the deletion file |
| `replace_data_file` | On the first file whose path equals `expected_path`, set `path`, `file_size_bytes`, and `base_id` from `file`. Keep its field ids, column indices, and file version. No match is invalid |

`Fragment.files` is an ordered list and paths may repeat. `replace_data_file`
edits a slot, not a path. Two replacements that chain through a renamed path
must not be folded into one. Starting from files `[A, B]`, renaming B to A and
then replacing the first A with C yields `[C, A]`. Folding them into one
replacement of B with C yields `[A, C]` and is wrong.

To resolve a fragment, collect every mutation for its id from the suffix, the
root buffer, and the buffer of each interior on its routing path, drop any
mutation whose sequence is at or below the owning leaf's
`materialized_through_action_sequence`, sort the rest by sequence, and apply
their actions in order to the leaf record, or to nothing if the leaf has no
record.

A leaf watermark `N` means every mutation for that leaf's range with a sequence
at or below `N` has been incorporated into the leaf's records, with no holes,
and no buffer above the leaf holds a mutation for its range with a sequence at
or below `N`. That is the safety property replay depends on. A writer
establishes it by moving whole per child batches from a buffer into the child
below, never part of a fragment's pending sequence, and by setting the
watermark to the highest sequence the flush incorporated. Splits and coalesces
move messages with their ranges and preserve it.

## Sequence numbers

Every mutation carries an `action_sequence` of at least 1. Sequence 0 is never a
mutation. Each tree has one sequence namespace, and a writer never reuses a
number anywhere in the tree.

- `FragmentMetadataRoot.next_action_sequence` is at least 1. Every sequence in
  the root buffer, in every interior buffer below it, and every leaf watermark
  below it is less than this value.
- `FragmentMetadataTree.next_action_sequence` is at least the root's value.
  Every suffix sequence is at or above the root's value and below the
  descriptor's value.
- A leaf watermark of 0 means the leaf has applied nothing, so every mutation
  routed to it replays.

A reader rejects a sequence that is 0, outside its owner's range, or repeated
across the mutation sources it decoded for the operation. A reader does not
check uniqueness across subtrees it did not read. That part is a writer
invariant.

## Fragment ids

Fragment id allocation is not tree state. `Manifest.max_fragment_id` is the
fragment id allocation high-water mark for flat and tree layouts alike, and it
includes reservations.

- Absent means nothing has been allocated or reserved, and the next id is 0.
- Otherwise the next id is `max_fragment_id` plus one.
- `u32` max means the id space is exhausted, and any further allocation fails
  before publication.

A reservation advances `max_fragment_id` without creating a fragment. Reserved
but unused ids stay consumed. Deleting a fragment never recycles its id. The
value never decreases within a lineage, restore included.

A checkpoint root may be reused by later versions whose `max_fragment_id` has
moved on, so a root holds no allocation state. The tree stores fragment state
only. No fragment id stored in the tree, whether in a leaf, a child summary,
a buffer, or the suffix, may exceed the manifest's `max_fragment_id`. If
`max_fragment_id` is absent, the tree contains no fragments and no mutation
that targets a fragment. A reader rejects a tree that violates this.

## Validation on open

A reader validates before it routes or replays. A failure is a corrupt dataset,
never an empty one. The checks are as follows.

- Manifest. Descriptor present, both flags set, flat list empty, layout set.
- Descriptor. Inline root with empty suffix, or a root path. `visible_rows` at
  most `total_rows`. Sequence relationships from the section above. No suffix
  mutation target above the manifest's `max_fragment_id`.
- Root. `version` not 0 and not above the manifest version. Encoded size and
  every child `object_size` at most `hard_capacity_bytes`. Children and buffer
  valid. Derived visible rows at most derived total rows. No child `max_key`
  and no buffered mutation target above the manifest's `max_fragment_id`.
- Children. Non empty unique paths. The root's first `min_key` is 0. An
  interior's first `min_key` equals the `min_key` its parent assigned to it.
  `min_key` at most `max_key`, at most 2^32 minus 1. `object_size` not 0. Interior `byte_size` not 0. Uniform
  height. Leaf and interior shape rules from Routing. Strictly increasing,
  non overlapping ranges.
- Buffer. Each mutation has a recognized action variant and its payload, a
  target fragment id at most 2^32 minus 1, and a sequence in its owner's
  range. Sequences are
  unique across every mutation source decoded for the operation, the suffix
  and each buffer on the path included. A reader is not required to prove
  uniqueness across subtrees it did not open. Global uniqueness stays a writer
  invariant.
- Totals. `total_fragments`, `total_rows`, and `visible_rows` on the descriptor
  equal the root's derived totals plus the suffix deltas.
- Objects. Every node and leaf read is checked against its `object_size` and
  the hard capacity, and its recomputed summary must equal the parent's child
  entry.

## Byte limits

`hard_capacity_bytes` in the root is the one size rule a reader enforces. No
object under that root, the root included, may exceed it, and a reader rejects
any object or child summary that does. A single fragment that cannot fit is an
error before publication. Each checkpoint records the cap its objects satisfy.

Node, leaf, and buffer byte targets are writer policy. They live in table
configuration, not in the tree, and a writer may rebalance an existing tree
under different targets. A reader never depends on them.

## Publication

Every `_bt/` object a version depends on is written before the manifest that
names it is committed. The manifest commit through the dataset's commit handler
is the visibility boundary for both table state and fragment state. There is no
other publication step.

A failed attempt leaves objects that no manifest names. They are unreachable
and are reclaimed by cleanup. A retry validates again against the winning
version and never reuses actions validated against a different descriptor. A
lost publish response is resolved by reading the published manifest and
comparing its descriptor to the one the attempt prepared.

Byte targets and publication choices such as inlining the root or
checkpointing are writer policy and may change between versions.

## Retention and cleanup

Cleanup computes the reachable set from every retained manifest. For a tree
table that set is the manifest's `root_path` if any, every `.node` and `.lance`
reachable from the root's children, and every data file, deletion file, and
related object those fragments name. Objects under `_bt/` outside that set are
reclaimed under the same age and in progress rules as data files. An object
still named by any retained manifest is never removed.

Objects are immutable and each manifest names its own base and objects, so an
older version stays readable for as long as its manifest is retained. A
checkpoint replaces the base only in the manifest that publishes it. A clone
must preserve the lifetime of every tree object reachable from its retained
manifests. It may copy immutable objects or share them safely, as the clone
implementation decides.
