# Branch and Tag Specification

## Overview

Lance supports branching and tagging for managing multiple independent version histories and creating named references to specific versions.
Branches enable parallel development workflows, while tags provide stable named references for important versions.

## Branching

### Branch Name

Branch names must follow these validation rules:

1. Cannot be empty
2. Cannot start or end with `/`
3. Cannot contain consecutive `//`
4. Cannot contain `..` or `\`
5. Segments must contain only alphanumeric characters, `.`, `-`, `_`
6. Cannot end with `.lock`
7. Cannot be named `main` (reserved for main branch)

### Branch Metadata Path

Branch and tag metadata are published together as a complete immutable catalog under
`_refs/catalog/{epoch}.json`. The greatest numeric epoch is authoritative for both point lookup and
enumeration, so a catalog is either wholly visible or wholly absent. A delayed lower-epoch writer
cannot supersede a newer catalog. Branch names are keys in the `branches` object and therefore do
not determine a physical metadata path:

```
{dataset_root}/
    _refs/
        catalog/
            00000000000000000042.json
```

Each catalog contains the epoch encoded by its file name and complete live tag and branch maps:

```json
{
  "_mutationEpoch": 42,
  "tags": {
    "release": {
      "version": 7,
      "manifestSize": 1024
    }
  },
  "branches": {
    "feature-a": {
      "parentBranch": null,
      "parentVersion": 7,
      "createAt": 1775600000,
      "manifestSize": 1024
    }
  },
  "_legacyBaseline": {
    "tags": {},
    "branches": {}
  }
}
```

Deletion removes a name from the next complete catalog instead of publishing a per-name tombstone.
After publishing a catalog, writers retain its immediate predecessor and remove older epochs on a
best-effort basis. The predecessor remains available to a reader that listed it immediately before
the new publication; a reader whose selected object was concurrently compacted restarts discovery.
A lower epoch that finishes after compaction remains harmless and is reclaimed by the next
successful reference publication. Thus normal sustained churn retains two catalogs plus only
records from delayed writers that have not yet crossed another publication boundary.

Reference mutation lease epochs have one immutable decision: the owner atomically records a
commit-ready publication, or its successor atomically closes the expired epoch. A successor applies
every publication decision before advancing its reconciled epoch, while a closed decision rejects a
late owner. Decision records remain after other lease state is reclaimed so a suspended writer
cannot reopen an epoch whose reconciliation already completed.

When no catalog exists, readers load legacy `_refs/tags/*.json` and `_refs/branches/*.json` files.
The first catalog publication imports every legacy reference and records those flat files in the
catalog's `_legacyBaseline`. The flat files remain unchanged as a durable migration fence. Current
readers compare them with the baseline and reconcile any later create, update, or delete performed
by a released writer, so a successful legacy mutation is never silently ignored. Each catalog
publication snapshots the legacy state immediately before its atomic create and records that state
as the next baseline. A legacy mutation that completed before the publication is therefore ordered
before, and superseded by, that catalog; a mutation completed afterward differs from the baseline
and is reconciled. A legacy mutation concurrent with the publication may be ordered either way.

Operators must quiesce clients that only understand flat reference files before allowing catalog
writers to mutate references: catalog state is not mirrored back to the legacy paths. Retaining the
baseline makes rollback state recoverable and lets a current client detect and incorporate a legacy
writer that was not quiesced; no automatic migration deletes released-format state.

### Branch Metadata File Format

Each branch metadata file is a JSON object with this schema:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": [
    "parentVersion",
    "createAt",
    "manifestSize"
  ],
  "properties": {
    "parentBranch": {
      "type": ["string", "null"]
    },
    "storage": {
      "type": "object",
      "required": ["layout", "generation"],
      "properties": {
        "layout": {"const": "detached"},
        "generation": {"type": "string", "pattern": "^[0-9a-f]{32}$"}
      }
    },
    "identifier": {
      "type": "object",
      "required": ["version_mapping"],
      "properties": {
        "version_mapping": {
          "type": "array",
          "items": {
            "type": "array",
            "prefixItems": [
              {"type": "integer", "minimum": 0},
              {"type": "string", "pattern": "^[0-9a-f]{32}$"}
            ],
            "items": false
          }
        }
      }
    },
    "parentVersion": {
      "type": "integer",
      "minimum": 1
    },
    "createAt": {
      "type": "integer",
      "minimum": 0
    },
    "manifestSize": {
      "type": "integer",
      "minimum": 0
    },
    "metadata": {
      "type": "object",
      "additionalProperties": {"type": "string"}
    }
  }
}
```

`parentBranch` is `null` when the branch was created from `main`. `identifier` records the
version and UUID for each branch in the lineage. `storage` explicitly selects the physical layout
and generation; new writers use the final UUID in `identifier.version_mapping` as the detached
generation. Metadata written before branch identifiers may omit `identifier`, `storage`, and
`metadata`. An absent `storage` field identifies the legacy name-backed layout.

### Branch Dataset Layout

Each branch dataset is technically a [shallow clone](layout.md#shallow-clone) of the source dataset.
Branch datasets are stored outside the logical alias namespace:

```
{dataset_root}/
    _branch_generations/
        {generation}/
            _versions/
                *.manifest
            _transactions/
                *.txn
            _deletions/
                *.arrow
                *.bin
            _indices/
                {UUID}/
                    index.idx
```

Named branches store their version-specific files under `_branch_generations/{generation}/`.
The metadata file is the authoritative mapping from a logical branch name to its physical layout
and generation. The logical URL
`{dataset_root}/tree/{branch_name}` remains a supported way to open a branch; the reader resolves
it through the metadata before loading the physical dataset.

```
{dataset_root}/
    _branch_generations/
        34e6c4b343a84a7ca40295852ed4d5d8/
            _versions/
                1.manifest
                2.manifest
        ecde70dc5d964b3b9626a1249682be52/
            _versions/
                1.manifest
```

Deleting detached branch metadata releases its logical name immediately. If a descendant still
references the branch, its generation directory remains until no remaining branch lineage contains
that UUID. Legacy name-based directories remain readable; a referenced legacy branch cannot be
deleted until its descendants are removed. Because tag metadata records a logical branch name, a
branch referenced by a tag cannot be deleted until that tag is removed.

## Tagging

### Tag Name

Tag names must follow these validation rules:

1. Cannot be empty
2. Must contain only alphanumeric characters, `.`, `-`, `_`
3. Cannot start or end with `.`
4. Cannot end with `.lock`
5. Cannot contain consecutive `..`

Note that tag names do not support `/` characters, unlike branch names.

### Tag Storage

Tags are entries in the same root-level catalog as branches. Publishing one complete snapshot keeps
tag point reads, tag enumeration, and branch metadata on the same fenced epoch. Tags remain global
to the root dataset regardless of which branch they reference.

```
{dataset_root}/
    _refs/
        catalog/
            00000000000000000042.json
```

### Tag File Format

Each tag file is a JSON file with the following fields:

| JSON Key        | Type   | Optional | Description                                                              |
|-----------------|--------|----------|--------------------------------------------------------------------------|
| `branch`        | string | Yes      | Branch name being tagged. `null` or absent indicates main branch.        |
| `version`       | number |          | Version number being tagged within that branch.                          |
| `createdAt`     | string | Yes      | RFC 3339 timestamp for when the tag was first created.                  |
| `updatedAt`     | string | Yes      | RFC 3339 timestamp for the latest tag reference update.                 |
| `manifestSize`  | number |          | Size of the manifest file in bytes. Used for efficient manifest loading. |
| `metadata`      | object | Yes      | String key/value metadata map. If absent, it is treated as an empty object. |
