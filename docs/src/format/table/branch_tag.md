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

Branch metadata is published as immutable fenced records under
`_refs/branches/_versions/{branch-name}/{epoch}.json`. The greatest numeric epoch is authoritative.
`_refs/branches/{branch-name}.json` mirrors the latest publication for compatibility with earlier
readers. Since branch names support hierarchical naming with `/` characters, the `/` is URL-encoded
as `%2F` in the filename and version directory (e.g., `bugfix/issue-123` becomes
`bugfix%2Fissue-123`):

```
{dataset_root}/
    _refs/
        branches/
            feature-a.json
            bugfix%2Fissue-123.json  # Note: '/' encoded as '%2F'
            _versions/
                feature-a/
                    00000000000000000042.json
                bugfix%2Fissue-123/
                    00000000000000000043.json
```

Each live fenced record contains the branch metadata below plus `_mutationEpoch`, whose value
matches the epoch in its file name. Deletion appends a tombstone instead of removing earlier
records:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["_mutationEpoch", "_deleted"],
  "properties": {
    "_mutationEpoch": {"type": "integer", "minimum": 1},
    "_deleted": {"const": true}
  },
  "additionalProperties": false
}
```

Readers ignore lower-epoch records even if a delayed writer publishes one after a newer epoch.
When no version records exist, readers load the unversioned metadata file as a legacy record.

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
    },
    "_mutationEpoch": {
      "type": "integer",
      "minimum": 1
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

Tags use the same immutable epoch publication protocol under
`_refs/tags/_versions/{tag-name}/{epoch}.json`. The greatest epoch is authoritative, and a deleted
tag is represented by the tombstone schema above. `_refs/tags/{tag-name}.json` mirrors the latest
publication for compatibility with earlier readers. Tags are always stored at the root dataset
level, regardless of which branch they reference.

```
{dataset_root}/
    _refs/
        tags/
            v1.0.0.json
            v1.1.0.json
            production.json
            _versions/
                production/
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
