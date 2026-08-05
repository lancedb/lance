# Branch, Tag, and Clone Reference Specification

## Overview

Lance supports branching and tagging for managing multiple independent version histories and creating named references to specific versions.
Branches enable parallel development workflows, while tags provide stable named references for important versions.
Clone pins record that a shallow clone at another URI reads this dataset's files. Cleanup keeps those files while a pin exists.
All three are stored under `_refs/` at the dataset root.

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

Branch metadata is stored at `_refs/branches/{branch-name}.json` in the dataset root.
Since branch names support hierarchical naming with `/` characters, the `/` is URL-encoded as `%2F` in the filename to distinguish it from directory separators (e.g., `bugfix/issue-123` becomes `bugfix%2Fissue-123.json`):

```
{dataset_root}/
    _refs/
        branches/
            feature-a.json
            bugfix%2Fissue-123.json  # Note: '/' encoded as '%2F'
```

### Branch Metadata File Format

Each branch metadata file is a JSON file with the following fields:

| JSON Key         | Type   | Optional | Description                                                                    |
|------------------|--------|----------|--------------------------------------------------------------------------------|
| `parentBranch`   | string | Yes      | Name of the branch this was created from. `null` indicates branched from main. |
| `parentVersion`  | number |          | Version number of the parent branch at the time this branch was created.       |
| `createAt`       | number |          | Unix timestamp (seconds since epoch) when the branch was created.              |
| `manifestSize`   | number |          | Size of the initial manifest file in bytes.                                    |
| `metadata`       | object | Yes      | String key/value metadata map. If absent, it is treated as an empty object.    |

### Branch Dataset Layout

Each branch dataset is technically a [shallow clone](layout.md#shallow-clone) of the source dataset.
Branch datasets are organized using the `tree/` directory at the dataset root:

```
{dataset_root}/
    tree/
        {branch_name}/
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

Named branches store their version-specific files under `tree/{branch_name}/`, resembling the GitHub branch path convention.
It uses the branch name as is to form the path, 
which means `/` would create a logical subdirectory (e.g., `bugfix/issue-123`, `feature/user-auth`):

```
{dataset_root}/
    tree/
        feature-a/
            _versions/
                1.manifest
                2.manifest
        bugfix/
            issue-123/
                _versions/
                    1.manifest
```

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

Tags are stored as JSON files under `_refs/tags/` at the dataset root:

```
{dataset_root}/
    _refs/
        tags/
            v1.0.0.json
            v1.1.0.json
            production.json
```

Tags are always stored at the root dataset level, regardless of which branch they reference.

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

## Clone Pin

A shallow clone stores its manifest at its own URI and reads the source's data, deletion, and index files in place through `base_paths`.
Cleanup on the source does not open clone URIs.
By default the source writes a pin under `_refs/clones/` for the cloned version, and the clone stores that pin id under `lance.clone.source_pin_id`.
A read-only source may rely on a tag the owner created instead, with no pin written.
Cleanup keeps a pinned version the same way it keeps a tagged version.

Remove the pin when the clone is deleted or detached. Several clones may pin the same version. Retention lasts until the last pin is removed.

Before deleting a version, cleanup writes a marker under `_refs/gc_markers/`, then relists pins and tags.
A pin for the current incarnation or a matching tag retains the version. Otherwise cleanup deletes it.
Tags are rechecked so a mid-cleanup `require_tag` clone still protects its version.
Markers stay after deletion, so a marked version stays closed to new shallow clones.
Marker paths and pin `branchId` values use the branch incarnation id. A recreated branch with the same name gets a new incarnation and does not inherit prior markers or pins.
Existing clones keep working until their pins are unregistered.

Branch deletion writes an incarnation-wide `deleted` marker, then lists pins and is refused while any pin targets that incarnation.
The `deleted` marker is never removed. A refused delete leaves the incarnation closed to new shallow clones. Rerun delete after unregistering the listed pins.
A force delete overrides the refusal and breaks those clones.
Tag-retained clones do not block branch deletion.
A pinned clone checks both markers, creates its pin, then checks both again before commit.
A tag-retained clone verifies the tag and checks both markers without creating a pin.

### Clone Pin Storage

Pins are stored as JSON files under `_refs/clones/` at the dataset root.
Cleanup markers live under `_refs/gc_markers/<incarnation>/<version>`:

```
{dataset_root}/
    _refs/
        clones/
            21b1c1b1a4f7f9f0d1e2c3b4a5968778.json
            8f0e1d2c3b4a5968778091a2b3c4d5e6.json
        gc_markers/
            main/
                1
                3
            a1b2c3d4e5f60718293a4b5c6d7e8f90/
                1
                deleted
```

Pins are always stored at the root dataset level, for every branch a clone reads from.
The pin file name is a random 32-character hexadecimal pin id. The same id is stored on the clone.
The main branch uses the path segment `main`. Named branches use the leaf UUID of their `BranchIdentifier`.
Path segments encode `%` as `%25` and `/` as `%2F`.
Numeric file names fence single versions. The `deleted` file name fences the whole incarnation and is written by branch deletion.

### Clone Pin File Format

Each pin file is a JSON file with the following fields:

| JSON Key    | Type   | Optional | Description                                                                 |
|-------------|--------|----------|-----------------------------------------------------------------------------|
| `id`        | string |          | Random pin id. Matches the file stem.                                       |
| `branch`    | string | Yes      | Display name of the branch that was cloned. Absent means the main branch.   |
| `branchId`  | string | Yes      | Leaf UUID of the branch incarnation. Absent means main. Used for retention. Required when `branch` is present. Readers reject a named-branch pin without it. |
| `version`   | number |          | Version of that incarnation that the clone reads from.                      |
| `createdAt` | string |          | RFC 3339 timestamp for when the pin was created.                            |
| `cloneUri`  | string | Yes      | URI of the clone dataset, for listing. Not used as the registry key.        |
