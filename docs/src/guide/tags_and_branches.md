# Manage Tags and Branches

Lance provides Git-like tag and branch capabilities through the `LanceDataset.tags` and `LanceDataset.branches` properties.

## Tags
Tags label specific versions within a branch's history.

`Tags` are particularly useful for tracking the evolution of datasets,
especially in machine learning workflows where datasets are frequently updated.
For example, you can `create`, `update`,
and `delete` or `list` tags.

The `reference` parameter (used in `create`, `update`, and `checkout_version`) accepts:

- An **integer**: version number in the **current branch** (e.g., `1`)
- A **string**: tag name (e.g., `"stable"`)
- A **tuple** `(branch_name, version)`: a specific version in a named branch
  - `(None, 2)` means version 2 on the main branch
  - `("main", 2)` means version 2 on the main branch (explicit)
  - `("experiment", 3)` means version 3 on the experiment branch
  - `("branch-name", None)` means the latest version on that branch

!!! note

    Creating or deleting tags does not generate new dataset versions.
    Tags exist as auxiliary metadata stored in a separate directory.

```python
import lance
import pyarrow as pa

ds = lance.dataset("./tags.lance")
print(len(ds.versions()))
# 2
print(ds.tags.list())
# {}
ds.tags.create("v1-prod", (None, 1))
print(ds.tags.list())
# {'v1-prod': {'version': 1, 'created_at': ..., 'updated_at': ..., 'manifest_size': ...}}
ds.tags.update("v1-prod", (None, 2))
print(ds.tags.list())
# {'v1-prod': {'version': 2, 'created_at': ..., 'updated_at': ..., 'manifest_size': ...}}
ds.tags.delete("v1-prod")
print(ds.tags.list())
# {}
print(ds.tags.list_ordered())
# []
ds.tags.create("v1-prod", (None, 1))
print(ds.tags.list_ordered())
# [('v1-prod', {'version': 1, 'created_at': ..., 'updated_at': ..., 'manifest_size': ...})]
ds.tags.update("v1-prod", (None, 2))
print(ds.tags.list_ordered())
# [('v1-prod', {'version': 2, 'created_at': ..., 'updated_at': ..., 'manifest_size': ...})]
ds.tags.delete("v1-prod")
print(ds.tags.list_ordered())
# []
```

!!! note

    Tagged versions are exempted from the `LanceDataset.cleanup_old_versions()`
    process.

    To remove a version that has been tagged, you must first `LanceDataset.tags.delete()`
    the associated tag. 

## Branches

Branches manage parallel lines of dataset evolution. You can create a branch from an existing version or tag, read and write to it independently, and checkout different branches. You can `create`, `delete`, `list`, and `checkout` branches.

The `reference` parameter works the same as for Tags (see above).

!!! note

    Creating or deleting branches does not generate new dataset versions.
    New versions are created by writes (append/overwrite/index operations).

    Each branch maintains its own linear version history, so version numbers may overlap across branches. Use `(branch_name, version_number)` tuples as global identifiers for operations like `checkout_version` and `tags.create`.

    "main" is a reserved branch name. Lance uses "main" to identify the default branch.

### Create and checkout branches
```python
import lance
import pyarrow as pa

# Open dataset
ds = lance.dataset("/tmp/test.lance")

# Create branch from latest version (default: current branch's latest)
experiment_branch = ds.create_branch("experiment")
experimental_data = pa.Table.from_pydict({"a": [11], "b": [12]})
lance.write_dataset(experimental_data, experiment_branch, mode="append")

# Create tag on the latest version of the experimental branch
ds.tags.create("experiment-rc", ("experiment", None))

# Checkout by tag name
experiment_rc = ds.checkout_version("experiment-rc")
# Checkout the latest version of the experimental branch by tuple
experiment_latest = ds.checkout_version(("experiment", None))

# Create a new branch from a tag
new_experiment = ds.create_branch("new-experiment", "experiment-rc")
```

### List branches
```python
print(ds.branches.list())
# {'experiment': {...}, 'new-experiment': {...}}
```

### Delete a branch
```python
# Ensure the branch is no longer needed before deletion
ds.branches.delete("experiment")
print(ds.branches.list_ordered(order="desc"))
# {'new-experiment': {'parent_branch': 'experiment', 'parent_version': 2, 'create_at': ..., 'manifest_size': ...}, ...}
```

!!! note

    Branches hold references to data files. Lance ensures that cleanup does not delete files still referenced by any branch.

    Delete unused branches to allow their referenced files to be cleaned up by `cleanup_old_versions()`.

## Shallow clones and cleanup on the source

A shallow clone created with `shallow_clone()` does not copy data.
Its manifest points back at the source dataset and reads the source's data files in place.

By default, `shallow_clone()` writes a pin under `_refs/clones/` on the source for the cloned version and stores that pin id on the clone.
`cleanup_old_versions()` on the source keeps pinned versions the same way it keeps tagged versions. It does not open clone URIs.
Creating a pin needs write access to the source. With read-only access, pass `retention="require_tag"` after the source owner tags the version. Keep that tag for as long as the clone reads source files.
Cleanup writes a marker under `_refs/gc_markers/` before deleting a version. The marker stays after deletion so `shallow_clone()` cannot reopen that version.
Markers and pin `branchId` values use the branch incarnation id, so a recreated branch with the same name does not inherit them. Existing clones keep working until their pins are unregistered.

Unregister the pin after deleting or detaching the clone:

```python
clone = source.shallow_clone("s3://experiments/test-variant", 3)
pin_id = clone.source_pin_id

# After the clone is gone
source.clone_pins.unregister(pin_id)
```

An unused pin is removed when cleanup marks the version before clone commit starts, or when the commit path verifies no manifest landed. If the commit outcome is unknown, the pin stays. A leaked pin wastes storage. Removing a pin for a live clone can lose data.

Pins also block branch deletion. `delete_branch` is refused while pins target that incarnation. Unregister the listed pins, then delete again. A refused delete leaves the incarnation closed to new shallow clones. `force_delete_branch` overrides the refusal and breaks those clones.
Tag-retained clones do not block branch deletion. Prefer pinned clones when the source branch may be deleted.

!!! warning

    Shallow clones created before this behavior existed have no pin and are not protected.
    Re-clone them, or create a tag on the source version they read so cleanup retains it.

    Libraries that predate pins ignore `_refs/clones/` during cleanup and can still delete
    files a new clone needs. Run cleanup with a library that understands pins, or keep a
    tag on the source version while older clients may vacuum the source.
