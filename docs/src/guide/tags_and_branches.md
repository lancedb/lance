# Manage Tags and Branches

Lance, much like Git, employs the `LanceDataset.tags`
property to label specific versions within a dataset's history.

`Tags` are particularly useful for tracking the evolution of datasets,
especially in machine learning workflows where datasets are frequently updated.
For example, you can `create`, `update`,
and `delete` or `list` tags.

!!! note

    Creating or deleting tags does not generate new dataset versions.
    Tags exist as auxiliary metadata stored in a separate directory.

```python
import lance
ds = lance.dataset("./tags.lance")
print(len(ds.versions()))
# 2
print(ds.tags.list())
# {}
ds.tags.create("v1-prod", 1)
print(ds.tags.list())
# {'v1-prod': {'version': 1, 'manifest_size': ...}}
ds.tags.update("v1-prod", 2)
print(ds.tags.list())
# {'v1-prod': {'version': 2, 'manifest_size': ...}}
ds.tags.delete("v1-prod")
print(ds.tags.list())
# {}
print(ds.tags.list_ordered())
# []
ds.tags.create("v1-prod", 1)
print(ds.tags.list_ordered())
# [('v1-prod', {'version': 1, 'manifest_size': ...})]
ds.tags.update("v1-prod", 2)
print(ds.tags.list_ordered())
# [('v1-prod', {'version': 2, 'manifest_size': ...})]
ds.tags.delete("v1-prod")
print(ds.tags.list_ordered())
# []
```

!!! note

    Tagged versions are exempted from the `LanceDataset.cleanup_old_versions()`
    process.

    To remove a version that has been tagged, you must first `LanceDataset.tags.delete()`
    the associated tag. 

---

### Branches

Lance provides `LanceDataset.branches` to manage parallel lines of dataset evolution.
Branches let you create an independent line from an existing version or tag, read/write on that branch,
and switch between branches.

!!! note

    Creating or deleting branches does not generate new dataset versions as well.
    New versions are created by writes (append/overwrite/index operations).

    Each branch has a linear history of versions and the version number is strictly increased, which means 
    the version number of different branches could overlap. We need to use (branch, version_number) as a global 
    version identifier for cases like checkout, tags.create.

    "main" is a reversed name of branch. Lance use "main" to identify the main branch.

#### Create and check out branches
```python
import lance

# Open dataset
ds = lance.dataset("/tmp/test.lance")

# Create branch from latest version
experiment_branch = ds.create_branch("experiment")
experimental_data = pa.Table.from_pydict({"a": [11], "b": [12]})
branch2 = lance.write_dataset(experimental_data, experiment_branch, mode="append")

# Create tag on the latest version of the experimental branch
ds.tags().create("experiment-rc", "experiment")

# Checkout tag from the main branch
experiment_rc = ds.checkout_version("experiment-rc")
# Or we can checkout the latest version of the experimental branch
experiment_latest = ds.checkout_version(("experiment", None))

# Create a new branch from the tag on experiment branch
new_experiment = ds.create_branch("new-experiment", "experiment-rc")
```

#### List branches
```python
print(ds.branches.list())
# {'experiment': {'parent_branch': None, 'parent_version': 1, 'create_at': ..., 'manifest_size': ...}, ...}
# {'new-experiment': {'parent_branch': 'experiment', 'parent_version': 2, 'create_at': ..., 'manifest_size': ...}, ...}
```

#### Delete a branch
```python
# Ensure the branch is no longer needed before deletion
ds.branches.delete("experiment")
print(ds.branches.list_ordered(order="desc"))
# {'new-experiment': {'parent_branch': 'experiment', 'parent_version': 2, 'create_at': ..., 'manifest_size': ...}, ...}
```

!!! note

    Branches establish references between files. Lance ensures that the cleanup process does not delete files 
    that are still referenced.

    In practice, it is advisable to configure branches with appropriate cleanup strategies to allow referenced files 
    to be cleaned up in a timely manner.