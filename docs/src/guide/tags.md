# Manage Tags

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