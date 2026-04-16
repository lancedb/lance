# 管理标签和分支

Lance 通过 `LanceDataset.tags` 和 `LanceDataset.branches` 属性提供类 Git 的标签（Tag）和分支（Branch）功能。

## 标签（Tags）

标签用于标记分支历史中的特定版本。

`Tags` 对于跟踪数据集的演变特别有用，尤其是在数据集频繁更新的机器学习工作流中。例如，你可以 `create`、`update`、`delete` 或 `list` 标签。

`reference` 参数（用于 `create`、`update` 和 `checkout_version`）接受：

- **整数**：**当前分支**中的版本号（例如 `1`）
- **字符串**：标签名（例如 `"stable"`）
- **元组** `(branch_name, version)`：指定分支中的特定版本
  - `(None, 2)` 表示主分支上的版本 2
  - `("main", 2)` 表示主分支上的版本 2（显式指定）
  - `("experiment", 3)` 表示 experiment 分支上的版本 3
  - `("branch-name", None)` 表示该分支的最新版本

!!! note

    创建或删除标签不会生成新的数据集版本。
    标签作为辅助元数据存储在单独的目录中。

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
# {'v1-prod': {'version': 1, 'manifest_size': ...}}
ds.tags.update("v1-prod", (None, 2))
print(ds.tags.list())
# {'v1-prod': {'version': 2, 'manifest_size': ...}}
ds.tags.delete("v1-prod")
print(ds.tags.list())
# {}
print(ds.tags.list_ordered())
# []
ds.tags.create("v1-prod", (None, 1))
print(ds.tags.list_ordered())
# [('v1-prod', {'version': 1, 'manifest_size': ...})]
ds.tags.update("v1-prod", (None, 2))
print(ds.tags.list_ordered())
# [('v1-prod', {'version': 2, 'manifest_size': ...})]
ds.tags.delete("v1-prod")
print(ds.tags.list_ordered())
# []
```

!!! note

    被标记的版本将从 `LanceDataset.cleanup_old_versions()` 过程中豁免。

    要移除已被标记的版本，你必须先 `LanceDataset.tags.delete()` 删除关联的标签。

## 分支（Branches）

分支管理数据集的并行演进线路。你可以从现有版本或标签创建分支，独立地进行读写操作，以及切换不同的分支。你可以 `create`、`delete`、`list` 和 `checkout` 分支。

`reference` 参数的用法与标签相同（见上文）。

!!! note

    创建或删除分支不会生成新的数据集版本。
    新版本由写入操作（追加/覆盖/索引操作）创建。

    每个分支维护自己的线性版本历史，因此版本号可能在不同分支间重叠。使用 `(branch_name, version_number)` 元组作为 `checkout_version` 和 `tags.create` 等操作的全局标识符。

    "main" 是保留的分支名。Lance 使用 "main" 来标识默认分支。

### 创建和切换分支
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

### 列出分支
```python
print(ds.branches.list())
# {'experiment': {...}, 'new-experiment': {...}}
```

### 删除分支
```python
# Ensure the branch is no longer needed before deletion
ds.branches.delete("experiment")
print(ds.branches.list_ordered(order="desc"))
# {'new-experiment': {'parent_branch': 'experiment', 'parent_version': 2, 'create_at': ..., 'manifest_size': ...}, ...}
```

!!! note

    分支持有对数据文件的引用。Lance 确保清理不会删除任何分支仍在引用的文件。

    删除不再使用的分支，以便 `cleanup_old_versions()` 可以清理它们引用的文件。
