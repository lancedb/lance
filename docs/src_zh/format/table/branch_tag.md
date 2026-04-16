# 分支和标签规范（Branch and Tag Specification）

## 概述

Lance 支持分支（Branch）和标签（Tag）功能，用于管理多个独立的版本历史和创建对特定版本的命名引用。
分支支持并行开发工作流，而标签为重要版本提供稳定的命名引用。

## 分支（Branching）

### 分支名称

分支名称必须遵循以下验证规则：

1. 不能为空
2. 不能以 `/` 开头或结尾
3. 不能包含连续的 `//`
4. 不能包含 `..` 或 `\`
5. 段（Segment）只能包含字母数字字符、`.`、`-`、`_`
6. 不能以 `.lock` 结尾
7. 不能命名为 `main`（保留给主分支）

### 分支元数据路径

分支元数据存储在数据集根目录的 `_refs/branches/{branch-name}.json` 中。
由于分支名称支持使用 `/` 字符的层级命名，文件名中的 `/` 被 URL 编码为 `%2F`，以区分目录分隔符（例如 `bugfix/issue-123` 变为 `bugfix%2Fissue-123.json`）：

```
{dataset_root}/
    _refs/
        branches/
            feature-a.json
            bugfix%2Fissue-123.json  # 注意: '/' 编码为 '%2F'
```

### 分支元数据文件格式

每个分支元数据文件是一个包含以下字段的 JSON 文件：

| JSON 键            | 类型   | 可选 | 描述                                                                       |
|------------------|--------|------|----------------------------------------------------------------------------|
| `parent_branch`  | string | 是   | 创建此分支的来源分支名称。`null` 表示从主分支创建。                          |
| `parent_version` | number |      | 创建此分支时父分支的版本号。                                                |
| `create_at`      | number |      | 分支创建时的 Unix 时间戳（自纪元以来的秒数）。                               |
| `manifest_size`  | number |      | 初始清单文件的大小（字节）。                                                |

### 分支数据集布局

每个分支数据集在技术上是源数据集的[浅克隆](layout.md#shallow-clone)。
分支数据集使用数据集根目录中的 `tree/` 目录组织：

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

命名分支将其特定版本的文件存储在 `tree/{branch_name}/` 下，类似于 GitHub 分支路径约定。
它直接使用分支名称来构成路径，
这意味着 `/` 会创建一个逻辑子目录（例如 `bugfix/issue-123`、`feature/user-auth`）：

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

## 标签（Tagging）

### 标签名称

标签名称必须遵循以下验证规则：

1. 不能为空
2. 只能包含字母数字字符、`.`、`-`、`_`
3. 不能以 `.` 开头或结尾
4. 不能以 `.lock` 结尾
5. 不能包含连续的 `..`

注意，标签名称不支持 `/` 字符，这与分支名称不同。

### 标签存储

标签以 JSON 文件的形式存储在数据集根目录的 `_refs/tags/` 下：

```
{dataset_root}/
    _refs/
        tags/
            v1.0.0.json
            v1.1.0.json
            production.json
```

无论标签引用哪个分支，标签始终存储在根数据集层级。

### 标签文件格式

每个标签文件是一个包含以下字段的 JSON 文件：

| JSON 键           | 类型   | 可选 | 描述                                                                   |
|-----------------|--------|------|------------------------------------------------------------------------|
| `branch`        | string | 是   | 被标记的分支名称。`null` 或缺失表示主分支。                              |
| `version`       | number |      | 该分支内被标记的版本号。                                                |
| `manifest_size` | number |      | 清单文件的大小（字节）。用于高效的清单加载。                              |
