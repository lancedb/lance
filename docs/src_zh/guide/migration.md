# 迁移指南

Lance 尽可能避免破坏性变更。目前，我们正在完善 Rust 公共 API，以便将其从实验状态中移出，并对向后兼容性做出更强的承诺。Python API 被认为是稳定的，破坏性变更通常会在最终确定之前通过警告通知 1-2 个月，以便用户有时间迁移。本页记录了版本之间的破坏性变更，并提供迁移建议。

## 5.0.0

* 默认数据存储版本从 2.0 更改为 2.1。这影响了 `DataFile` protobuf 消息中的 `column_indices` 字段。在 2.0 中，每个字段（包括非叶字段如 struct 容器和 list 容器）都被分配了一个顺序列索引。在 2.1 中，非叶字段（非打包的 struct、list 容器）被分配 `-1`，因为它们的有效性信息现在被折叠到重复/定义级别中。只有叶字段和打包的 struct 被分配列索引。

    例如，给定以下 Schema：

    ```
    x: i32, y: [f32], z: { a: i32 }
    ```

    字段（按深度优先顺序）为：

    | 字段 ID | 字段           |
    |---------|---------------|
    | 0       | `x` (i32)     |
    | 1       | `y` (list)    |
    | 2       | `y.item` (f32)|
    | 3       | `z` (struct)  |
    | 4       | `z.a` (i32)   |

    在 **2.0** 中，`column_indices` = `[0, 1, 2, 3, 4]` — 每个字段都有一个列。

    在 **2.1** 中，`column_indices` = `[0, -1, 1, -1, 2]` — 非叶字段（`y` 和 `z`）为 `-1`。

* 此更改仅影响直接构造 `DataFile` 消息的高级用户，例如在手动构建操作以传递给 `Dataset.commit` 时。正常的读写路径不受影响。

* 要恢复到 2.0 格式，在创建数据集时设置 `data_storage_version="2.0"`。

## 1.0.0

* 标量索引返回的 `SearchResult` 现在必须输出有关空值（null）的信息。它不再包含 `RowIdTreeMap`，而是包含 `NullableRowIdSet`。解析为空值的表达式必须包含在搜索结果的空值集中。这确保 `NOT` 可以正确应用于索引搜索结果。

## 0.39

* `lance` crate 不再从 `lance-arrow` 重新导出工具类，如 `RecordBatchExt` 或 `SchemaExt`。短期内，如果你依赖这些工具类，可以添加对 `lance-arrow` crate 的依赖。但我们不期望 `lance-arrow` 会变得稳定，你可能需要考虑 fork 这些工具类。

* 之前我们同时将 `Error` 和 `Result` 导出为 `lance::Error` 和 `lance::error::Error`。现在我们将其简化为仅 `lance::Error`。我们还从公共 API 中移除了一些内部错误工具类（如 `OptionExt`），且不计划重新引入。

* Python 和 Rust 的 `dataset::diff_meta` API 已被移除，取而代之的是 `dataset::delta`，它返回一个 `DatasetDelta`，通过 `list_transactions` 提供元数据差异，通过 `get_inserted_rows` 和 `get_updated_rows` 提供数据差异。

* 一些之前公开的其他小型工具类现在变为私有。不太可能有人在使用这些。如果你依赖其中任何一个，请提交一个 issue。

* `lance-namespace` Rust crate 现在拆分为包含主要 `LanceNamespace` trait 和数据模型的 `lance-namespace`，以及包含不同实现的 `lance-namespace-impls`。`DirectoryNamespace` 和 `RestNamespace` 接口已重构以更加用户友好。`DirectoryNamespace` 现在也使用 Lance ObjectStore 进行 IO，而不是直接依赖 Apache OpenDAL。
