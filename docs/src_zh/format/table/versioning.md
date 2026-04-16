# 格式版本管理（Versioning）

## 特性标志（Feature Flags）

随着表格式的演进，新的特性标志会被添加到格式中。
有两个独立的字段用于检查特性标志，
取决于你是要读取还是写入表。
读取者应该检查 `reader_feature_flags` 以查看是否有它不认识的标志。
写入者应该检查 `writer_feature_flags`。如果任一方看到不认识的标志，
都应该在任何读取或写入操作时返回 "unsupported" 错误。

## 当前特性标志

<style>
.feature-flags-table th:nth-child(2),
.feature-flags-table td:nth-child(2) {
  white-space: nowrap;
  min-width: 250px;
}
</style>

<div class="feature-flags-table" markdown="1">

| 标志位 | 标志名称                          | 读取者必需 | 写入者必需 | 描述                                                                                     |
|--------|----------------------------------|-----------|-----------|------------------------------------------------------------------------------------------|
| 1      | `FLAG_DELETION_FILES`           | 是        | 是        | 片段（Fragment）可能包含删除文件，记录软删除行的墓碑标记。                                    |
| 2      | `FLAG_STABLE_ROW_IDS`           | 是        | 是        | 行 ID 对于移动和更新都是稳定的。片段包含一个将行 ID 映射到行地址的索引。                        |
| 4      | `FLAG_USE_V2_FORMAT_DEPRECATED` | 否        | 否        | 文件使用新的 v2 格式写入。此标志已弃用，不再使用。                                            |
| 8      | `FLAG_TABLE_CONFIG`             | 否        | 是        | 表配置存在于 Manifest 中。                                                                |
| 16     | `FLAG_BASE_PATHS`               | 是        | 是        | 数据集使用多个基础路径（用于浅克隆或多基础路径数据集）。                                       |

</div>

位值为 32 及以上的标志是未知的，将导致实现拒绝数据集并返回 "unsupported" 错误。
