# Zone Map 索引

Zone Map 是一种列式数据库技术，用于谓词下推和扫描裁剪。
它将数据分成固定大小的块（称为"zone"），并为每个 zone 维护摘要统计信息
（最小值、最大值、空值计数），通过排除不可能包含匹配值的 zone 来实现高效过滤。

Zone Map 是"非精确"过滤器——它可以明确排除 zone，但可能包含需要重新检查的假阳性。

## 索引详情

```protobuf
%%% proto.message.ZoneMapIndexDetails %%%
```

## 存储布局

Zone Map 索引在单个文件中存储 zone 统计信息：

1. `zonemap.lance` - 用于查询裁剪的 zone 统计信息

### Zone 统计文件 Schema

| 列            | 类型       | 可空  | 描述                    |
|---------------|------------|-------|-------------------------|
| `min`         | {DataType} | true  | zone 中的最小值          |
| `max`         | {DataType} | true  | zone 中的最大值          |
| `null_count`  | UInt32     | false | zone 中的空值数量        |
| `nan_count`   | UInt32     | false | NaN 值数量（浮点类型）    |
| `fragment_id` | UInt64     | false | 包含此 zone 的片段       |
| `zone_start`  | UInt64     | false | 片段内的起始行偏移       |
| `zone_length` | UInt32     | false | 此 zone 中的行数         |

### Schema 元数据

| 键              | 类型   | 描述                              |
|-----------------|--------|-----------------------------------|
| `rows_per_zone` | String | 每个 zone 的行数（默认值："8192"）  |

## 加速查询

Zone Map 索引为以下查询类型提供非精确结果：

| 查询类型   | 描述                      | 操作                               | 结果类型 |
|-----------|---------------------------|------------------------------------|---------|
| **Equals** | `column = value`          | 包含 min ≤ value ≤ max 的 zone     | AtMost  |
| **Range**  | `column BETWEEN a AND b`  | 包含范围重叠的 zone                 | AtMost  |
| **IsIn**   | `column IN (v1, v2, ...)` | 包含可能含有任一值的 zone            | AtMost  |
| **IsNull** | `column IS NULL`          | 包含 null_count > 0 的 zone         | AtMost  |
