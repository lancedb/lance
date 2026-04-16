# 位图索引（Bitmap Index）

位图索引使用位数组表示值的存在与否，
为低基数列提供极快的查询性能。

## 索引详情

```protobuf
%%% proto.message.BitmapIndexDetails %%%
```

## 存储布局

位图索引由单个文件 `bitmap_page_lookup.lance` 组成，存储从值到其位图的映射。

### 文件 Schema

| 列        | 类型       | 可空 | 描述                                                       |
|-----------|------------|------|------------------------------------------------------------|
| `keys`    | {DataType} | true | 索引列中的唯一值                                            |
| `bitmaps` | Binary     | true | 序列化的 RowAddrTreeMap，包含该值出现的行地址                 |

## 加速查询

| 查询类型   | 描述                      | 操作                                |
|-----------|---------------------------|-------------------------------------|
| **Equals** | `column = value`          | 返回特定值的位图                     |
| **Range**  | `column BETWEEN a AND b`  | 合并范围内所有值的位图               |
| **IsIn**   | `column IN (v1, v2, ...)` | 合并所有指定值的位图                 |
| **IsNull** | `column IS NULL`          | 返回预计算的空值位图                 |
