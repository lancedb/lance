# 模式格式规范（Schema Format Specification）

## 概述

模式描述了 Lance 表的结构，包括所有字段、它们的数据类型和元数据。
模式使用逻辑类型系统，其中数据类型表示为映射到 Apache Arrow 数据类型的字符串。
模式中的每个字段都有一个唯一标识符（字段 ID），以支持健壮的模式演进和版本跟踪。

!!! note

    逻辑类型目前正在通过讨论 [#5864](https://github.com/lance-format/lance/discussions/5864) 进行简化。
    提议的变更包括将编码特定的变体（例如 `large_string` 和 `string`、`large_binary` 和 `binary`）
    合并为具有运行时优化的单一逻辑类型。此外，[#5817](https://github.com/lance-format/lance/discussions/5817) 提议添加
    `string_view` 和 `binary_view` 类型。本文档描述当前的实现。

## 数据类型

Lance 支持一套全面的数据类型，映射到 Apache Arrow 类型。
数据类型在模式中以字符串形式表示，可以分为几个类别。

### 原始类型（Primitive Types）

| 逻辑类型 | Arrow 类型 | 描述 |
|---|---|---|
| `null` | `Null` | 空类型（无值） |
| `bool` | `Boolean` | 布尔类型（true/false） |
| `int8` | `Int8` | 有符号 8 位整数 |
| `uint8` | `UInt8` | 无符号 8 位整数 |
| `int16` | `Int16` | 有符号 16 位整数 |
| `uint16` | `UInt16` | 无符号 16 位整数 |
| `int32` | `Int32` | 有符号 32 位整数 |
| `uint32` | `UInt32` | 无符号 32 位整数 |
| `int64` | `Int64` | 有符号 64 位整数 |
| `uint64` | `UInt64` | 无符号 64 位整数 |

### 浮点类型（Floating Point Types）

| 逻辑类型 | Arrow 类型 | 描述 |
|---|---|---|
| `halffloat` | `Float16` | IEEE 754 半精度浮点数（16 位） |
| `float` | `Float32` | IEEE 754 单精度浮点数（32 位） |
| `double` | `Float64` | IEEE 754 双精度浮点数（64 位） |

### 字符串和二进制类型（String and Binary Types）

| 逻辑类型 | Arrow 类型 | 描述 |
|---|---|---|
| `string` | `Utf8` | 可变长度 UTF-8 编码字符串 |
| `binary` | `Binary` | 可变长度二进制数据 |
| `large_string` | `LargeUtf8` | 可变长度 UTF-8 字符串（支持大偏移量） |
| `large_binary` | `LargeBinary` | 可变长度二进制数据（支持大偏移量） |

### 十进制类型（Decimal Types）

十进制类型支持任意精度的数值。格式为：`decimal:<bit_width>:<precision>:<scale>`

| 逻辑类型 | Arrow 类型 | 精度 | 示例 |
|---|---|---|---|
| `decimal:128:P:S` | `Decimal128` | 最多 38 位 | `decimal:128:10:2`（共 10 位数字，小数点后 2 位） |
| `decimal:256:P:S` | `Decimal256` | 最多 76 位 | `decimal:256:20:5` |

- **精度（Precision, P）**：总位数（Decimal128 为 1-38，Decimal256 最多 76）
- **标度（Scale, S）**：小数点后的位数（0 ≤ S ≤ P）

### 日期和时间类型（Date and Time Types）

| 逻辑类型 | Arrow 类型 | 描述 |
|---|---|---|
| `date32:day` | `Date32` | 日期（自纪元以来的天数） |
| `date64:ms` | `Date64` | 日期（自纪元以来的毫秒数） |
| `time32:s` | `Time32` | 时间（自午夜以来的秒数） |
| `time32:ms` | `Time32` | 时间（自午夜以来的毫秒数） |
| `time64:us` | `Time64` | 时间（自午夜以来的微秒数） |
| `time64:ns` | `Time64` | 时间（自午夜以来的纳秒数） |
| `duration:s` | `Duration` | 持续时间（秒） |
| `duration:ms` | `Duration` | 持续时间（毫秒） |
| `duration:us` | `Duration` | 持续时间（微秒） |
| `duration:ns` | `Duration` | 持续时间（纳秒） |

### 时间戳类型（Timestamp Types）

时间戳类型表示一个时间点，可以包含时区信息。
格式：`timestamp:<unit>:<timezone>`

- **单位（Unit）**：`s`（秒）、`ms`（毫秒）、`us`（微秒）、`ns`（纳秒）
- **时区（Timezone）**：IANA 时区字符串（例如 `UTC`、`America/New_York`）或 `-` 表示无时区

示例：
- `timestamp:us:UTC` - UTC 时区的微秒精度时间戳
- `timestamp:ms:America/New_York` - America/New_York 时区的毫秒精度时间戳
- `timestamp:ns:-` - 无时区的纳秒精度时间戳

### 复合类型（Complex Types）

#### 结构体类型（Struct Type）

结构体是包含具有异构类型的命名字段的容器。

| 逻辑类型 | Arrow 类型 | 描述 |
|---|---|---|
| `struct` | `Struct` | 包含多个命名字段的复合类型 |

结构体字段在模式中表示为子字段。

包含结构体的示例模式：
```protobuf
Field {
    name: "address"
    type: "struct"
    children: [
        Field { name: "street", type: "string" },
        Field { name: "city", type: "string" },
        Field { name: "zip", type: "int32" }
    ]
}
```

#### 列表类型（List Types）

列表表示单一类型的可变长度数组。

| 逻辑类型 | Arrow 类型 | 描述 |
|---|---|---|
| `list` | `List` | 可变长度值列表 |
| `list.struct` | `List(Struct)` | 可变长度结构体值列表 |
| `large_list` | `LargeList` | 可变长度列表（支持大偏移量） |
| `large_list.struct` | `LargeList(Struct)` | 可变长度结构体值列表（大偏移量） |

元素类型通过子字段指定。

#### 固定大小列表类型（Fixed-Size List Types）

固定大小列表在模式定义时具有预先确定的大小。
格式：`fixed_size_list:<element_type>:<size>`

| 逻辑类型 | 描述 | 示例 |
|---|---|---|
| `fixed_size_list:float:128` | 128 个浮点数的固定大小列表 | 向量嵌入（128 维） |
| `fixed_size_list:int32:10` | 10 个整数的固定大小列表 | |

特殊扩展类型：
- `fixed_size_list:lance.bfloat16:256` - bfloat16 值的固定大小列表

#### 固定大小二进制类型（Fixed-Size Binary Type）

具有预定字节大小的固定大小二进制数据。
格式：`fixed_size_binary:<size>`

| 逻辑类型 | 描述 | 示例 |
|---|---|---|
| `fixed_size_binary:16` | 16 字节的固定大小二进制 | MD5 哈希 |
| `fixed_size_binary:32` | 32 字节的固定大小二进制 | SHA-256 哈希 |

#### 字典类型（Dictionary Type）

使用独立键和值的字典编码数据。
格式：`dict:<value_type>:<key_type>:<ordered>`

- **值类型（Value type）**：字典值的类型
- **键类型（Key type）**：用于字典索引的类型（通常为 int8、int16 或 int32）
- **有序（Ordered）**：布尔值，指示字典值是否已排序（目前尚未完全支持）

示例：`dict:string:int16:false` - 使用 int16 键的字典编码字符串

#### 映射类型（Map Type）

以结构化格式存储的键值对。

| 逻辑类型 | Arrow 类型 | 描述 |
|---|---|---|
| `map` | `Map` | 键值对（目前仅支持无序键） |

映射的键和值类型通过子字段指定。

### 扩展类型（Extension Types）

Lance 支持自定义扩展类型，在 Arrow 类型之上提供语义含义。

#### Blob 类型

表示外部存储的大型二进制数据。

| 逻辑类型 | 描述 |
|---|---|
| `blob` | 具有外部存储引用的大型二进制数据 |
| `json` | 以二进制形式存储的 JSON 编码数据 |

Blob 类型以大型二进制数据的形式存储，元数据描述存储位置。

#### BFloat16 类型

脑浮点数（Brain Float, bfloat16）是一种针对机器学习优化的 16 位浮点格式。
在固定大小列表中使用：`fixed_size_list:lance.bfloat16:SIZE`

## 字段 ID（Field IDs）

字段 ID 是分配给模式中每个字段的唯一整数标识符。
它们对于健壮的模式演进至关重要，因为它们允许字段在不破坏引用的情况下被重命名或重新排序。

### 字段 ID 分配

**初始分配（深度优先顺序）：**
创建表时，按深度优先顺序为所有字段分配字段 ID，从 0 开始。

嵌套字段通过 protobuf 消息中的 `parent_id` 字段关联。例如，如果字段 "c"（id: 2）是一个包含字段 "x"、"y"、"z" 的结构体，那些子字段的 `parent_id` 为 2。顶级字段的 `parent_id` 为 -1。

嵌套结构示例：
```
字段顺序：a, b, c.x, c.y, c.z, d

分配的 ID 及父级关系：
- a: 0 (parent_id: -1)
- b: 1 (parent_id: -1)
- c: 2 (parent_id: -1, struct 类型)
- c.x: 3 (parent_id: 2)
- c.y: 4 (parent_id: 2)
- c.z: 5 (parent_id: 2)
- d: 6 (parent_id: -1)
```

注意：`parent_id` 为 -1 表示顶级字段。对于嵌套字段，`parent_id` 引用父字段的 ID。子字段通过 `parent_id` 引用其父级，而不是在 protobuf 消息中作为单独的 "children" 数组存储（尽管 Rust 内存表示为方便起见维护了一个 children 向量）。

**新字段分配（递增方式）：**
当后续添加字段时（例如通过模式演进），它们会按递增方式获得下一个可用的 ID。
这保留了字段添加的历史记录。

### 字段 ID 属性

- **不可变（Immutable）**：一旦分配，字段的 ID 永远不会改变
- **唯一（Unique）**：表内每个字段都有唯一的 ID
- **稳定（Stable）**：ID 在模式演进操作中保持不变
- **稀疏（Sparse）**：模式演进后，字段 ID 可能不会形成连续序列

### 使用字段 ID

在格式内部引用字段时，应使用字段 ID 而非字段名称或位置。

## 字段元数据（Field Metadata）

字段可以携带额外的键值对元数据，用于配置编码、主键行为和其他属性。

### 主键元数据

主键配置由 Field 消息中的两个 protobuf 字段处理：
- **unenforced_primary_key**（bool）：该字段是否为主键的一部分
- **unenforced_primary_key_position**（uint32）：主键排序中的位置（有序时从 1 开始，无序时为 0）

有关主键配置的详细讨论，请参阅表格式概述中的[非强制主键](index.md#unenforced-primary-key)。

### 编码元数据

列编码配置使用 `lance-encoding:` 前缀指定。
有关可用编码的完整详细信息，请参阅[文件格式编码规范](../file/encoding.md)。

### Arrow 扩展类型元数据

自定义 Arrow 扩展类型可能在 `ARROW:extension:` 命名空间下有元数据
（例如 `ARROW:extension:name`）。

## 模式 Protobuf 定义

模式使用 protobuf 消息进行序列化。关键消息包括：

### Field 消息

```protobuf
%%% proto.message.lance.file.Field %%%
```

Field 消息包含：
- **id**：唯一字段标识符（int32）
- **name**：字段名称（string）
- **type**：字段类型枚举（PARENT、REPEATED 或 LEAF）
- **logical_type**：逻辑类型字符串表示（string）- 例如 "int64"、"struct"、"list"
- **nullable**：字段是否可以为空（bool）
- **parent_id**：嵌套字段的父字段 ID；顶级字段为 -1（int32）
- **metadata**：用于额外配置的键值对（map<string, bytes>）
- **unenforced_primary_key**：该字段是否为主键的一部分（bool）
- **unenforced_primary_key_position**：主键排序中的位置（uint32，0 = 无序）

### Schema 消息

完整的模式表示为顶级字段加上元数据的集合。

## 模式演进（Schema Evolution）

字段 ID 使高效的模式演进成为可能：

- **添加列**：分配新的字段 ID 并添加到模式
- **删除列**：从模式中移除字段；其 ID 在某些系统中可能被重用
- **重命名列**：更改字段名称；ID 保持不变
- **重新排序列**：更改模式中的字段顺序；ID 保持不变
- **类型演进**：数据类型可以更改。这可能需要重写列中的数据，具体取决于类型更改的方式。

字段 ID 的使用确保即使模式随时间变化，数据文件也能被正确解析。

## 示例模式

以下示例使用简化的字段结构表示。在实际的 protobuf 格式中，`type` 指字段类型枚举（PARENT/REPEATED/LEAF），`logical_type` 包含数据类型字符串表示。

### 简单表

```
Field {
    id: 0
    name: "id"
    logical_type: "int64"
    nullable: false
    parent_id: -1
}
Field {
    id: 1
    name: "name"
    logical_type: "string"
    nullable: true
    parent_id: -1
}
Field {
    id: 2
    name: "created_at"
    logical_type: "timestamp:us:UTC"
    nullable: true
    parent_id: -1
}
```

### 嵌套结构

```
Field {
    id: 0
    name: "id"
    logical_type: "int64"
    nullable: false
    parent_id: -1  // 顶级字段
}
Field {
    id: 1
    name: "user"
    logical_type: "struct"
    nullable: true
    parent_id: -1  // 顶级字段
}
Field {
    id: 2
    name: "name"
    logical_type: "string"
    nullable: true
    parent_id: 1  // 嵌套在 "user" struct (id: 1) 下
}
Field {
    id: 3
    name: "email"
    logical_type: "string"
    nullable: true
    parent_id: 1  // 嵌套在 "user" struct (id: 1) 下
}
Field {
    id: 4
    name: "tags"
    logical_type: "list"
    nullable: true
    parent_id: -1  // 顶级字段
}
Field {
    id: 5
    name: "item"
    logical_type: "string"
    nullable: true
    parent_id: 4  // 嵌套在 "tags" list (id: 4) 下
}
```

### 带向量嵌入

```
Field {
    id: 0
    name: "id"
    logical_type: "int64"
    nullable: false
    parent_id: -1  // 顶级字段
    unenforced_primary_key: true
    unenforced_primary_key_position: 1  // 主键中的有序位置
}
Field {
    id: 1
    name: "text"
    logical_type: "string"
    nullable: true
    parent_id: -1  // 顶级字段
}
Field {
    id: 2
    name: "embedding"
    logical_type: "fixed_size_list:lance.bfloat16:384"
    nullable: true
    parent_id: -1  // 顶级字段
}
```

## 类型转换参考

在逻辑类型和 Arrow 类型之间转换时，Lance 使用以下映射：

| Arrow 类型 | 逻辑类型格式 |
|---|---|
| `Arrow::Null` | `null` |
| `Arrow::Boolean` | `bool` |
| `Arrow::Int8` 到 `Int64` | `int8`, `int16`, `int32`, `int64` |
| `Arrow::UInt8` 到 `UInt64` | `uint8`, `uint16`, `uint32`, `uint64` |
| `Arrow::Float16` | `halffloat` |
| `Arrow::Float32` | `float` |
| `Arrow::Float64` | `double` |
| `Arrow::Utf8` | `string` |
| `Arrow::LargeUtf8` | `large_string` |
| `Arrow::Binary` | `binary` |
| `Arrow::LargeBinary` | `large_binary` |
| `Arrow::Decimal128(p, s)` | `decimal:128:p:s` |
| `Arrow::Decimal256(p, s)` | `decimal:256:p:s` |
| `Arrow::Date32` | `date32:day` |
| `Arrow::Date64` | `date64:ms` |
| `Arrow::Time32(TimeUnit)` | `time32:s`, `time32:ms` |
| `Arrow::Time64(TimeUnit)` | `time64:us`, `time64:ns` |
| `Arrow::Timestamp(unit, tz)` | `timestamp:unit:tz` |
| `Arrow::Duration(unit)` | `duration:s`, `duration:ms`, `duration:us`, `duration:ns` |
| `Arrow::Struct` | `struct` |
| `Arrow::List(Element)` | `list` 或 `list.struct`（当元素为 Struct 时） |
| `Arrow::LargeList(Element)` | `large_list` 或 `large_list.struct` |
| `Arrow::FixedSizeList(Element, Size)` | `fixed_size_list:type:size` |
| `Arrow::FixedSizeBinary(Size)` | `fixed_size_binary:size` |
| `Arrow::Dictionary(KeyType, ValueType)` | `dict:value_type:key_type:false` |
| `Arrow::Map` | `map` |
