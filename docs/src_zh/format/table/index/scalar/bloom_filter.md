# 布隆过滤器索引（Bloom Filter Index）

布隆过滤器（Bloom Filter）是一种概率数据结构，允许快速的成员资格测试。
它们空间效率高，可以测试一个元素是否属于某个集合。
它是一种非精确过滤器——可能包含假阳性（False Positive），但不会产生假阴性（False Negative）。

## 索引详情

```protobuf
%%% proto.message.BloomFilterIndexDetails %%%
```

## 存储布局

布隆过滤器索引在单个文件中存储基于 zone 的布隆过滤器：

1. `bloomfilter.lance` - 每个 zone 的布隆过滤器统计信息和数据

### 布隆过滤器文件 Schema

| 列                  | 类型    | 可空  | 描述                                              |
|---------------------|---------|-------|---------------------------------------------------|
| `fragment_id`       | UInt64  | false | 包含此 zone 的片段                                 |
| `zone_start`        | UInt64  | false | 片段内的起始行偏移                                  |
| `zone_length`       | UInt64  | false | 此 zone 中的行数                                   |
| `has_null`          | Boolean | false | 此 zone 是否包含空值                                |
| `bloom_filter_data` | Binary  | false | 序列化的 SBBF（Split Block Bloom Filter）数据       |

### Schema 元数据

| 键                        | 类型   | 描述                                                      |
|---------------------------|--------|-----------------------------------------------------------|
| `bloomfilter_item`        | String | 每个 zone 的预期元素数量（默认值："8192"）                   |
| `bloomfilter_probability` | String | 假阳性概率（默认值："0.00057"，约 1/1754）                   |

## 布隆过滤器规格

布隆过滤器索引使用 Split Block Bloom Filter（SBBF）实现，
该实现针对 SIMD 操作进行了优化。

### SBBF 结构

SBBF 将位数组划分为 256 位的块（Block），每个块由 8 个连续的 32 位字组成。
这种结构支持高效的 SIMD 操作和缓存友好的内存访问模式。
块布局如下：

- **块大小**：256 位（32 字节）
- **每块字数**：8 × 32 位整数
- **最小过滤器大小**：32 字节（1 个块）
- **最大过滤器大小**：128 MiB

### 哈希机制

SBBF 使用 xxHash64（seed=0）作为主哈希，结合基于盐值（Salt）的二次哈希方案：

1. **主哈希**：xxHash64(value) → 64 位哈希值
2. **块选择**：高 32 位决定使用哪个块
3. **位选择**：低 32 位与 8 个盐值组合，在块中设置 8 个位

#### 盐值

```
0x47b6137b
0x44974d91
0x8824ad5b
0xa2b7289d
0x705495c7
0x2df1424b
0x9efc4947
0x5c6bfb31
```

每个盐值在块内生成一个位位置，确保均匀分布。

### 过滤器大小算法

SBBF 根据以下参数自动确定最优过滤器大小：
- **NDV**（Number of Distinct Values，不同值数量）：预期的唯一元素数量
- **FPP**（False Positive Probability，假阳性概率）：目标错误率

实现使用二分搜索找到满足目标 FPP 的最小 log₂(bytes)，
采用 Putze 等人的缓存高效布隆过滤器公式。

#### FPP 收敛

实现使用最多 750 次泊松分布（Poisson Distribution）计算迭代来确保准确的 FPP 估计，
特别是在 NDV 接近过滤器容量的密集过滤器场景中。

### 序列化

SBBF 序列化为连续的字节数组，存储在 `bloom_filter_data` 列中：

```
[Block 0][Block 1]...[Block N-1]
```

其中每个块为 32 字节：

```
[Word 0][Word 1][Word 2][Word 3][Word 4][Word 5][Word 6][Word 7]
```

每个字为 32 位小端序整数（4 字节），具体规则：

- **总大小**：必须是 32 字节的倍数
- **字节序**：所有 32 位字均为小端序（Little-Endian）
- **块对齐**：每个块从偏移量 `i * 32` 开始
- **字偏移**：块 `i` 中的字 `j` 位于字节偏移量 `i * 32 + j * 4`

#### 示例

对于包含 2 个块的过滤器（共 64 字节）：
```
Offset  0-3:   Block 0, Word 0 (32-bit LE)
Offset  4-7:   Block 0, Word 1 (32-bit LE)
...
Offset 28-31:  Block 0, Word 7 (32-bit LE)
Offset 32-35:  Block 1, Word 0 (32-bit LE)
...
Offset 60-63:  Block 1, Word 7 (32-bit LE)
```

## 加速查询

布隆过滤器索引为以下查询类型提供非精确结果：

| 查询类型   | 描述                      | 操作                                   | 结果类型 |
|-----------|---------------------------|----------------------------------------|---------|
| **Equals** | `column = value`          | 测试值是否存在于布隆过滤器中             | AtMost  |
| **IsIn**   | `column IN (v1, v2, ...)` | 测试任一值是否存在于布隆过滤器中          | AtMost  |
| **IsNull** | `column IS NULL`          | 返回 has_null 为 true 的 zone           | AtMost  |
