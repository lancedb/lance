# R-Tree 索引

R-Tree 索引是一种静态的、不可变的二维空间索引（2D Spatial Index）。它基于边界框（Bounding Box）组织数据，旨在加速基于矩形的裁剪操作。

它设计为多级层次结构：叶子页存储被索引几何体的元组 `(bbox, id=rowid)`；分支页聚合子边界框并存储指向子页的 `id=pageid`；单个根页包围整棵树。从概念上讲，它可以被视为 B+ 树（B+-tree）向多维对象的扩展，其中边界框充当空间裁剪的键。

索引使用紧凑构建策略（Packed-Build），先对元素排序，然后分组到固定大小的叶子页中。

紧凑构建流程如下：
- 根据排序算法对元素（边界框）进行排序。
- 将连续的元素打包到 `page_size` 个条目的叶子页中；然后自底向上通过聚合子页边界框构建父页。

## 排序

排序不会改变 R-Tree 的数据结构，但对性能至关重要。目前实现了 Hilbert 排序，但该设计可扩展到其他空间排序算法。

### Hilbert 曲线排序

Hilbert 排序使用空间填充的 Hilbert 曲线对二维元素施加线性顺序，以最大化两个轴上的局部性。这改善了叶子聚类，有利于查询裁剪。

Hilbert 排序分三步执行：

1. **全局边界框**：计算所有训练索引元素的全局边界框 `[xmin_g, ymin_g, xmax_g, ymax_g]`。
2. **归一化并计算 Hilbert 值**：
    - 对于每个元素边界框 `[xmin_i, ymin_i, xmax_i, ymax_i]`，计算其中心：
        - `cx = (xmin_i + xmax_i) / 2`
        - `cy = (ymin_i + ymax_i) / 2`
    - 使用全局边界框将中心映射到每轴 16 位的网格。令 `W = xmax_g - xmin_g`，`H = ymax_g - ymin_g`。归一化整数坐标为：
        - `xi = round(((cx - xmin_g) / W) * (2^16 - 1))`
        - `yi = round(((cy - ymin_g) / H) * (2^16 - 1))`
    - 如果全局宽度或高度实际为零，则相应轴视为退化，所有元素设为 `0`（排序退化为另一轴上的一维排序）。
    - 对于 `[0 .. 2^16-1] × [0 .. 2^16-1]` 中的每个 `(xi, yi)`，使用标准的二维 Hilbert 算法计算 32 位 Hilbert 值。伪代码如下（`bits = 16`）：
      ```
      fn hilbert_value(x, y, bits):
          # x, y: integers in [0 .. 2^bits - 1]
          h = 0
          mask = (1 << bits) - 1
 
          for s from bits-1 down to 0:
              rx = (x >> s) & 1
              ry = (y >> s) & 1
              d  = ((3 * rx) XOR ry) << (2 * s)
              h  = h | d
 
              if ry == 0:
                  if rx == 1:
                      x = (~x) & mask
                      y = (~y) & mask
                  swap(x, y)
 
          return h
      ```
      - 结果 `h` 作为元素的 Hilbert 值存储（类型为 `u32`，`bits = 16`）。
3. **排序**：按 Hilbert 值排序元素。

## 索引详情

```protobuf
%%% proto.message.RTreeIndexDetails %%%
```

## 存储布局

R-Tree 索引由两个文件组成：

1. `page_data.lance` - 存储所有页面（叶子页、分支页），以重复的 `(bbox, id)` 元组形式自底向上写入（先写叶子，然后写分支层级）
2. `nulls.lance` - 存储具有空值行的序列化 RowAddrTreeMap

### 页文件 Schema

| 列     | 类型     | 可空  | 描述                                                                                                                                                                                                                                                           |
|:-------|:---------|:------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `bbox` | RectType | false | 类型为 [geoarrow-rs](https://github.com/geoarrow/geoarrow-rs) 定义的 Rect RectType；物理存储为 Struct<xmin: Float64, ymin: Float64, xmax: Float64, ymax: Float64>。表示节点边界框（叶子：元素边界框；分支：子节点聚合）。                                           |
| `id`   | UInt64   | false | 复用 `id` 列，在叶子页中存储 `rowid`，在分支页中存储 `pageid`                                                                                                                                                                                                   |

### 空值文件 Schema

| 列      | 类型   | 可空  | 描述                                              |
|:--------|:-------|:------|:--------------------------------------------------|
| `nulls` | Binary | false | 具有空值/无效几何体的行的序列化 RowAddrTreeMap       |

### Schema 元数据

以下可选键可供实现使用，存储在 schema 元数据中：

| 键          | 类型   | 描述                                   |
|:------------|:-------|:---------------------------------------|
| `page_size` | String | 每页的页大小                            |
| `num_pages` | String | 写入的总页数                            |
| `num_items` | String | 索引中非空叶子元素的数量                 |
| `bbox`      | String | JSON 序列化的数据集全局 BoundingBox      |

### 查询遍历

此索引将多级层次 R-Tree 结构序列化为遵循上述 schema 的单个页文件。查找时，读取器使用以下算法计算每个页的偏移量并重建层次结构以进行遍历。

偏移量从元数据的 `num_items` 和 `page_size` 推导如下：

- 叶子：`leaf_pages = ceil(num_items / page_size)`；叶子 `i` 的 `page_offset = i * page_size`。
- 分支：令 `level_offset` 为当前层级的起始偏移量，实际上表示所有较低层级的元素总数；令 `prev_pages` 为下一层级的页数；`level_pages = ceil(prev_pages / page_size)`。对于分支 `j`，`page_offset = j * page_size + level_offset`。
- 迭代各层级直到剩余一个页面；根节点是最后一个页面，其 `pageid = num_pages - 1`。
- 页长度：收集所有页偏移量后，通过下一个偏移量的差值计算每个 `page_len`；对于最终页面（根节点），`page_len = page_file_total_rows - page_offset`（其中 `page_file_total_rows` 是 `page_data.lance` 中的总行数）。

遍历从根节点（`pageid = num_pages - 1`）开始：

- 如果 `page_offset < num_items`（叶子），读取 `[page_offset .. page_offset + page_len)` 范围内的元素，输出与查询边界框匹配的候选 `rowid`。
- 否则（分支），下降到边界框与查询边界框匹配的子节点。
- 持续直到没有更多页面需要访问；输出的 `rowid` 的并集构成待评估的候选集。

## 加速查询

R-Tree 索引通过返回匹配边界框的候选集来加速以下查询类型。精确的几何验证必须由执行引擎完成。

| 查询类型       | 描述                       | 操作                                   | 结果类型 |
|:---------------|:---------------------------|:---------------------------------------|:---------|
| **Intersects** | `St_Intersects(col, geom)` | 通过边界框相交裁剪候选                   | AtMost   |
| **Contains**   | `St_Contains(col, geom)`   | 通过边界框包含裁剪候选                   | AtMost   |
| **Within**     | `St_Within(col, geom)`     | 通过边界框内含关系裁剪候选               | AtMost   |
| **Touches**    | `St_Touches(col, geom)`    | 通过边界框接触关系裁剪候选               | AtMost   |
| **Crosses**    | `St_Crosses(col, geom)`    | 通过边界框交叉关系裁剪候选               | AtMost   |
| **Overlaps**   | `St_Overlaps(col, geom)`   | 通过边界框重叠关系裁剪候选               | AtMost   |
| **Covers**     | `St_Covers(col, geom)`     | 通过边界框覆盖关系裁剪候选               | AtMost   |
| **CoveredBy**  | `St_Coveredby(col, geom)`  | 通过边界框被覆盖关系裁剪候选             | AtMost   |
| **IsNull**     | `col IS NULL`              | 返回空值文件中记录的行                   | Exact    |
