# 读取和写入数据

## 写入 Lance 数据集

如果你熟悉 [Apache PyArrow](https://arrow.apache.org/docs/python/getstarted.html)，你会发现创建 Lance 数据集非常简单。
首先使用 `lance.write_dataset` 函数写入一个 `pyarrow.Table`。

```python
import lance
import pyarrow as pa

table = pa.Table.from_pylist([{"name": "Alice", "age": 20},
                              {"name": "Bob", "age": 30}])
ds = lance.write_dataset(table, "./alice_and_bob.lance")
```

如果数据集太大无法完全加载到内存中，你可以使用 `lance.write_dataset` 流式写入数据，
该函数也支持 `pyarrow.RecordBatch` 的 `Iterator`。
在这种情况下，你需要提供一个 `pyarrow.Schema`。

```python
from typing import Iterator

def producer() -> Iterator[pa.RecordBatch]:
    """An iterator of RecordBatches."""
    yield pa.RecordBatch.from_pylist([{"name": "Alice", "age": 20}])
    yield pa.RecordBatch.from_pylist([{"name": "Bob", "age": 30}])

schema = pa.schema([
    ("name", pa.string()),
    ("age", pa.int32()),
])

ds = lance.write_dataset(producer(),
                         "./alice_and_bob.lance",
                         schema=schema, mode="overwrite")
print(ds.count_rows())  # Output: 2
```

`lance.write_dataset` 支持写入 `pyarrow.Table`、`pandas.DataFrame`、
`pyarrow.dataset.Dataset` 和 `Iterator[pyarrow.RecordBatch]`。

## 添加行

要向数据集插入数据，你可以使用 `LanceDataset.insert`
或带 `mode=append` 参数的 `lance.write_dataset`。

```python
import lance
import pyarrow as pa

table = pa.Table.from_pylist([{"name": "Alice", "age": 20},
                              {"name": "Bob", "age": 30}])
ds = lance.write_dataset(table, "./insert_example.lance")

new_table = pa.Table.from_pylist([{"name": "Carla", "age": 37}])
ds.insert(new_table)
print(ds.to_table().to_pandas())
#     name  age
# 0  Alice   20
# 1    Bob   30
# 2  Carla   37

new_table2 = pa.Table.from_pylist([{"name": "David", "age": 42}])
ds = lance.write_dataset(new_table2, ds, mode="append")
print(ds.to_table().to_pandas())
#     name  age
# 0  Alice   20
# 1    Bob   30
# 2  Carla   37
# 3  David   42
```

## 删除行

Lance 支持使用 SQL 过滤器从数据集中删除行，如[过滤下推](#filter-pushdown)中所述。
例如，要从上面的数据集中删除 Bob 的行，可以使用：

```python
import lance

dataset = lance.dataset("./alice_and_bob.lance")
dataset.delete("name = 'Bob'")
dataset2 = lance.dataset("./alice_and_bob.lance")
print(dataset2.to_table().to_pandas())
#     name  age
# 0  Alice   20
```

!!! note

    [Lance 格式是不可变的](../format/index.md)。每次写入操作都会创建数据集的新版本，
    因此用户必须重新打开数据集才能看到变更。同样，行的删除是通过在单独的删除索引中标记为已删除来实现的，
    而不是重写文件。这种方式更快，并且避免了使引用这些文件的索引失效，
    确保后续查询不会返回已删除的行。

## 更新行

Lance 支持使用 SQL 表达式通过 `lance.LanceDataset.update` 方法更新行。
例如，如果我们发现数据集中 Bob 的名字有时被写成了 `Blob`，我们可以这样修复：

```python
import lance

dataset = lance.dataset("./alice_and_bob.lance")
dataset.update({"name": "'Bob'"}, where="name = 'Blob'")
```

更新值是 SQL 表达式，这就是为什么 `'Bob'` 需要用单引号括起来。
这意味着我们可以使用引用现有列的复杂表达式。例如，如果两年过去了，
我们希望同时更新 Alice 和 Bob 的年龄，可以这样写：

```python
import lance

dataset = lance.dataset("./alice_and_bob.lance")
dataset.update({"age": "age + 2"})
```

如果你要用新值更新一组独立的行，通常使用下面描述的合并插入（Merge Insert）操作会更高效。

```python
import lance

# Change the ages of both Alice and Bob
new_table = pa.Table.from_pylist([{"name": "Alice", "age": 30},
                                  {"name": "Bob", "age": 20}])

# This works, but is inefficient, see below for a better approach
dataset = lance.dataset("./alice_and_bob.lance")
for idx in range(new_table.num_rows):
  name = new_table[0][idx].as_py()
  new_age = new_table[1][idx].as_py()
  dataset.update({"age": new_age}, where=f"name='{name}'")
```

## 合并插入（Merge Insert）

Lance 支持合并插入操作。此操作可用于批量添加新数据，同时（可选地）与现有数据进行匹配。
该操作可用于多种不同的使用场景。

### 批量更新

`lance.LanceDataset.update` 方法适用于基于过滤器更新行。
但如果我们想用新行替换现有行，`lance.LanceDataset.merge_insert` 操作会更高效：

```python
import lance

dataset = lance.dataset("./alice_and_bob.lance")
print(dataset.to_table().to_pandas())
#     name  age
# 0  Alice   20
# 1    Bob   30

# Change the ages of both Alice and Bob
new_table = pa.Table.from_pylist([{"name": "Alice", "age": 2},
                                  {"name": "Bob", "age": 3}])
# This will use `name` as the key for matching rows.  Merge insert
# uses a JOIN internally and so you typically want this column to
# be a unique key or id of some kind.
rst = dataset.merge_insert("name") \
       .when_matched_update_all() \
       .execute(new_table)
print(dataset.to_table().to_pandas())
#     name  age
# 0  Alice    2
# 1    Bob    3
```

注意，与更新操作类似，被修改的行将被移除并重新插入到表的末尾，改变它们的位置。
此外，这些行的相对顺序可能会改变，因为内部使用了哈希连接（Hash Join）操作。

### 不存在则插入

有时我们只想在之前没有插入过数据的情况下才插入。例如，当我们有一批数据但不知道哪些行之前已经添加过，
又不想创建重复行时。我们可以使用合并插入操作来实现：

```python
# Bob is already in the table, but Carla is new
new_table = pa.Table.from_pylist([{"name": "Bob", "age": 30},
                                  {"name": "Carla", "age": 37}])

dataset = lance.dataset("./alice_and_bob.lance")

# This will insert Carla but leave Bob unchanged
_ = dataset.merge_insert("name") \
       .when_not_matched_insert_all() \
       .execute(new_table)
# Verify that Carla was added but Bob remains unchanged
print(dataset.to_table().to_pandas())
#     name  age
# 0  Alice   20
# 1    Bob   30
# 2  Carla   37
```

### 更新或插入（Upsert）

有时我们希望组合以上两种行为。如果行已存在则更新它，如果行不存在则添加它。
这种操作有时被称为"upsert"。我们同样可以使用合并插入操作来实现：

```python
import lance
import pyarrow as pa

# Change Carla's age and insert David
new_table = pa.Table.from_pylist([{"name": "Carla", "age": 27},
                                  {"name": "David", "age": 42}])

dataset = lance.dataset("./alice_and_bob.lance")

# This will update Carla and insert David
_ = dataset.merge_insert("name") \
       .when_matched_update_all() \
       .when_not_matched_insert_all() \
       .execute(new_table)
# Verify the results
print(dataset.to_table().to_pandas())
#     name  age
# 0  Alice   20
# 1    Bob   30
# 2  Carla   27
# 3  David   42
```

### 替换部分数据

一种不太常见但仍然有用的行为是用新数据替换某个区域的现有行（由过滤器定义）。
这类似于在单个事务中同时执行删除和插入操作。例如：

```python
import lance
import pyarrow as pa

new_table = pa.Table.from_pylist([{"name": "Edgar", "age": 46},
                                  {"name": "Francene", "age": 44}])

dataset = lance.dataset("./alice_and_bob.lance")
print(dataset.to_table().to_pandas())
#       name  age
# 0    Alice   20
# 1      Bob   30
# 2  Charlie   45
# 3    Donna   50

# This will remove anyone above 40 and insert our new data
_ = dataset.merge_insert("name") \
       .when_not_matched_insert_all() \
       .when_not_matched_by_source_delete("age >= 40") \
       .execute(new_table)
# Verify the results - people over 40 replaced with new data
print(dataset.to_table().to_pandas())
#        name  age
# 0     Alice   20
# 1       Bob   30
# 2     Edgar   46
# 3  Francene   44
```

## 读取 Lance 数据集

要打开 Lance 数据集，使用 `lance.dataset` 函数：

```python
import lance
ds = lance.dataset("s3://bucket/path/imagenet.lance")
# Or local path
ds = lance.dataset("./imagenet.lance")
```

!!! note

    Lance 目前支持本地文件系统、AWS `s3` 和 Google Cloud Storage（`gs`）作为存储后端。
    更多信息请参阅[对象存储配置](object_store.md)。

读取 Lance 数据集最直接的方法是使用 `lance.LanceDataset.to_table` 方法将整个数据集加载到内存中。

```python
table = ds.to_table()
```

由于 Lance 是高性能的列式格式，它可以通过**列（投影）**下推和**过滤（谓词）**下推来高效读取数据集的子集。

```python
table = ds.to_table(
    columns=["image", "label"],
    filter="label = 2 AND text IS NOT NULL",
    limit=1000,
    offset=3000)
```

Lance 理解读取重型列（如 `image`）的成本。
因此，它会采用优化的查询计划来高效执行操作。

### 迭代读取

如果数据集太大无法放入内存，你可以使用 `lance.LanceDataset.to_batches` 方法分批读取：

```python
for batch in ds.to_batches(columns=["image"], filter="label = 10"):
    # do something with batch
    compute_on_batch(batch)
```

`lance.LanceDataset.to_batches` 接受与 `lance.LanceDataset.to_table` 相同的参数。

### 过滤下推 { #filter-pushdown }

Lance 使用标准 SQL 表达式作为数据集过滤的谓词。
通过将 SQL 谓词直接下推到存储系统，扫描期间的整体 I/O 负载显著降低。

目前，Lance 支持不断增加的表达式列表：

* `>`, `>=`, `<`, `<=`, `=`
* `AND`, `OR`, `NOT`
* `IS NULL`, `IS NOT NULL`
* `IS TRUE`, `IS NOT TRUE`, `IS FALSE`, `IS NOT FALSE`
* `IN`
* `LIKE`, `NOT LIKE`
* `regexp_match(column, pattern)`
* `CAST`

例如，以下过滤字符串是可接受的：

```sql
((label IN [10, 20]) AND (note['email'] IS NOT NULL))
    OR NOT note['created']
```

嵌套字段可以使用下标访问。结构体（Struct）字段可以使用字段名下标访问，
列表字段可以使用索引下标访问。

如果你的列名包含特殊字符或是 [SQL 关键字](https://docs.rs/sqlparser/latest/sqlparser/keywords/index.html)，
你可以使用反引号（`` ` ``）进行转义。对于嵌套字段，路径的每个段都必须用反引号包裹。

```sql
`CUBE` = 10 AND `column name with space` IS NOT NULL
  AND `nested with space`.`inner with space` < 2
```

!!! warning

    不支持包含句点（`.`）的字段名。

日期、时间戳和小数的字面量可以在类型名称后面写字符串值来表示。例如：

```sql
date_col = date '2021-01-01'
and timestamp_col = timestamp '2021-01-01 00:00:00'
and decimal_col = decimal(8,3) '1.000'
```

对于时间戳列，可以在类型参数中指定精度。微秒精度（6）是默认值。

| SQL | 时间单位 |
|-----|---------|
| `timestamp(0)` | 秒 |
| `timestamp(3)` | 毫秒 |
| `timestamp(6)` | 微秒 |
| `timestamp(9)` | 纳秒 |

Lance 内部以 Arrow 格式存储数据。SQL 类型到 Arrow 的映射关系如下：

| SQL 类型 | Arrow 类型 |
|----------|------------|
| `boolean` | `Boolean` |
| `tinyint` / `tinyint unsigned` | `Int8` / `UInt8` |
| `smallint` / `smallint unsigned` | `Int16` / `UInt16` |
| `int` or `integer` / `int unsigned` or `integer unsigned` | `Int32` / `UInt32` |
| `bigint` / `bigint unsigned` | `Int64` / `UInt64` |
| `float` | `Float32` |
| `double` | `Float64` |
| `decimal(precision, scale)` | `Decimal128` |
| `date` | `Date32` |
| `timestamp` | `Timestamp` (1) |
| `string` | `Utf8` |
| `binary` | `Binary` |

(1) 精度映射参见上表。

### 随机读取

Lance 作为列式格式的一个独特特性是允许你快速读取随机样本。

```python
# Access the 2nd, 101th and 501th rows
data = ds.take([1, 100, 500], columns=["image", "label"])
```

快速随机访问单行的能力在促进各种工作流中扮演着关键角色，
例如 ML 训练中的随机采样和数据混洗。
此外，它使用户能够构建二级索引，实现快速查询执行以提升性能。

## 表维护

随着时间的推移，一些操作会导致 Lance 数据集的布局变差。
例如，许多小的追加操作会导致大量小的片段（Fragment）。
或者删除许多行会导致查询变慢，因为需要过滤掉已删除的行。

为了解决这个问题，Lance 提供了优化数据集布局的方法。

### 压缩数据文件

可以重写数据文件以减少文件数量。当向 `lance.dataset.DatasetOptimizer.compact_files`
传递 `target_rows_per_fragment` 时，Lance 会跳过已经超过该行数的片段，并重写其他片段。
片段将按照它们的片段 ID 进行合并，因此数据的固有顺序将被保留。

!!! note

    压缩会创建表的新版本。它不会删除旧版本的表及其引用的文件。

```python
import lance

dataset = lance.dataset("./alice_and_bob.lance")
dataset.optimize.compact_files(target_rows_per_fragment=1024 * 1024)
```

在压缩过程中，Lance 还可以移除已删除的行。重写后的片段将不包含删除文件。
这可以提高扫描性能，因为软删除的行不需要在扫描过程中被跳过。

当文件被重写时，原始的行地址将失效。这意味着受影响的文件如果之前是 ANN 索引的一部分，
将不再属于任何 ANN 索引。因此，建议在重建索引之前先重写文件。
