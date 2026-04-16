---
title: 全文搜索
description: 使用倒排 BM25 索引和 N-gram 搜索在 Lance 中进行全文搜索（FTS）
---

# Lance 中的全文搜索

Lance 使用倒排索引（Inverted Index）提供强大的全文搜索（Full-Text Search，FTS）能力。本教程将指导你构建和使用 FTS 索引，以大幅加速文本搜索操作，同时保持高精度。

完成本教程后，你将能够构建和使用 FTS 索引，了解索引搜索与非索引搜索之间的性能差异，以及学习如何调优搜索参数以获得最佳性能。

## 安装 Python SDK

首先，安装所需的依赖：

```bash
pip install pylance pyarrow
```

## 设置你的环境

导入使用 Lance 数据集所需的库：

```python
import lance
import pyarrow as pa
```

## 准备你的文本数据

在本快速入门中，我们将创建一个包含文本文档的简单数据集：

```python
table = pa.table(
    {
        "id": [1, 2, 3],
        "text": [
            "I left my umbrella on the evening train to Boston",
            "This ramen recipe simmers the broth for three hours with dried mushrooms.",
            "This train is scheduled to leave for Edinburgh at 9:30 in the morning",
        ],
    }
)

# Write to a new Lance dataset
lance.write_dataset(table, "/tmp/fts.lance", mode="overwrite")
```

这将创建一个包含三个文本文档的 Lance 数据集，其中包含重叠的关键词，我们将用它们来演示不同的搜索场景。

## 查看你的数据集 Schema

让我们查看数据集的结构：

```python
ds = lance.dataset("/tmp/fts.lance")
print(ds.schema)
```

这将打印数据集的 PyArrow Schema：

```
id: int64
text: large_string
```

## 构建全文搜索索引

全文搜索通过在文本列上创建倒排标量索引（Inverted Scalar Index）来实现。调用 Lance 数据集的 `create_scalar_index` 时选择 `INVERTED` 索引类型。Lance 使用 BM25 排名算法进行相关性评分。结果按相关性自动排序，分数越高表示匹配度越好。

```python
ds.create_scalar_index(
    column="text",
    index_type="INVERTED"
)
```

索引创建过程会构建一个高效的查找结构，将词语映射到包含它们的文档。这使得即使在大型数据集上也能实现高性能的基于关键词的搜索。

!!! warning "索引创建时间"
索引创建时间取决于你的文本数据大小。对于大型数据集，此过程可能需要几分钟，但查询时的性能收益是巨大的。

## 高级索引配置

你可以使用各种参数自定义索引创建，以针对你的特定用例进行优化：

```python
ds.create_scalar_index(
    column="text",
    index_type="INVERTED",
    name="text_idx",              # Optional index name (if omitted, default is "text_idx")
    with_position=False,          # Set True to enable phrase queries (stores token positions)
    base_tokenizer="simple",      # Tokenizer: "simple" (whitespace+punct), "whitespace", or "raw" (no tokenization)
    language="English",           # Language used for stemming + stop words (only used if `stem` or `remove_stop_words` is True)
    max_token_length=40,          # Drop tokens longer than this length
    lower_case=True,              # Lowercase text before tokenization
    stem=True,                    # Stem tokens (language-dependent)
    remove_stop_words=True,       # Remove stop words (language-dependent)
    custom_stop_words=None,       # Optional additional stop words (only used if remove_stop_words=True)
    ascii_folding=True,           # Fold accents to ASCII when possible (e.g., "é" -> "e")
)
```

### 分词器选项

- **simple**：按空格和标点符号分割 token
- **whitespace**：仅按空格分割 token
- **raw**：不进行分词（适用于精确匹配）

Lance 还支持多语言分词：

- **jieba/default**：使用 Jieba 进行中文文本分词
- **lindera/ipadic**：使用 Lindera 配合 IPAdic 词典进行日文文本分词
- **lindera/ko-dic**：使用 Lindera 配合 Ko-dic 词典进行韩文文本分词
- **lindera/unidic**：使用 Lindera 配合 UniDic 词典进行日文文本分词

### 语言处理功能

- **词干提取（Stemming）**：将词语还原为词根形式（例如 "running" → "run"）
- **停用词（Stop Words）**：移除常见词汇，如 "the"、"and"、"is"
- **ASCII 折叠（ASCII Folding）**：将带重音的字符转换为 ASCII（例如 "é" → "e"）

## 使用 FTS 查询进行搜索

现在你可以使用倒排索引运行 FTS 查询：

```python
import lance

# Open dataset
ds = lance.dataset("/tmp/fts.lance")

# Specify keyword phrases when calling the `to_table` method
query_result = ds.to_table(
    full_text_query="umbrella train"
)
print(query_result)
```

此查询返回包含 "umbrella" 或 "train"（或两者）的文档。搜索不区分大小写，并使用倒排索引进行快速检索。

```
id: [[1, 3]]
text: [["I left my umbrella on the evening train to Boston", "This train is scheduled to leave for Edinburgh at 9:30 in the morning"]]
_score: [[..., ...]]
```

## 将全文搜索与元数据结合

将 FTS 与元数据过滤结合在单次查询中，可以找到更相关的结果。
你可以通过向 `filter` 参数传递过滤表达式来实现这一点。

```python
import lance
import pyarrow as pa

table = pa.table(
    {
        "id": [1, 2, 3],
        "text": [
            "I left my umbrella on the morning train to Boston",
            "This ramen recipe simmers the broth for three hours with dried mushrooms.",
            "This train is scheduled to leave for Edinburgh at 9:30 AM",
        ],
        "category": ["travel", "food", "travel"],
    }
)

# Temp write dataset
lance.write_dataset(table, "./fts_test_with_metadata.lance", mode="overwrite")

ds = lance.dataset("./fts_test_with_metadata.lance")

# Create FTS index
ds.create_scalar_index(
    column="text",
    index_type="INVERTED",
)

# Run FTS query with metadata filter
query_result = ds.to_table(
    full_text_query="three",
    filter='category = "food"',
)

# Returns
# id: [[2]]
# text: [["This ramen recipe simmers the broth for three hours with dried mushrooms."]]
# category: [["food"]]
```

## 高级搜索功能

### 布尔搜索运算符

你可以通过构造结构化查询对象来使用布尔搜索运算符。

#### 所有词项：`AND`

```python
from lance.query import FullTextOperator, MatchQuery

# Require the terms 'umbrella AND train AND boston' to be present
and_query = MatchQuery("umbrella train boston", "text", operator=FullTextOperator.AND)
query_result = ds.to_table(full_text_query=and_query)

# Returns
# text: [["I left my umbrella on the evening train to Boston"]]
```

#### 任意词项：`OR`

```python
from lance.query import FullTextOperator, MatchQuery

# Require the terms 'morning OR evening' to be present
or_query = MatchQuery("morning evening", "text", operator=FullTextOperator.OR)
query_result = ds.to_table(full_text_query=or_query)

# Returns the Boston document that mentions 'evening', and the Edinburgh document that mentions 'morning'
# text: [["This train is scheduled to leave for Edinburgh at 9:30 in the morning", "I left my umbrella on the evening train to Boston"]]
```

#### 通过运算符混合 `AND`/`OR` 查询

你可以在 Python 中使用运算符混合 `AND`/`OR` 查询：

```python
from lance.query import FullTextOperator, MatchQuery

# Combine AND and OR semantics
# Require 'train' AND ('morning' OR 'evening')
q1 = MatchQuery("morning evening", "text", operator=FullTextOperator.OR)
q2 = MatchQuery("train", "text")
query_result = ds.to_table(full_text_query=(q1 & q2))

# Returns both the Boston and Edinburgh documents that mention 'train'
# text: [["I left my umbrella on the evening train to Boston", "This train is scheduled to leave for Edinburgh at 9:30 in the morning"]]
```

要通过运算符组合 `OR` 查询，使用模式 `q1 | q2`。

#### 排除词项：`NOT`

排除特定关键词的查询需要使用 `BooleanQuery`/`Occur` 来显式编写，如下所示。

```python
from lance.query import MatchQuery, BooleanQuery, Occur

# Require that 'umbrella' be present, but 'train' NOT be present
q = BooleanQuery(
    [
        (Occur.MUST, MatchQuery("umbrella", "text")),
        (Occur.MUST_NOT, MatchQuery("train", "text")),
    ]
)
query_result = ds.to_table(full_text_query=q)

# Returns empty result, as no document matches this condition
# text: []
```

### 短语搜索

要进行精确短语匹配，请确保在创建索引时启用 `with_position=True`，该选项默认是关闭的。

```python
# Rebuild the index with positions enabled (required for phrase queries)
ds.create_scalar_index(
    "text",
    "INVERTED",
    with_position=True,
    remove_stop_words=False,
)
# Search for the exact phrase "train to boston"
table = ds.to_table(full_text_query="'train to boston'")

# If stopwords are removed, this phrase query would return an empty result
# text: [["I left my umbrella on the evening train to Boston"]]
```

!!! warning "停用词默认被移除"
"to"、"the" 等常见词汇被归类为停用词，在创建索引时默认会被移除。如果你想搜索包含停用词的精确短语，请在创建索引时设置 `remove_stop_words=False`。

### 使用 N-gram 索引进行子串匹配

`NGRAM` 是一种用于文本**子串/模式搜索**的标量索引类型。它是通配符查询（如 `term*` / `*term`，在 Lance 的 `full_text_query` 中不被解析）的良好替代方案。

N-gram 索引为字符串中的每个 N-gram 创建一个位图。默认情况下，Lance 使用三元组（trigram）。此索引可用于加速过滤器中使用 `contains` 函数的查询。

```python
import lance

ds = lance.dataset("/tmp/fts.lance")

# Build an NGRAM index for substring search (speeds up `contains(...)` filters)
# Give the index a distinct name so it won't replace your FTS index
ds.create_scalar_index(column="text", index_type="NGRAM", name="text_ngram")

# Substring search
q1 = ds.to_table(filter="contains(text, 'ramen')")

# Returns the document about ramen
# text: [["This ramen recipe simmers the broth for three hours with dried mushrooms."]]
```

你可以解释查询计划来确认 N-gram 索引的使用情况，如下所示：

```python
# Inspect the query plan to confirm index usage
print(ds.scanner(filter="contains(text, 'train')").explain_plan())
```

### 模糊搜索

对于 `INVERTED` 索引上的 FTS `MatchQuery`，支持模糊搜索（Fuzzy Search）。它使用 Levenshtein 编辑距离来匹配有拼写错误或轻微变化的词项。

```python
from lance.query import MatchQuery

# Explicit edit distance (1)
query_result = ds.to_table(
    full_text_query=MatchQuery(
        "rammen",  # Misspelled 'ramen'
        "text",
        fuzziness=1,
        max_expansions=50,  # default: 50
    )
)
```

你也可以设置 `fuzziness=None` 使用自动模糊度：

- 词项长度 `<= 2` 时为 `0`
- 词项长度 `<= 5` 时为 `1`
- 词项长度 `> 5` 时为 `2`

```python
query_result = ds.to_table(
    full_text_query=MatchQuery(
        "rammen",
        "text",
        fuzziness=None,
    )
)
```

要在模糊匹配期间强制精确前缀，设置 `prefix_length`。
这意味着前 `N` 个字符必须精确匹配，然后才允许对词项的其余部分进行模糊编辑。
例如，`prefix_length=2` 时，`"rammen"` 可以匹配以 `"ra"` 开头的词项（如 `"ramen"`），但不能匹配以其他前缀开头的词项。

```python
query_result = ds.to_table(
    full_text_query=MatchQuery(
        "rammen",
        "text",
        fuzziness=1,
        prefix_length=2,  # "ra" must match exactly
    )
)
```

## 性能提示

### 索引维护

当你在创建 `INVERTED` 索引后追加新行时，Lance 仍然会在 `full_text_query` 结果中返回这些行。它使用 FTS 索引搜索已索引的片段，对未索引的片段进行平坦搜索，然后合并结果。

为了在新数据到达时保持较低的 FTS 延迟，定期通过调用 `ds.optimize.optimize_indices()` 将未索引的片段添加到现有的 FTS 索引中：

```python
# Append new data
new_rows = pa.table(
    {
        "id": [4],
        "text": ["The next train leaves at noon"],
    }
)
ds.insert(new_rows)

# Incrementally update existing indices (including "text_idx")
ds.optimize.optimize_indices(index_names=["text_idx"])

# Optional: monitor index coverage
stats = ds.stats.index_stats("text_idx")
print(stats["num_unindexed_rows"], stats["num_indexed_rows"])
```

!!! info
如果你使用了自定义索引名称，请将 `"text_idx"` 替换为你的索引名称。
如果你在对 `"text"` 列创建 FTS 索引时未设置 `name=...`，默认索引名称为 `"text_idx"`。

如果你更改了分词器设置（如 `with_position`、`base_tokenizer`、停用词或词干提取），请使用 `create_scalar_index(..., replace=True)` 重建索引，以便使用新配置对完整数据集进行索引。

### 索引配置最佳实践

- 当你需要短语查询时启用 `with_position`，因为它会存储文档中的词语位置。对于简单的词项搜索，禁用此选项可以在不影响性能的情况下节省大量存储空间。

- 在大多数应用中保持 `lower_case=True` 以确保不区分大小写的搜索行为。这提供了更好的用户体验并符合常见的搜索预期，不过如果大小写敏感性对你的用例很重要，你可以禁用它。

- 当你希望通过匹配词语变体（例如 "running" 匹配 "run"）来获得更好的召回率时，启用词干提取（`stem=True`）。如果你需要精确的词项匹配或你的领域需要精确的术语，则禁用词干提取。

- 考虑启用 `remove_stop_words=True` 以获得更清晰的搜索结果，特别是在内容密集型应用中。这会从索引中移除 "the"、"and"、"is" 等常见词汇，减少噪音并提高相关性。如果停用词在你的领域中承载重要含义，请保留它们。

### 查询优化

使用特定、有针对性的搜索词项通常比宽泛、通用的查询能产生更好的性能。更具体的词项减少了潜在匹配的数量，使索引能够更高效地工作。建议分析你最常见的搜索模式，并相应地优化索引配置。

将全文搜索与元数据过滤结合可以显著缩小搜索范围并提高性能。使用结构化数据过滤器在应用文本搜索之前缩小结果范围，反之亦然。这种方法对于大型数据集特别有效，因为你可以在查询过程的早期消除许多不相关的文档。

### 延伸阅读

关于使用不同分词器的高级用法说明以及索引训练过程的更多技术细节（包括预期的内存和磁盘使用信息），请访问[全文索引](../format/table/index/scalar/fts.md)规范。

## 下一步

查看 **[用户指南](../guide/read_and_write.md)** 并更深入地探索 Lance API。
