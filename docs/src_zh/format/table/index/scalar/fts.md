# 全文搜索索引（Full Text Search Index）

全文搜索（FTS）索引（又称倒排索引，Inverted Index）通过将词元映射到包含它们的文档来提供高效的文本搜索。
它专为高性能文本搜索而设计，支持多种评分算法和短语查询。

## 索引详情

```protobuf
%%% proto.message.InvertedIndexDetails %%%
```

## 存储布局

FTS 索引由多个文件组成，存储词元字典、文档信息和倒排列表：

1. `tokens.lance` - 词元字典，将词元映射到词元 ID
2. `docs.lance` - 文档元数据，包括词元计数
3. `invert.lance` - 每个词元的压缩倒排列表
4. `metadata.lance` - 索引元数据和配置

一个 FTS 索引可以包含多个分区（Partition）。每个分区有自己的词元、文档和倒排列表文件集，以分区 ID 为前缀（例如 `part_0_tokens.lance`、`part_0_docs.lance`、`part_0_invert.lance`）。`metadata.lance` 文件列出索引中的所有分区 ID。查询时，必须搜索每个分区并合并结果以产生最终的排序输出。分区数量越少通常意味着查询性能越好，因为每个分区需要独立的词元字典查找和倒排列表扫描。分区数量由训练配置控制——具体来说，`LANCE_FTS_TARGET_SIZE` 决定每个合并分区可以增长到多大（详见[训练过程](#training-process)）。

### 词元字典文件 Schema

| 列          | 类型   | 可空  | 描述               |
|-------------|--------|-------|--------------------|
| `_token`    | Utf8   | false | 词元字符串          |
| `_token_id` | UInt32 | false | 词元的唯一标识符     |

### 文档文件 Schema

| 列            | 类型   | 可空  | 描述               |
|---------------|--------|-------|--------------------|
| `_rowid`      | UInt64 | false | 文档行 ID           |
| `_num_tokens` | UInt32 | false | 文档中的词元数量     |

### 倒排列表文件 Schema

| 列                     | 类型                    | 可空  | 描述                                                    |
|------------------------|-------------------------|-------|---------------------------------------------------------|
| `_posting`             | List<LargeBinary>       | false | 压缩倒排列表（差分编码的行 ID 和频率）                     |
| `_max_score`           | Float32                 | false | 词元的最大分数（用于查询优化）                              |
| `_length`              | UInt32                  | false | 包含该词元的文档数量                                       |
| `_compressed_position` | List<List<LargeBinary>> | true  | 可选的压缩位置列表，用于短语查询                            |

### 元数据文件 Schema

元数据文件包含 JSON 序列化的配置和分区信息：

| 键           | 类型          | 描述                                          |
|--------------|---------------|-----------------------------------------------|
| `partitions` | Array<UInt64> | 分区 ID 列表，用于分布式索引组织                 |
| `params`     | JSON Object   | 序列化的 InvertedIndexParams，包含分词器配置     |

#### InvertedIndexParams 结构

| 字段                | 类型    | 默认值    | 描述                                                      |
|---------------------|---------|-----------|-----------------------------------------------------------|
| `base_tokenizer`    | String  | "simple"  | 基础分词器类型（参见分词器章节）                             |
| `language`          | String  | "English" | 用于词干提取和停用词的语言                                   |
| `with_position`     | Boolean | false     | 存储词元位置以支持短语查询（会增加索引大小）                  |
| `max_token_length`  | UInt32? | None      | 最大词元长度（超过此长度的词元将被移除）                      |
| `lower_case`        | Boolean | true      | 将词元转换为小写                                            |
| `stem`              | Boolean | false     | 应用特定语言的词干提取（Stemming）                           |
| `remove_stop_words` | Boolean | false     | 移除指定语言的常见停用词（Stop Words）                       |
| `ascii_folding`     | Boolean | true      | 将重音字符转换为 ASCII 等价字符                              |
| `min_gram`          | UInt32  | 2         | 最小 n-gram 长度（仅用于 ngram 分词器）                     |
| `max_gram`          | UInt32  | 15        | 最大 n-gram 长度（仅用于 ngram 分词器）                     |
| `prefix_only`       | Boolean | false     | 仅生成前缀 n-gram（仅用于 ngram 分词器）                    |

## 分词器

全文搜索索引支持多种分词器类型，适用于不同的文本处理需求：

### 基础分词器

| 分词器         | 描述                                              | 使用场景              |
|----------------|---------------------------------------------------|-----------------------|
| **simple**     | 按空格和标点分割，移除非字母数字字符                  | 通用文本（默认）       |
| **whitespace** | 仅按空格字符分割                                    | 保留标点              |
| **raw**        | 不分词，将整个文本视为单个词元                        | 精确匹配              |
| **ngram**      | 将文本分解为重叠的字符序列                           | 子串/模糊搜索          |
| **jieba/***    | 中文分词器，支持分词                                 | 中文文本              |
| **lindera/***  | 日文分词器，支持形态学分析                            | 日文文本              |

#### Jieba 分词器（中文）

Jieba 是一个流行的中文分词库，使用基于词典的方法结合统计方法进行分词。

- **配置**：使用模型目录中的 `config.json` 文件
- **模型**：必须下载并放置在 Lance 主目录的 `jieba/` 下
- **用法**：指定为 `jieba/<model_name>` 或使用默认模型时直接指定 `jieba`
- **配置结构**：
  ```json
  {
    "main": "path/to/main/dictionary",
    "users": ["path/to/user/dict1", "path/to/user/dict2"]
  }
  ```
- **功能**：
  - 对简体和繁体中文进行精确分词
  - 支持自定义用户词典
  - 多种分词模式（精确模式、全模式、搜索引擎模式）

#### Lindera 分词器（日文）

Lindera 是一个专为日文文本设计的形态学分析分词器。它为不使用空格分隔单词的日文提供正确的分词功能。

- **配置**：使用模型目录中的 `config.yml` 文件
- **模型**：必须下载并放置在 Lance 主目录的 `lindera/` 下
- **用法**：指定为 `lindera/<model_name>`，其中 `<model_name>` 是包含模型文件的子目录
- **功能**：
  - 带有词性标注的形态学分析
  - 基于词典的分词
  - 支持自定义用户词典

### 词元过滤器（Token Filters）

词元过滤器在基础分词器之后按顺序应用：

| 过滤器           | 描述                                  | 配置                            |
|------------------|---------------------------------------|---------------------------------|
| **RemoveLong**   | 移除超过 max_token_length 的词元       | `max_token_length`              |
| **LowerCase**    | 将词元转换为小写                       | `lower_case`（默认值：true）     |
| **Stemmer**      | 将单词还原为词根形式                    | `stem`、`language`              |
| **StopWords**    | 移除常见词如 "the"、"is"、"at"         | `remove_stop_words`、`language` |
| **AsciiFolding** | 将重音字符转换为 ASCII                 | `ascii_folding`（默认值：true）  |

### 支持的语言

词干提取和停用词移除支持以下语言：
Arabic、Danish、Dutch、English、Finnish、French、German、Greek、Hungarian、Italian、Norwegian、Portuguese、Romanian、Russian、Spanish、Swedish、Tamil、Turkish

## 文档类型

Lance 支持两种文档类型：文本（Text）和 JSON。不同的文档类型有不同的分词规则和词元解析格式。

### 文本类型

文本类型包括文本和文本列表。词元由 base_tokenizer 生成。

下面的示例展示了文本文档如何被解析为词元。
```text
Tom lives in San Francisco.
```

生成的词元如下。
```text
Tom
lives
in
San
Francisco
```

### JSON 类型

JSON 是嵌套结构，Lance 将 JSON 文档分解为三元组（Triplet）格式的词元 `path,type,value`。有效的类型为：str、number、bool、null。

当三元组的值为 str 时，文本值将使用 base_tokenizer 进一步分词，从而产生多个三元组词元。

查询时，JSON 分词器使用三元组格式而非 JSON 格式，这简化了查询语法。

下面的示例展示了 JSON 文档如何被分词。假设有以下 JSON 文档：
```json
{
  "name": "Lance",
  "legal.age": 30,
  "address": {
    "city": "San Francisco",
    "zip:us": 94102
  }
}
```

解析后，文档将被分词为以下词元：
```
name,str,Lance
legal.age,number,30
address.city,str,San
address.city,str,Francisco
address.zip:us,number,94102
```

然后以三元组格式进行全文搜索。要搜索 "San Francisco"，可以使用以下任一三元组：
```
address.city:San Francisco
address.city:San
address.city:Francisco
```

## 训练过程 { #training-process }

构建 FTS 索引是一个多阶段流水线：扫描源列、并行分词文档、中间结果溢写到磁盘上的 part 文件，然后将 part 文件合并为最终的输出分区。

### 阶段 1：分词

输入列作为 record batch 流读取，并分派到分词器工作线程池。每个工作线程独立地对文档进行分词，在内存中累积词元、倒排列表和文档元数据。

当工作线程的累积数据达到分区大小限制或文档计数达到 `u32::MAX` 时，它会将数据刷写到磁盘作为一组 part 文件（`part_<id>_tokens.lance`、`part_<id>_invert.lance`、`part_<id>_docs.lance`）。如果单个工作线程处理了足够多的数据，它可能会生成多个 part 文件。

### 阶段 2：合并

所有工作线程完成后，part 文件被合并为输出分区。Part 文件以有界缓冲的流式方式读取，因此不需要将所有数据同时加载到内存中。对于每个 part 文件，词元字典被统一，文档集被拼接，倒排列表使用调整后的 ID 重新写入。

当合并后的分区达到目标大小时，它会被写入目标存储并开始新的分区。所有 part 文件消耗完毕后，最终分区被刷写，并写入一个 `metadata.lance` 文件列出分区 ID 和索引参数。

### 配置

| 环境变量                     | 默认值                     | 描述                                                                                            |
|------------------------------|----------------------------|-------------------------------------------------------------------------------------------------|
| `LANCE_FTS_NUM_SHARDS`       | 计算密集型 CPU 数量         | 并行分词器工作线程数。更高的值增加索引吞吐量但使用更多内存。                                       |
| `LANCE_FTS_PARTITION_SIZE`   | 256（MiB）                 | 工作线程内存缓冲区溢写到 part 文件前的最大未压缩大小。                                             |
| `LANCE_FTS_TARGET_SIZE`      | 4096（MiB）                | 合并输出分区的目标未压缩大小。更少、更大的分区可提升查询性能。                                      |

### 内存和性能注意事项

内存使用主要由两个因素决定：

- **`LANCE_FTS_NUM_SHARDS`** -- 每个工作线程持有一个独立的内存缓冲区。峰值内存大约为 `NUM_SHARDS * PARTITION_SIZE` 加上词元字典和倒排列表结构的开销。
- **`LANCE_FTS_PARTITION_SIZE`** -- 更大的值减少 part 文件数量，使合并阶段更高效。更小的值以更多 part 文件为代价减少每个工作线程的内存占用。

合并阶段的内存受流式方法限制：part 文件逐个加载，具有小的并发缓冲区。合并后分区的内存大小受 `LANCE_FTS_TARGET_SIZE` 限制。

构建 FTS 索引需要临时磁盘空间来存储分词阶段生成的 part 文件。临时空间的大小很大程度上取决于是否启用了位置信息。启用 `with_position: true` 的索引会存储每个文档中每个词元出现的位置，这很容易需要原始列大小 10 倍或更多的临时磁盘空间。不带位置信息的索引通常小于原始列，总磁盘空间需求通常不到列大小的 2 倍。

性能建议：

- 更大的 `LANCE_FTS_TARGET_SIZE` 产生更少的输出分区，这有利于查询性能，因为查询必须扫描每个分区的词元字典。在内存允许的情况下，优先选择更少、更大的分区。
- `with_position: true` 会显著增加索引大小，因为每次出现都会存储词元位置。仅在需要短语查询时启用。
- ngram 分词器比词级分词器为每个文档生成更多的词元，因此预期索引大小和内存使用量更大。

### 分布式训练

FTS 索引支持分布式训练，不同的工作节点各自索引数据的一个子集，然后汇总结果。

1. 每个分布式工作节点被分配一个**片段掩码**（`(fragment_id as u64) << 32`），该掩码与其生成的分区 ID 进行 OR 运算，确保跨工作节点的全局唯一 ID。
2. 工作节点设置 `skip_merge: true`，直接写入 part 文件而不运行合并阶段。
3. 每个工作节点写入以分区为单位的元数据文件 `part_<id>_metadata.lance`，而不是单一的 `metadata.lance`。
4. 所有工作节点完成后，协调器合并元数据文件：收集所有分区 ID，将其重新映射为从 0 开始的连续范围（重命名相应的数据文件），并写入最终的统一 `metadata.lance`。

这使得每个工作节点在分词阶段可以独立运行。只有最终的元数据合并需要单节点步骤，而且它是轻量级的，因为只需重命名文件和写入一个小的元数据文件。

## 加速查询

Lance SDK 提供专用的全文搜索 API 来利用 FTS 索引的能力。
这些 API 支持超越简单词元匹配的复杂查询类型，
实现了精细的文本搜索操作。
以下是 FTS 索引支持的查询类型：

| 查询类型            | 描述                                                                     | 示例用法                                                 | 结果类型 |
|---------------------|--------------------------------------------------------------------------|----------------------------------------------------------|---------|
| **contains_tokens** | 基本的基于词元的搜索（UDF），使用 BM25 评分并自动排序结果                    | SQL: `contains_tokens(column, 'search terms')`           | AtMost  |
| **match**           | 匹配查询，支持可配置的 AND/OR 运算符和相关性评分                            | `{"match": {"query": "text", "operator": "and/or"}}`     | AtMost  |
| **phrase**          | 精确短语匹配，使用位置信息（需要 `with_position: true`）                    | `{"phrase": {"query": "exact phrase"}}`                  | AtMost  |
| **boolean**         | 复杂的布尔查询，支持 must/should/must_not 子句以实现精细搜索逻辑            | `{"boolean": {"must": [...], "should": [...]}}`          | AtMost  |
| **multi_match**     | 同时跨多个字段搜索，使用统一评分                                           | `{"multi_match": [{"field1": "query"}, ...]}`            | AtMost  |
| **boost**           | 通过可配置的因子提升特定词元或查询的相关性分数                               | `{"boost": {"query": {...}, "factor": 2.0}}`             | AtMost  |
