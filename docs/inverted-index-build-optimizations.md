# Inverted 索引建索引流程：可显著优化点

> 目标：找出**显著提升性能**或**显著降低内存**的优化方向（不改代码，仅列建议与原因）。

## 现有流程简述（定位用）
- 入口：`InvertedIndexBuilder::update` / `update_index`（`rust/lance-index/src/scalar/inverted/builder.rs`）
- 并行：按 `LANCE_FTS_NUM_SHARDS` 创建多个 `IndexWorker`，每个 worker 消费 RecordBatch
- 索引构建：`IndexWorker::process_batch` -> Tokenize -> 组装 posting lists / docs
- 写盘：`InnerBuilder::write_*`（posting list / tokens / docs）
- 合并：`SizeBasedMerger::merge`（`rust/lance-index/src/scalar/inverted/merger.rs`）

---

## 高优先级优化点（显著收益潜力）

### 1) JSON 索引的双重转换（CPU + 内存热点）
- 位置：
  - `document_input` -> `JsonTextStream`（`rust/lance-index/src/scalar/inverted/builder.rs` / `json.rs`）
  - `JsonTokenizer::token_stream_for_doc`（`rust/lance-index/src/scalar/inverted/tokenizer/lance_tokenizer.rs`）
- 现状问题：JSONB -> JSON 字符串 -> `serde_json::Value` 解析，**双重转换 + 全量构建 AST**。
- 优化建议：
  - 增加 JSONB 直通 tokenizer（直接从 `jsonb::RawJsonb` 或字节流产出 token），绕过字符串中间层。
  - 或使用 `serde_json::Deserializer`/`simd-json` 的 streaming 访问器直接发 token，避免构建完整 `Value`。
- 预期收益：对 JSON 列的索引速度与峰值内存**大幅下降**。

### 2) 每文档 `HashMap` + `Vec` 的高频分配
- 位置：`IndexWorker::process_batch`（`builder.rs`）
- 现状问题：每条 doc 创建 `HashMap<u32, PositionRecorder>`；每个 token 还可能分配 `Vec<u32>` 保存 positions。
- 优化建议：
  - 复用 per-worker 的 `HashMap`/临时缓冲：`clear()` 保留容量；根据 token 数量 `reserve()`。
  - `with_position=false` 时：收集 `Vec<u32>` token_id -> `sort_unstable()` -> run-length，**避免 HashMap**。
  - `with_position=true` 时：用 `SmallVec<[u32; N]>` 或 arena/pool 降低 per-token Vec 分配。
- 预期收益：减少大量临时分配，提升吞吐，降低 RSS 峰值。

### 3) Merge 阶段 token 映射的额外拷贝
- 位置：`SizeBasedMerger::merge`（`merger.rs`）
- 现状问题：`inv_token: HashMap<u32, &String>` + `token.clone()`；token 重复时仍频繁 clone。
- 优化建议：
  - 用 `Vec<&str>` 按 token_id 索引替代 `HashMap`（token_id 连续）。
  - 给 `TokenSet` 增加 `get_or_add(&str)`，避免 token 已存在时的 clone。
- 预期收益：**显著减少 merge 内存与 CPU**，尤其是高重叠词表场景。

### 4) Posting list 写盘前的 `RecordBatch` 拼接成本
- 位置：`InnerBuilder::write_posting_lists`（`builder.rs`）
- 现状问题：每个 posting list 先 `to_batch`，再 `concat_batches` 聚合；中间对象多、内存占用高。
- 优化建议：
  - 直接用 `ListBuilder/Float32Builder/UInt32Builder` 构建列，按行追加 posting list，按阈值 flush。
  - 或降低 buffer 规模，改为“更多小批量写入”以换取低峰值内存。
- 预期收益：降低峰值内存，减少 `concat` 复制成本。

### 5) 多 worker 内存线性放大（缺少全局预算）
- 位置：`LANCE_FTS_NUM_SHARDS` + `LANCE_FTS_PARTITION_SIZE`（`builder.rs`）
- 现状问题：每个 worker 最高可吃掉一个分区规模（默认 256MiB），总内存随 worker 线性增长。
- 优化建议：
  - 引入全局 `AtomicU64` 统计 estimated_size；超过预算时让 worker 触发 flush 或暂停。
  - 自动根据内存预算和分区大小调整 shard 数量。
- 预期收益：对大数据集构建时**显著降低峰值内存**，避免 OOM。

### 6) Merge 阶段的“全量解压再重压”
- 位置：`SizeBasedMerger::merge`（`merger.rs`）
- 现状问题：遍历 posting list 迭代器会解压后再写，CPU 与内存成本高。
- 优化建议：
  - 做“流式 k-way merge”：按 token 流式读取并写出，不保留全量 posting list。
  - 分批 merge（多轮 merge，小批次输入），降低峰值占用。
  - 评估“块级重写”：仅调整 doc_id 偏移与 block header，避免完全解压。
- 预期收益：大规模索引合并时 CPU/内存消耗**明显下降**。

---

## 中优先级优化点（需评估收益）

### 7) 大批次下的并行粒度不足
- 位置：`update_index` 把整批 RecordBatch 发给单个 worker
- 现状问题：若上游 batch 大而数量少，多核利用率下降。
- 优化建议：按行切分 batch（分片）后入队，或将 batch 切成更小块。
- 预期收益：提升 CPU 利用率（注意可能增加 partition 数量）。

### 8) `FlattenStream` 扩展 list 时复制 row_id
- 位置：`FlattenStream::poll_next` / `flatten_string_list`（`builder.rs`）
- 现状问题：为 list 展平复制 row_id 数组，长列表场景内存增加明显。
- 优化建议：避免构造新 RecordBatch，改为“行迭代器”直接喂给 `IndexWorker`；或用字典编码 row_id。
- 预期收益：降低峰值内存，特别是 list-of-strings 字段。

### 9) Posting list 内部结构的缓存局部性
- 位置：`PostingListBuilder` 使用 `ExpLinkedList`（`index.rs`, `lance-core/src/container/list.rs`）
- 现状问题：`LinkedList<Vec<T>>` 带来指针跳转和额外节点开销。
- 优化建议：评估 `Vec` 或 `ChunkedVec`（连续块）替代，提升缓存局部性。
- 预期收益：CPU 性能提升 + 小幅内存优化（需基准验证）。

---

## 建议的验证方式（避免盲改）
- 针对 `with_position=false` / JSON 文档 / 超大词表分别做 micro-bench。
- 记录：
  - tokenization 时间
  - 进程 RSS 峰值
  - 每百万 doc 的索引吞吐
- 建议在 `builder.rs` / `merger.rs` 增加临时 metrics 打点（统计 doc token 数分布、per-doc map 容量、flush 次数）。
