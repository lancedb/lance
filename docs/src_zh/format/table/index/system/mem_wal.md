# MemWAL 索引

MemWAL 索引是一个系统索引，作为所有 MemWAL 元数据的集中式结构。
它存储配置（分片规范、要维护的索引）、合并进度和分片状态快照。

一个表最多有一个 MemWAL 索引。

完整规范请参阅：

- [MemWAL 索引概述](../../mem_wal.md#memwal-index) - 目的和高层描述
- [MemWAL 索引详情](../../mem_wal.md#memwal-index-details) - 存储格式、Schema 和过期处理
- [MemWAL 实现](../../mem_wal.md#implementation-expectation) - 实现细节和期望
