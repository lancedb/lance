# Lance 社区治理

Lance 社区由志愿者以协作和开放的方式运营。
其治理灵感来源于 ASF、CNCF 和 Substrait 等开源基金会和项目。

## 治理结构

Lance 社区设定了三个层级的参与方式：

### 贡献者（Contributors）

每一位为 Lance 做出贡献的人都是贡献者。

"贡献"不仅限于代码变更。
在个人或公司项目中采用 Lance、提交 Bug 报告和功能请求、进行代码审查、
组织或策划社区活动、发表演讲、创建和协助品牌设计、
编写文档，以及许多其他活动都被视为贡献。

所有贡献，无论形式如何，都受到重视和极大的赞赏。
完全可以在不编写代码的情况下在治理层级中晋升。

### 维护者（Maintainers）

维护者是对 Lance 社区做出持续且有价值贡献的贡献者。
维护者因其工作而受到认可，并被授予各种权利以支持其持续贡献。
有关活动、权利、名册以及如何成为维护者的更多详情，请参阅 [维护者](./maintainers.md)。

### 项目管理委员会（PMC）

PMC 成员是在项目中展现出领导力的维护者。
PMC 指导项目的长期方向，对治理和项目变更做出决策，并保护 Lance 品牌。
有关活动、权利、名册以及如何成为 PMC 成员的更多详情，请参阅 [PMC](./pmc.md)。

### 名册信息

维护者和 PMC 名册信息遵循以下准则：

- **排序**：名册中的人员按姓氏字母顺序排列
- **自行上报**：个人信息如 _所属机构_ 和 _生态系统角色_ 由本人自行上报并由个人自行决定何时更新。
- **生态系统角色**：此字段记录个人在其他开源项目中的参与情况（如有）。它有助于识别与更广泛的开源生态系统的潜在协作和集成路径。

## 项目

本节详述 Lance 社区维护的项目。

### 核心项目

[lance](https://github.com/lance-format/lance) 是 lance-format GitHub 组织的核心项目，
承载了表格式和文件格式、Rust SDK、Python 和 Java 绑定 SDK、文档和讨论的大部分开发工作。

核心项目由 Lance 社区以严格的质量和发布标准进行维护。
[贡献指南](./contributing.md)、[社区投票流程](./voting.md) 和 [发布指南](./release.md)
均适用于核心项目。

### 子项目（Subprojects）

子项目是扩展 Lance 功能的计划或仓库。
它们必须与 Lance 的整体使命和技术方向保持一致。
新的子项目通过 PMC 投票从孵化子项目毕业后创建。

与核心项目相比，子项目的要求较为宽松：

- 贡献者即使不是维护者也可能获得写入权限
- 在维护者的自主判断下，可以允许不经审查直接合并
- 发布流程可能比核心项目更加简化

以下是当前子项目列表：

| 项目名称          | 仓库                                              | 内容                                                                   |
|-------------------|---------------------------------------------------|------------------------------------------------------------------------|
| lance-duckdb      | https://github.com/lance-format/lance-duckdb      | Lance 的 DuckDB 扩展                                                   |
| lance-huggingface | https://github.com/lance-format/lance-huggingface | Lance 的 Hugging Face 集成                                             |
| lance-namespace   | https://github.com/lance-format/lance-namespace   | Lance 命名空间格式规范、Rust/Python/Java 代码生成 SDK                   |
| lance-namespace-impls | https://github.com/lance-format/lance-namespace-impls | Lance 命名空间实现 - Apache Hive、Apache Polaris、Apache Gravitino、Unity Catalog、AWS Glue 等 |
| lance-python-docs | https://github.com/lance-format/lance-python-docs | Lance Python SDK 自动生成的文档及与 readthedocs 的集成钩子              |
| lance-ray         | https://github.com/lance-format/lance-ray         | Lance 的 Ray 集成                                                      |
| lance-spark       | https://github.com/lance-format/lance-spark       | Lance 的 Apache Spark 连接器                                           |

### 孵化子项目（Incubating Subprojects）

孵化子项目是 Lance 生态系统中的实验性或早期阶段仓库。
任何 PMC 成员都可以无需正式投票即创建孵化子项目。
这些项目为新想法在承诺达到完整子项目标准之前提供了开发空间。

孵化子项目的要求最为宽松：

- 项目创建者或现有 PMC 成员可以将任何人添加为提交者
- 允许不经审查直接合并
- 不需要正式的发布流程

**重要**：所有孵化子项目必须在其 README 中包含以下醒目的声明：

> ⚠️ **孵化子项目**：本项目正在孵化中，尚未成为正式的 Lance 子项目。
> API 和功能可能会在没有通知的情况下发生变更。在生产环境中使用需自行承担风险。

以下是当前孵化子项目列表：

| 项目名称          | 仓库                                              | 内容                                               |
|-------------------|---------------------------------------------------|----------------------------------------------------|
| lance-context     | https://github.com/lance-format/lance-context     | 使用 Lance 管理多模态智能体上下文生命周期           |
| lance-data-viewer | https://github.com/lance-format/lance-data-viewer | 用于浏览 Lance 数据集的只读 Web 界面               |
| lance-flink       | https://github.com/lance-format/lance-flink       | Lance 的 Apache Flink 连接器                       |
| lance-graph       | https://github.com/lance-format/lance-graph       | 基于 Lance 的 Cypher 图查询引擎                    |
| lance-trino       | https://github.com/lance-format/lance-trino       | Lance 的 Trino 连接器                              |
| pglance           | https://github.com/lance-format/pglance           | Lance 的 PostgreSQL 扩展                           |

### 从孵化毕业为子项目

一旦项目满足以下条件，PMC 可以投票将孵化子项目提升为子项目：

- 适当的仓库配置，包括 CI、问题跟踪和贡献指南
- 适当的代码标准执行，包括代码检查和测试
- 已建立的使用案例
- 除主要贡献者之外的社区采用
- 至少一名 Lance 维护者积极维护该项目

拥有写入权限的贡献者在毕业为子项目后将保留其访问权限。

### 项目许可证

所有托管在 [lance-format](https://github.com/lance-format) GitHub 组织中的 Lance 项目
均采用 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) 许可证。

### 外部集成

我们欢迎并鼓励在外部项目中进行 Lance 集成。
这些集成是对 Lance 社区的宝贵贡献，有助于扩展 Lance 生态系统。
当集成在外部项目中开发时，
集成代码和许可应遵循外部项目的准则和许可证。

## 报告安全漏洞

如发现任何安全漏洞，
请通过 [Lance 私有邮件列表](mailto:private@lance.org) 联系 PMC，
并在问题解决之前避免公开披露。

## 报告骚扰行为

Lance 社区遵循 [Rust 社区行为准则](https://www.rust-lang.org/policies/code-of-conduct)。
我们致力于为所有人提供一个热情和鼓舞人心的社区。
对参与者的骚扰行为将不会被容忍。
如遇此类情况，请向 [Lance 私有邮件列表](mailto:private@lance.org) 报告。
