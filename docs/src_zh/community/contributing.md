# 贡献指南

通常，代码贡献以 GitHub PR 的形式进行，需要拥有写入权限的维护者进行审查和批准。

## 约定式提交（Conventional Commits）

Lance 项目使用 [Conventional Commits](https://www.conventionalcommits.org) 标准编写提交信息。
此标准有助于区分：

- **破坏性变更** 与 **非破坏性变更**（使用 `!` 和 `BREAKING CHANGE:` 页脚）
- **功能**（`feat:`）、**修复**（`fix:`）、**文档更新**（`docs:`）和其他变更类型

遵循此标准的提交信息将在每次发布时用于自动生成发布说明。

## 功能设计提案

Lance 中的设计随着社区的输入和共识自然演进。
重大技术变更通过以下方式进行有机讨论：

- **发起讨论**：创建 GitHub Discussion 发布你的设计提案并收集社区反馈。使用讨论线程探索不同方面和替代方案
- **迭代设计**：与社区互动，根据他们的意见和专业知识完善方案
- **起草 PR 讨论细节**：一旦大致方向被社区接受，发布 Draft PR 帮助确定实现细节。鼓励使用 Draft PR，因为它们有助于进行具体的讨论
- **分解变更**：将大型 Draft PR 拆分为更小的增量 PR，以便更容易审查并展示进展
- **正式投票**：拥有写入权限的维护者可以批准与设计相关的代码修改。如果设计需要 Lance 格式规范变更，将在 GitHub Discussions 上按照 [投票要求](./voting.md#voting-requirements) 进行单独投票

## AI 工具集成

我们鼓励贡献者持续改进与 AI 工具的集成，包括：

- 增强编码代理指南，如 `AGENTS.md` 和 `CLAUDE.md`
- 向 AI 代码审查工具提供反馈
- 开发和改进 AI 驱动的 GitHub Actions

## 项目特定贡献指南

每个项目在名为 `CONTRIBUTING.md` 的文件中维护其详细的贡献指南。
