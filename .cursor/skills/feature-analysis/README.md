# Feature Analysis Skill

本目录提供面向 **MindIE-LLM** 项目的"特性深度理解 Skill"。Skill 的目标是让任何工程师（无论新人还是资深维护者）都能用同一套**七步法 + 13 维竞品矩阵 + 8 节强制报告结构**，把一个特性讲清楚、讲透彻、可对标、可演进。

## 文件清单

| 文件 | 作用 |
|------|------|
| `SKILL.md` | Skill 主入口，包含使用场景、七步分析流程、补充维度、质量自检清单。 |
| `report_template.md` | 标准化的特性分析报告模板，直接复制即可填写。 |
| `competitor_matrix_template.md` | 与 vLLM / SGLang 的 13 维竞品对比矩阵骨架。 |

## 快速开始

1. 阅读 `SKILL.md`，明确分析边界与七步流程。
2. 复制 `report_template.md` 到 `docs/zh/developer_guide/feature_analysis/<feature>.md`。
3. 复制 `competitor_matrix_template.md` 到同目录，按 0–13 节逐格填写。
4. 完成后，按 `SKILL.md` 第 7 节"质量自检清单"自检。
5. 通过 PR 进入仓库，归档到 `docs/zh/developer_guide/feature_analysis/`。

## 何时触发本 Skill

- "分析 / 解读 / 讲讲某特性"。
- "和 vLLM、SGLang 比起来怎么样"。
- "新人 onboarding，先看哪些代码"。
- "评估某特性是否值得借鉴 / 移植 / 重构"。

## 何时不应使用

- 修一个小 bug、回答一个 API 问题、调一个配置参数。

## 维护约定

- 每次 `docs/zh/user_guide/feature/README.md` 大类变更、`src/scheduler/` 或 `src/block_manager/` 重构、vLLM / SGLang 重大版本发布时，同步更新本 Skill。
- 语义化版本：`major.minor.patch`，与 `SKILL.md` frontmatter 一致。
- License：Apache-2.0，与主仓库一致。
