# 结构化输出（Structured Output）特性分析文档

本目录收录 MindIE-LLM 结构化输出（Guided Decoding / Structured Output）特性的端到端深度剖析文档，可作为：

- 内部 onboarding 材料
- 简历项目素材
- 技术分享 PPT 蓝本
- 面试问答库

## 文件清单

| 文件 | 格式 | 大小 | 说明 |
|---|---|---|---|
| [`structured_output_analysis.md`](./structured_output_analysis.md) | Markdown | ~84 KB | 完整分析文档，1838 行，18 个章节 |
| [`structured_output_wiki.html`](./structured_output_wiki.html) | 单文件 HTML Wiki | ~140 KB | 带侧边栏导航 / 暗色主题 / mermaid 流程图渲染 / 代码高亮，3009 行；25 章 + 3 附录 |

## 内容总览

文档按"五部分 + 附录"组织：

### 第一部分 · 总览与架构（章节 1-3）
- 特性总览、配置项、互斥矩阵
- 端到端架构与数据流（含 mermaid 全栈图）
- 三层抽象设计（Manager / Backend / Grammar）

### 第二部分 · 设计决策（章节 4）
- 10 个关键设计决策（每条带真实代码行号）

### 第三部分 · 深度专题（章节 5-7）
- **专题 A**：`sync_states_for_decode` 状态机流程（mermaid 决策树）
- **专题 B**：NPU bitmask 位运算正确性证明
- **专题 C**：与 vLLM 源码逐函数对照

### 第四部分 · 工程化（章节 8-10）
- 测试策略与覆盖
- 简历 STAR 范本（中英双版）
- 可外推的工程经验

### 第五部分 · 进阶详解（章节 11-18）
- 端到端请求生命周期追踪（含真实 token 例子）
- 关键代码精读（带行号注释）
- C++ / Python / Proto 三端字段映射表
- Suffix Search 兜底算法剖析
- 性能模型与瓶颈分析
- 已知问题、限制与未来演进路线
- 面试常见追问 30 题 + 答题要点
- 关键单测剖析

### 第六部分 · v3 新增（仅 HTML 版有，章节 19-25）
- JSON Schema → FSM 编译图解
- 异步路径深度追踪（forward_loop）
- GIL 与并发模型分析
- 三方对比：MindIE vs vLLM vs SGLang
- C++/Python 状态机翻译层
- Bug 模式总结与防御性编码
- Production Readiness 检查表

### 附录
- A. 相关代码索引
- B. 术语表
- C. 参考文献

## 本地查看 HTML Wiki

HTML 版本是单文件，可直接用浏览器打开（CDN 依赖：mermaid.js / highlight.js）：

```bash
# 方式 1：直接用浏览器打开
xdg-open structured_output_wiki.html       # Linux
open structured_output_wiki.html           # macOS
start structured_output_wiki.html          # Windows

# 方式 2：本地起 HTTP server（推荐，避免 file:// 协议下某些浏览器限制）
python3 -m http.server 8000
# 然后访问 http://localhost:8000/structured_output_wiki.html
```

## 相关代码位置

- 源代码：`mindie_llm/text_generator/plugins/structured_output/`
- 入参校验：`src/server/endpoint/utils/infer_param.cpp`
- proto 定义：`proto/model_execute_data.proto`（field 34 / 35）
- 用户文档：`docs/zh/user_guide/feature/structured_output.md`
- 单测：`tests/pythontest/cpu/text_generator/plugins/structured_output/`

## 文档维护

- **基于版本**：MindIE-LLM master 分支（2026/05 时点）
- **特性版本**：MindIE 2.3.0（CANN 8.5.0）
- **文档版本**：v3（深度迭代版）
