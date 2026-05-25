---
name: feature-analysis
description: >-
  对 MindIE-LLM（昇腾大语言模型推理引擎）的任意特性进行端到端深度分析，覆盖：需求价值、需求设计、
  代码结构、实现细节、工程 trick 与不足、与 vLLM / SGLang 的竞品对比，以及性能 / 精度 / 兼容性 /
  可观测性等关键补充维度。适用于新人上手、特性 review、技术汇报、架构演进决策、迁移评估等场景。
  触发词：特性分析、feature 分析、feature understanding、技术解读、源码走读、竞品对比、对比 vllm、
  对比 sglang、深度解读、技术调研、特性 review、prefix cache 分析、MTP 分析、MoE 分析、调度分析、
  KV Cache 分析、量化分析、PD 分离分析、SplitFuse 分析、Speculative Decoding 分析等。
metadata:
  author: MindIE-LLM Community
  version: 1.0.0
  scope: MindIE-LLM repository
  language: zh-CN
license: Apache-2.0
compatibility: 适用于 MindIE-LLM master 及后续版本，与 docs/zh/user_guide/feature/ 列表保持同步
---

# Feature Analysis Skill —— MindIE-LLM 特性深度理解

> 目的：用一套**可复用、可验证、可量化**的方法论，把一个特性从"是什么 / 为什么 / 怎么做 / 做得怎么样 /
> 别人怎么做"五件事讲透。最终交付一份**结构化报告**，让读者既能在 5 分钟内把握全貌，也能按图索骥定位到具体文件 / 函数 / 算子。

---

## 1. 何时使用本 Skill

**应当使用**：
- 用户说"分析一下 XXX 特性"、"讲讲 prefix cache 是怎么实现的"、"对比一下 vllm 的 PagedAttention"。
- 需要为新人 onboarding 编写技术解读、为 PR review 提供背景、为架构演进做调研。
- 评估某特性是否值得移植 / 借鉴 / 重构 / 下线。

**不应使用**：
- 仅是修复一行 bug 或写一个补丁（用 debug / edit 流程即可）。
- 只是查询 API 用法（直接看 `docs/zh/user_guide/`）。

---

## 2. 标准分析流程（七步法）

> 每一步都必须**先用工具采集证据**，再形成结论；禁止凭主观印象总结。

### Step 1 — 锁定特性边界

1. 从 `docs/zh/user_guide/feature/README.md` 的"特性大类"找到归属（基础 / 量化 / 长序列 / 调度 / 加速 / 交互）。
2. 阅读对应的 `docs/zh/user_guide/feature/<feature>.md` 拿到官方简介、价值、限制、配置项。
3. 在 `docs/zh/developer_guide/architecture_design/` 中查找是否有专门的架构文档（如 `MoE.md`、`mtp.md`、`MultiLora.md`、`MicroBatch.md`）。
4. 用 Grep 检索 README 中的"特性叠加矩阵"，记录该特性与其他特性的兼容/不兼容关系。

**产出**：特性一句话定义、归属大类、官方价值描述、兼容性表行。

### Step 2 — 需求价值（Why）

围绕"业务收益 + 工程收益 + 场景适配"三层：

| 维度 | 关注点 | 取证方法 |
|------|--------|----------|
| 业务收益 | TTFT / TPOT / 吞吐 / 显存 / 精度的预期改善幅度 | docs 中的 benchmark 表、release_notes |
| 工程收益 | 是否解耦/简化了原有路径、是否减少手工调参 | 对比新增前后的调用链 |
| 适用场景 | 模型类型（Dense / MoE / 多模态）、序列长度、batch size、PD 拆分 | docs feature 页"使用约束"小节 |
| 反场景 | 何时**不应**开启（开销 > 收益） | 兼容性矩阵中标 ❌ / ❔ 的组合 |

**产出**：3–5 条价值描述 + 2–3 条反场景，每条至少引用一处文档或基准数据。

### Step 3 — 需求设计（What）

回答"为了拿到这些收益，系统需要做哪些抽象 / 协议 / 数据结构 / 流程"：

1. **抽象模型**：列出关键概念（如 `Block`、`SeqGroup`、`Scheduler Policy`、`KVConnector`、`PluginContext`）。
2. **生命周期**：请求从进入到响应，该特性在哪几个阶段介入（pre-schedule / schedule / pre-forward / forward / post-forward / sample / output）。
3. **接口契约**：对外暴露的 Python/C++ 接口、配置项（搜索 `config_manager/`、`mindie_llm/runtime/conf/`、`mindie_llm/runtime/config/`）。
4. **状态/数据结构变更**：是否新增表、新增字段、新增共享内存段（看 `src/utils/` 和 `src/sequence/`）。

**产出**：1 张时序图或状态图（mermaid），1 张数据结构关系图，关键配置项表。

### Step 4 — 代码结构（Where）

按 MindIE-LLM 的双层结构定位：

```
Python 层（mindie_llm/）
  ├── connector/               # 请求接入（HTTP/gRPC/OpenAI 协议）
  ├── server/                  # 服务化入口（main.py）
  ├── text_generator/          # 推理执行编排
  │   ├── generator.py         # 主流程
  │   ├── plugins/             # 高阶特性插件（prefix_cache / splitfuse / memory_decoding / la …）
  │   ├── samplers/            # 采样器
  │   ├── mempool/             # 显存池
  │   └── adapter/             # 模型适配
  ├── runtime/                 # 编译与运行时（compilation / layers / lora / model_runner / models / ops）
  ├── modeling / model_wrapper # 模型抽象（ATB / ACLGraph）
  ├── distributed/             # 分布式（TP/DP/EP/CP/SP）
  └── utils/                   # 日志/张量/Profiling

C++ 层（src/）
  ├── engine/                  # llm_engine、调度编排、Lora 管理、layerwise/loraops mixin
  ├── scheduler/               # FCFS / PDDS / Layerwise + latency_predictor + policy
  ├── block_manager/           # Block 分配 / Prefix Cache / CoW / LRU / hashless / CPU-NPU 双层
  ├── sequence/                # 请求与序列状态机
  ├── llm_manager / llm_manager_v2  # Python ↔ C++ 桥接 API
  ├── server/                  # gRPC / HTTP endpoint
  ├── executor/                # 执行器（与 worker 通信）
  ├── kernels/                 # 自定义算子
  ├── load_balance/            # 跨实例负载均衡
  ├── config_manager/          # 配置中心
  └── utils/                   # 共享内存 / 加密 / 日志 / ID 生成
```

**分析方法**：
1. 在 `docs/zh/user_guide/feature/<feature>.md` 中找出关键词（如 `Prefix Cache`、`MTP`、`SplitFuse`）。
2. 用 Grep 在 `mindie_llm/` 和 `src/` 中分别检索类名 / 文件名（**用 ripgrep 或 Grep 工具，禁用 cat/find**）。
3. 画出"调用链路图"：`Server endpoint → LLM Manager → Engine → Scheduler/BlockManager → Executor → Worker → Modeling → Kernel`，标注该特性插入的节点。

**产出**：一张分层调用链路图（mermaid 或 ASCII），每条边附 `path/to/file.cpp:Class::method` 形式的引用（用文中规定的 startLine:endLine:filepath 代码引用语法）。

### Step 5 — 实现细节（How）

针对每个关键文件，回答如下问题（至少给出 3 个）：

1. **核心数据结构**：字段含义、生命周期、所有者、并发可见性。
2. **核心算法**：复杂度、空间复杂度、最坏/平均情况。
3. **同步与并发**：锁粒度（`std::mutex` / `shared_mutex` / `atomic` / 无锁队列）、host-device 流同步点。
4. **内存管理**：是否使用 `std::shared_ptr` 引用计数、是否触发 host↔device 拷贝、是否复用 buffer。
5. **错误处理**：异常类型、降级路径（rollback / preempt / swap-out）。
6. **可配置项**：开关、阈值、超时、批大小上限。
7. **统计/可观测**：是否打点（命中率、抢占次数、转 swap 次数）、日志级别、是否上报 metrics。

> 引用代码必须使用 `startLine:endLine:filepath` 格式，并保留至少 1 行真实代码。

### Step 6 — 工程 Trick 与实现不足

**Trick 清单**（至少枚举 3 条，每条必须落到代码行）：
- 哈希型 vs 哈希less 块管理（`block_manager/hashless_block_allocator.cpp` vs `prefix_cache_block_allocator.cpp`）。
- CoW / 引用计数 / LRU evict 三件套。
- CPU-NPU 双层 Block 池化（`cpu_npu_block_allocator.cpp`）规避显存爆炸。
- Layerwise scheduler 混入（`engine/layerwise_mixin.cpp`）实现层粒度调度。
- ATB / ACLGraph 两套后端的静态化加速。
- 共享内存零拷贝（`src/utils/shared_memory*`）跨进程交换 KV / token。
- mempool（`mindie_llm/text_generator/mempool/`）避免反复 alloc/free。
- 预测式 latency 调度（`scheduler/latency_predictor/`）做 SLO 感知。

**不足清单**（至少列出 3 条，并给出**可验证**的复现路径或代码处）：
- 配置爆炸 / 特性兼容矩阵中存在 ❌ / ❔ 的"未支持组合"。
- 异常路径未覆盖（例如 OOM 时是 throw 还是 preempt）。
- 缺失指标 / 缺失 trace（grep `metrics`、`statistic`、`profiling` 看覆盖度）。
- 仅支持特定硬件代际 / 特定模型族。
- 是否存在硬编码常量、TODO、FIXME（`rg "TODO|FIXME|XXX|HACK"`）。

### Step 7 — vLLM / SGLang 竞品对比

> **必做**：本节是 skill 的差异化价值。对每个对照点，至少给出**结论 + 出处**两要素。

#### 7.1 标准对比维度（13 项）

| 维度 | 关注点示例 |
|------|------------|
| 1. 调度模型 | continuous batching、chunked prefill、PDDS、layerwise、prefix-aware |
| 2. KV Cache 管理 | PagedAttention block size、Prefix Cache、CoW、CPU offload、远端池化 |
| 3. 并行策略 | TP / PP / DP / EP / CP / SP 组合粒度 |
| 4. 量化支持 | W8A8 / W4A8 / W8A16 / KV int8 / FA3 / Attention 量化 / 稀疏量化 |
| 5. MoE | EP、共享专家外置、负载均衡（EPLB）、redundant experts |
| 6. 加速 | Speculative Decoding、MTP、Lookahead、Memory Decoding、SplitFuse |
| 7. 长序列 | RoPE 扩展、CP、SP、Ring/Striped Attention |
| 8. PD 分离 | 是否支持、链路、KV 传输（NCCL / HCCL / RDMA / 共享内存） |
| 9. 多 LoRA | 动态加载、batched LoRA kernels、CPU↔NPU swap |
| 10. 服务化 | OpenAI / vLLM / Triton / gRPC、流式、function call、结构化输出 |
| 11. 后端 | 图模式（ACLGraph / ATBGraph vs CUDA Graph）、kernel 库 |
| 12. 硬件 | Ascend NPU vs NVIDIA GPU vs AMD MI vs TPU |
| 13. 可观测性 | metrics、tracing、profiler |

#### 7.2 取证规则

- **vLLM**：以 `vllm-project/vllm` 主分支为准，引用 `docs/source/` 与 `vllm/core/`、`vllm/engine/`、`vllm/worker/`。
- **SGLang**：以 `sgl-project/sglang` 主分支为准，引用 `python/sglang/srt/` 下的 `scheduler.py`、`mem_cache/`、`speculative/`。
- 使用 WebSearch / WebFetch 获取**当前年份的最新文档**（注意系统时间，必要时检索"vLLM 2026"、"SGLang 2026"等带年份查询词，避免引用过期信息）。
- 当三方实现命名不一致时，做**概念映射表**而非术语堆叠（例：MindIE `Prefix Cache` ↔ vLLM `Automatic Prefix Caching` ↔ SGLang `RadixAttention`）。

#### 7.3 对比表必备模板

```
| 维度 | MindIE-LLM | vLLM | SGLang | 差距/优势 | 风险/借鉴点 |
|------|-----------|------|--------|-----------|-------------|
| ... | ...       | ...  | ...    | ...       | ...         |
```

最后给出 **3–5 条结论**：哪些维度领先、哪些维度对齐、哪些维度落后；落后项**建议借鉴的具体函数/模块名**。

---

## 3. 关键补充维度（务必覆盖）

仅做完前七步还不够，以下补充维度是工程上**必答题**：

1. **精度对齐**：是否提供与 HuggingFace / vLLM 的 logits / token 对齐脚本（看 `tests/` 与 `tools/`）。
2. **性能基准**：是否有可复现的 benchmark 入口（`examples/`、`tools/`、`scripts/`），KPI 是 TTFT / TPOT / 吞吐 / 显存 / 抢占次数 / 命中率。
3. **可观测性**：日志（`utils/log*`）、metrics（`src/utils/` 共享内存计数器）、profiling（`mindie_llm/utils/profiling`）。
4. **安全性**：输入校验、`security.md` 中的告警面、加密（`src/utils/` crypto）、共享内存权限。
5. **兼容性矩阵**：与 `docs/zh/user_guide/feature/README.md` 中的叠加表交叉验证。
6. **回滚 / Feature Flag**：是否可通过 `config.conf` 或环境变量在生产侧热关闭。
7. **测试覆盖**：单测 (`tests/ut`)、集成测试 (`tests/st`)、模型一致性测试。
8. **演进路线**：`docs/zh/release_notes.md`、git log、未来 milestone（搜索 `TODO`、`FIXME`、`WIP`、`v2`）。
9. **跨团队边界**：Python / C++ 边界、与 ATB / ACLGraph / CANN 的耦合点。
10. **失败学习**：历史 issue / PR 中的 regression、典型故障模式。

---

## 4. 取证工具白名单与禁忌

**优先使用**：
- `Read` 读取整文件；
- `Grep`（ripgrep 封装）按正则 / glob 检索类名、关键字、TODO；
- `Glob` 找文件；
- `WebSearch` / `WebFetch` 获取 vLLM、SGLang 最新文档与代码；
- `mermaid` 生成时序 / 数据流 / 模块图。

**禁止**：
- 使用 `cat / head / tail / find / sed / awk` —— 一律改用上面专用工具。
- 凭印象描述实现 —— 必须给出 `startLine:endLine:filepath` 引用。
- 抄旧版本文档结论 —— 始终核对 master 分支当前代码。

---

## 5. 输出报告模板（强制结构）

> 最终交付一篇 Markdown，建议命名 `docs/zh/developer_guide/feature_analysis/<feature>.md`。

```markdown
# <Feature 名> 深度解读（vX.Y）

## 0. TL;DR
一句话定义 + 三条核心价值 + 是否推荐开启。

## 1. 需求价值
- 业务收益：…
- 工程收益：…
- 适用 / 反场景：…
- 量化预期：TTFT/TPOT/吞吐/显存 数值或文档引用。

## 2. 需求设计
- 抽象模型与关键概念
- 端到端时序图（mermaid）
- 核心数据结构关系图
- 关键配置项表

## 3. 代码结构
- 调用链路图（标注插入点）
- 关键文件清单（含 startLine:endLine:filepath 引用）
- Python↔C++ 边界

## 4. 实现细节
- 关键算法 + 复杂度
- 同步 / 并发 / 内存
- 错误处理 / 降级
- 可观测性

## 5. 工程 Trick
逐条列出，每条带代码引用与收益评估。

## 6. 实现不足 / 风险
逐条列出，每条给出复现路径与改进建议。

## 7. 竞品对比（vLLM / SGLang）
- 维度对比表（13 维标准模板）
- 概念映射表
- 结论：领先 / 对齐 / 落后；可借鉴模块清单。

## 8. 补充维度
精度对齐 / 基准 / 可观测 / 安全 / 兼容矩阵 / 回滚 / 测试 / 演进 / 跨团队 / 历史故障。

## 9. 行动建议
- 短期可落地的优化点（≤3 条）
- 中期演进方向（≤3 条）
- 需要更多调研的开放问题
```

---

## 6. 三种典型范例（按特性套用）

为减少冷启动成本，下列三类特性已被预先打点，新分析时优先复用：

### 6.1 KV / 调度类（示例：Prefix Cache）
- 关键路径：`src/block_manager/prefix_cache_block_allocator.cpp` → `src/scheduler/scheduler.cpp` → `src/engine/llm_engine.cpp`
- 竞品对照：vLLM `vllm/v1/core/kv_cache_manager.py`、SGLang `python/sglang/srt/mem_cache/radix_cache.py`
- 重点指标：命中率（`hit_rate_calculator.cpp`）、复用块数、CoW 触发次数。

### 6.2 加速 / Decoding 类（示例：MTP / Speculative Decoding）
- 关键路径：`mindie_llm/text_generator/plugins/memory_decoding/`、`la/`，`docs/.../mtp.md`
- 竞品对照：vLLM `vllm/spec_decode/`、SGLang `python/sglang/srt/speculative/`
- 重点指标：草稿命中率、TPOT 改善、回退率。

### 6.3 量化 / MoE / 并行类（示例：MoE + EP + EPLB）
- 关键路径：`mindie_llm/runtime/layers/`、`mindie_llm/distributed/`，`docs/.../MoE.md`、`expert_parallelism_load_balancer.md`
- 竞品对照：vLLM `vllm/model_executor/layers/fused_moe/`、SGLang `python/sglang/srt/layers/moe/`
- 重点指标：路由不均衡度、TPOT、EP 通信占比。

---

## 7. 质量自检清单（提交前过一遍）

- [ ] 报告 7 大节齐全，且每节都有真实代码引用或文档链接。
- [ ] 调用链路图节点与代码行号一一对应。
- [ ] 竞品对比表 ≥ 8 行，且每行都标注了 vLLM / SGLang 出处。
- [ ] 至少给出 3 条 trick + 3 条不足，**不允许全是优点**。
- [ ] 至少给出 3 条可执行的改进建议。
- [ ] 已用 ripgrep 搜过 `TODO/FIXME/HACK` 并在报告中点名。
- [ ] 已交叉验证 `docs/zh/user_guide/feature/README.md` 中的兼容性矩阵。
- [ ] 报告语言为简体中文，符合本仓库语言约定。

---

## 8. 反例与禁忌（避免落入的陷阱）

- **只讲优点不讲缺点** —— 失去工程价值。
- **只画大图不下沉到行号** —— 等于没分析。
- **竞品对比只罗列功能名** —— 必须做"概念映射 + 实现差异 + 性能差异"三段式。
- **不区分 Python 与 C++ 路径** —— 在 MindIE-LLM 里二者是双层骨干，必须分开标注。
- **忽略硬件差异** —— Ascend NPU 的内存模型、流模型、图模式与 GPU 有本质区别，迁移结论前必须声明。
- **结论时效失效** —— vLLM / SGLang 演进极快，引用必须带 commit / tag / 日期。

---

## 9. 引用样例（务必使用本仓库规定的引用格式）

行级代码引用（来自本仓库现有文件）：

```32:55:README.md
## 🔍 目录结构

``` text
 ├── mindie_llm                                     # Python 推理框架主模块
 │   ├── connector                                  # 请求接入层
 │   ├── text_generator                             # 核心推理引擎
 │   ├── modeling                                   # 模型封装抽象
 │   ├── runtime                                    # 运行时编译和模型加载
 │   ├── utils                                      # 工具模块：日志/张量/Profiling/验证等
```

```5:32:docs/zh/developer_guide/architecture_design/architecture_overview.md
## 概述

**MindIE LLM**（Mind Inference Engine for Large Language Models）作为 MindIE 系列推理组件中的重要一员...
```

> 写作时把 `startLine:endLine:filepath` 替换为实际文件与行号；新增/未提交的代码用 markdown 代码块（带语言标签）。

---

## 10. Skill 元数据 & 维护

- **维护者**：MindIE-LLM 文档/架构 SIG。
- **更新触发**：每次 `docs/zh/user_guide/feature/README.md` 大类变更、`src/scheduler/` 或 `src/block_manager/` 重构、vLLM / SGLang 重大版本发布时更新本 Skill。
- **版本策略**：语义化版本（major 重构方法论 / minor 新增维度 / patch 修订示例）。
- **License**：Apache-2.0，与主仓库一致。

---

> **一句话总结**：本 Skill 用"七步法 + 13 维竞品矩阵 + 10 项补充维度 + 8 节强制报告模板"，把 MindIE-LLM 任意特性从需求到代码、从优点到缺陷、从单仓视角到生态对标，讲成一份"看完即可上手 / 借鉴 / 演进"的工程文档。
