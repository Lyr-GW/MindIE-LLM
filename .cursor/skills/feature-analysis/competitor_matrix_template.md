# 竞品对比矩阵模板（MindIE-LLM × vLLM × SGLang）

> 用法：复制本表为新分析的初始骨架，逐格填写"实现位置 / 关键算法 / 性能数据 / 限制"。
> 每格务必给出**出处**：`仓库:路径:行号` 或 `文档链接`。
> 严禁堆砌术语；如三方命名不一致，请先在最上方建立"概念映射表"。

## 0. 概念映射表（示例）

| 通用概念 | MindIE-LLM 术语 | vLLM 术语 | SGLang 术语 |
|----------|-----------------|-----------|-------------|
| 前缀复用 | Prefix Cache | Automatic Prefix Caching (APC) | RadixAttention |
| 分块预填 | SplitFuse | Chunked Prefill | Chunked Prefill |
| 推测解码 | 并行解码 / MTP | Speculative Decoding / MTP | Speculative Decoding (EAGLE / MTP) |
| KV 块管理 | BlockManager | KVCacheManager (v1) | TokenToKVPoolAllocator + RadixCache |
| 调度器 | Scheduler (FCFS / PDDS / Layerwise) | Scheduler (FCFS + chunked) | Scheduler (Async + Overlap) |
| 多 LoRA | Multi-LoRA | LoRA Manager (Punica) | LoRA |
| 专家并行 | Expert Parallel + EPLB | EP + DeepEP | EP + DeepEP |
| PD 分离 | PD 分离 / PDDS / KV Connector | Disaggregated Serving (v1) | Disaggregated Prefill / Decode |
| 量化 | W8A8 / W4A8 / W8A16 / KV int8 / FA3 / Attn 量化 / W8A8SC | FP8 / INT8 / AWQ / GPTQ / Marlin | FP8 / INT8 / AWQ / GPTQ |
| 图模式 | ACLGraph / ATBGraph | CUDA Graph / torch.compile | CUDA Graph / torch.compile |
| 硬件 | Ascend NPU (910B/910C/...) | NVIDIA GPU / AMD ROCm / TPU | NVIDIA GPU / AMD ROCm |

## 1. 调度与批处理

| 维度 | MindIE-LLM | vLLM | SGLang | 差距 / 优势 | 借鉴点 |
|------|-----------|------|--------|------------|--------|
| 调度策略 | FCFS / PDDS / Layerwise（`src/scheduler/policy/`、`scheduler.cpp`） | FCFS + chunked + priority（`vllm/v1/core/sched/scheduler.py`） | Async scheduler + cpu-gpu overlap | … | … |
| Chunked Prefill | SplitFuse（`mindie_llm/text_generator/plugins/splitfuse/`） | Chunked Prefill（`vllm/v1/`） | Default on | … | … |
| 异步调度 | Async Scheduling | Async output processor | Overlap CPU↔GPU | … | … |
| SLO 感知 | latency_predictor（`src/scheduler/latency_predictor/`） | – | – | 领先 | – |
| 抢占 | Preempt / Swap-out | Preempt + Recompute | Recompute | … | … |

## 2. KV Cache 管理

| 维度 | MindIE-LLM | vLLM | SGLang | 差距 / 优势 | 借鉴点 |
|------|-----------|------|--------|------------|--------|
| Block 分配 | hashless + prefix_cache 双轨（`src/block_manager/`） | Paged + APC hashed | Token-to-KV + RadixTree | … | … |
| Prefix Cache | `prefix_cache_block_allocator.cpp` | `vllm/v1/core/kv_cache_manager.py` | `mem_cache/radix_cache.py` | … | … |
| CPU-NPU Offload | `cpu_npu_block_allocator.cpp` | CPU swap | Hierarchical cache | 与 SGLang 思路一致 | 参考 SGLang `hicache` |
| KV 池化 (远端) | KV Cache 池化（DRAM/SSD） | LMCache / Mooncake 集成 | Mooncake / NIXL | … | … |
| CoW | `copy_on_write_tracker.cpp` | CoW on prefix sharing | CoW | … | … |
| 命中率统计 | `hit_rate_calculator.cpp` | Prometheus metrics | Prometheus metrics | … | … |

## 3. 并行策略

| 维度 | MindIE-LLM | vLLM | SGLang | 差距 / 优势 |
|------|-----------|------|--------|------------|
| TP | ✅ | ✅ | ✅ | 对齐 |
| PP | 部分 | ✅ | ✅ | 落后 |
| DP | ✅ | ✅（attention DP） | ✅ | 对齐 |
| EP | ✅ + 共享专家外置 + EPLB | ✅（DeepEP） | ✅（DeepEP） | 共享专家外置为差异点 |
| CP | ✅ | – / 实验性 | – | 领先 |
| SP | ✅ | 实验性 | 实验性 | 领先 |

## 4. 量化矩阵

| 量化方案 | MindIE-LLM | vLLM | SGLang | 备注 |
|----------|-----------|------|--------|------|
| W8A8 | ✅ | ✅ (FP8 + INT8) | ✅ | … |
| W4A8 混合 | ✅ | 实验性 | 实验性 | MindIE 差异点 |
| W8A16 | ✅ | ✅ (AWQ/GPTQ) | ✅ | … |
| KV int8 | ✅ | ✅ FP8 KV | ✅ FP8 KV | 类型差异 |
| FA3 量化 | ✅ | – | – | 领先 |
| Attention 量化 | ✅ | – | – | 领先 |
| 稀疏量化 W8A8SC / W16A16SC | ✅ | – | – | 领先 |
| Anti-Outlier | ✅ | SmoothQuant | SmoothQuant | 对齐 |

## 5. MoE 专项

| 维度 | MindIE-LLM | vLLM | SGLang |
|------|-----------|------|--------|
| Fused MoE Kernel | ATB / 自研 | Triton fused_moe | Triton fused_moe + Marlin |
| EPLB | ✅（`expert_parallelism_load_balancer.md`） | ✅ (DeepEP redundant) | ✅ |
| 共享专家外置 | ✅（`mix_shared_routing.md`） | – | – |
| Redundant Experts | ✅ | ✅ | ✅ |

## 6. 加速 / Speculative

| 维度 | MindIE-LLM | vLLM | SGLang |
|------|-----------|------|--------|
| MTP | ✅（`docs/.../mtp.md`） | ✅ | ✅ |
| EAGLE / EAGLE-2 | ❔ | ✅ | ✅ |
| Lookahead | ✅（`text_generator/plugins/la/`） | – | – |
| Memory Decoding | ✅（`plugins/memory_decoding/`） | – | – |
| Medusa | ❔ | ✅ | ✅ |

## 7. 长序列

| 维度 | MindIE-LLM | vLLM | SGLang |
|------|-----------|------|--------|
| RoPE 扩展 | YaRN / NTK（`docs/.../RoPEFactoryGuide.md`） | ✅ | ✅ |
| Ring/Striped Attention | CP / SP | 实验性 | 实验性 |
| 上下文长度上限 | 128k+ | 1M (block table 优化) | 1M+ |

## 8. PD 分离

| 维度 | MindIE-LLM | vLLM | SGLang |
|------|-----------|------|--------|
| 架构 | PDDS + KV Connector | Disaggregated v1 | Disaggregated |
| KV 链路 | HCCL / RDMA / 共享内存 | NIXL / Mooncake | Mooncake / NIXL |
| Routing | `src/load_balance/` | Frontend router | sgl-router |

## 9. 服务化

| 维度 | MindIE-LLM | vLLM | SGLang |
|------|-----------|------|--------|
| OpenAI 协议 | ✅ | ✅ | ✅ |
| vLLM 协议 | ✅ | 原生 | – |
| Triton 协议 | ✅ | 集成 | – |
| Function Call | ✅（`function_call.md`） | ✅ | ✅ |
| 结构化输出 | ✅（`structured_output.md`） | ✅ (xgrammar) | ✅ (xgrammar/llguidance) |
| 思考解析 | ✅ | ✅ | ✅ |
| Streaming | ✅ | ✅ | ✅ |

## 10. 后端 / 图模式

| 维度 | MindIE-LLM | vLLM | SGLang |
|------|-----------|------|--------|
| 图模式 | ACLGraph / ATBGraph | CUDA Graph + torch.compile | CUDA Graph + torch.compile |
| Kernel 库 | ATB / 自研 NPU kernel | FlashAttention / FlashInfer / Triton | FlashInfer / Triton |
| Attention 后端 | FA / FA2 / FA3-quant on NPU | FA2 / FA3 / FlashInfer / xFormers | FlashInfer / Triton |

## 11. 硬件矩阵

| 硬件 | MindIE-LLM | vLLM | SGLang |
|------|-----------|------|--------|
| Ascend NPU | ✅ | 社区适配中 | 社区适配中 |
| NVIDIA GPU | – | ✅ | ✅ |
| AMD MI | – | ✅ | ✅ |
| TPU | – | ✅ (TPU v5e/v6e) | – |
| Intel HPU / Gaudi | – | ✅ | – |

## 12. 可观测性

| 维度 | MindIE-LLM | vLLM | SGLang |
|------|-----------|------|--------|
| 指标输出 | `src/utils/` 共享内存计数器 + 内置 metrics | Prometheus + OTel | Prometheus + OTel |
| Trace | profiling util | OTel | OTel |
| Profiler | `mindie_llm/utils/` | torch profiler | torch profiler |

## 13. 结论模板

> 完成填表后，按"领先 / 对齐 / 落后"三档汇总：
>
> - **领先**：共享专家外置、FA3/Attn 量化、W8A8SC/W16A16SC、Layerwise Scheduler、SLO 感知调度。
> - **对齐**：Continuous Batching、Prefix Cache、Chunked Prefill、TP/DP/EP、Function Call、结构化输出。
> - **落后 / 待跟进**：EAGLE-2 / Medusa / xgrammar / 流水线并行成熟度 / 多硬件生态。
>
> 借鉴清单（按优先级）：
> 1. SGLang `hicache` 的多级 KV 层次（DRAM/SSD/Remote）抽象设计。
> 2. vLLM `v1` 的统一调度器代码结构与可插拔 Scheduler Policy。
> 3. vLLM / SGLang 的 xgrammar / llguidance 结构化输出引擎。
