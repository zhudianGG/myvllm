这个系列要实现的推理框架包含以下核心组件：

组件	职责
LLMEngine	主循环，协调调度和执行
Scheduler	决定每一步执行哪些请求
BlockManager	页式 KV 缓存的分配与回收
ModelRunner	准备输入、执行模型、处理输出
Attention	集成 Flash Attention 和 KV Cache 写入
并行层	支持张量并行的线性层和 Embedding
整体数据流是这样的：

用户请求 → LLMEngine.add_request() → Scheduler.waiting 队列
                                          ↓
主循环 → Scheduler.schedule() → 选出本步要执行的请求
                                          ↓
       → ModelRunner.run() → 模型前向 + 采样
                                          ↓
       → Scheduler.postprocess() → 更新状态，终止判断
                                          ↓
       → 输出完成的请求

step() 被调用
    │
    ├─ scheduler.schedule()
    │   ├─ waiting 非空？尝试 Prefill admit
    │   └─ 否则 Decode，必要时抢占
    │
    ├─ model_runner.call("run", seqs, is_prefill)
    │   ├─ prepare_prefill() 或 prepare_decode()
    │   │   └─ 构造 input_ids, positions, slot_mapping, ...
    │   │   └─ set_context(...)
    │   │
    │   ├─ run_model()
    │   │   ├─ Prefill/Eager: model(input_ids) → compute_logits()
    │   │   └─ Graph: staging → replay → compute_logits()
    │   │
    │   ├─ sampler(logits, temperatures)
    │   │   └─ 只在 rank 0 执行
    │   │
    │   └─ reset_context()
    │
    └─ scheduler.postprocess(seqs, token_ids)
        ├─ 追加 token
        ├─ 检查终止条件
        └─ 终止的序列 deallocate + FINISHED

关键配置参数
@dataclass
class Config:
    model: str                          # 模型路径
    max_num_batched_tokens: int = 16384 # 单步最大 token 数
    max_num_seqs: int = 512             # 最大并发请求数
    max_model_len: int = 4096           # 最大序列长度
    gpu_memory_utilization: float = 0.9 # 显存利用率
    tensor_parallel_size: int = 1       # 张量并行度
    enforce_eager: bool = False         # 强制 Eager 模式
    kvcache_block_size: int = 256       # KV Cache 块大小

参数影响分析：

参数	影响
max_num_batched_tokens	Prefill 吞吐上限，太大可能 OOM
max_num_seqs	并发度上限，影响 Decode 吞吐
max_model_len	决定 Graph 捕获时的 block_tables 大小
gpu_memory_utilization	KV Cache 块数，太高容易 OOM
kvcache_block_size	块粒度，影响缓存命中率和元数据开销
enforce_eager	调试用，关闭 Graph 便于排查问题

相关note见 engine/note.txt