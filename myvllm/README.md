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

相关note见 engine/note.txt