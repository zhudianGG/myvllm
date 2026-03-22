import torch

@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor = None
    Context_lens: torch.Tensor = None
    block_tables: torch.Tensor = None

_CONTEXT = Context()

def set_context(is_prefill, cu_seqlens_q = None, cu_seqlens_k = None,
                max_seqlen_q = 0, max_seqlen_k = 0, slot_mapping=None,
                context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k,
                       max_seqlen_q, max_seqlen_k, slot_mapping,
                       context_lens, block_tables)

def get_context():
    return _CONTEXT

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()

def prepare_prefill(self, seqs):
    input_ids = []
    positions = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []
    block_tables = None

    for seq in seqs:
        seqlen = len(seq)
        # for no cache part
        input_ids.extend(seq[seq.num_cached_tokens:])
        positions.extend(list(range(seq.num_cached_tokens, seqlen)))

        seqlen_q = seqlen - seq.num_cached_tokens
        seqlen_k = seqlen
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)

        if not seq.block_table:
            continue
        for i in range(seq.num_cached_blocks, seq.num_blocks):
            start = seq.block_table[i] * self.block_size
            if i != seq.num_blocks - 1:
                end = start + self.block_size
            else:
                end = start + seq.last_block_num_tokens
            slot_mapping.extend(list(range(start, end)))
    # prefix cache
    if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
        block_tables = self.prepare_block_tables(seqs)
    
    input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(input_ids, dtype= torch.int64, pin_memory=True).cuda(non_blocking=True)
    cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                slot_mapping, None, block_tables)
    return input_ids, positions

def prepare_decode(self, seqs):
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []

    for seq in seqs:
        input_ids.append(seq.last_token)
        positions.append(len(seq) - 1)
        context_lens.append(len(seq))
        
        slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
    
    input_ids = torch.tensor(input_ids, dtype = torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(positions, dtype = torch.int64, pin_memory=True).cuda(non_blocking=True)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    block_tables = self.prepare_block_tables(seqs)

    set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
    return input_ids, positions

def prepare_block_tables(self, seqs):
    max_len = max(len(seq.block_table) for seq in seqs)
    block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
    block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True)
    return block_tables

@torch.inference_mode()
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        # CUDA Graph path
        bs = input_ids,size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars

        # write staging tensor
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero()
        graph_vars["context_lens"][:bs] = context.Context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])
@torch.inference_mode()
def capture_cudagraph(self):
    max_bs = min(self.config.max_num_seqs, 512)
    max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size

    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max,bs, dtype=torch.int32)
    block_tables = toch.zeros(max_bs, max_num_blocks, dtype = torch.int32)
    outputs = torch.zeros(max_bs, self.config.hf_config.hiden_size)

    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}
    self.graph_pool = None

    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()
        set_context(False, slot_mapping=slot_mapping[:bs],
                    context_lens=context_lens[:bs], block_tables=block_tables[:bs])

        # warm up
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

        # capture
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
        
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        self.graph[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    self.graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        outputs=outputs,
    )
def run(self, seqs, is_prefill):
    # prepare input
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

    # execute the model
    logits = self.run_model(input_ids, positions, is_prefill)

    # sample (only in rank 0)
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

    reset_context()
    return token_ids

class ModelRunner:
    def __init__(self, config, rank, event):
        self.config = config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank

        dist.init_process_group("nccl", "tcp:/localhost:2333", 
                                self.world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.hf_config.torch_dtype)
        torch.set_default_devices("cuda")

        self.model = Qwen3ForCausalLM(config.hf_config)
        load_model(self.model, config.model)
        self.sampler = self.Sampler()

        # warm-up and resource allocate
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # IPC Setting
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop() # enter loop
    
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()
    
