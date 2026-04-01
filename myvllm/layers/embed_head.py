class VocalbParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader
    
    def weight_loader(self, param, loaded_weight):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)
    
    def forward(self, x):
        if self.tp_size > 1:
            # only handle with this rank's vocab range
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # set to zero for pos not in the range, then use all_reduce to gather
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y
    
    class ParallelLMHead(VocalbParallelEmbedding):
        def __init__(self, nu_embeddings, embedding_dim, bias=False)
            assert not bias
            super().__init__(num_embeddings, embedding_dim)
        
        def forward(self, x):
            context = get_context()
            if context.is_prefill:
                # prefill only use each seq's last pos
                last_indices = context.cu_seqlens_q[1:] - 1
                x = x[last_indices].contigouous()
            
            logits = F.linear(x, self.weight)

            if self.tp_size > 1:
                # rank 0 gather -> gather to full vocab
                all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
                dist.gather(logits, all_logits, dst=0)
                logits = torch.cat(all_logits, dim=-1) if self.tp_rank == 0 else None
            
            return logits