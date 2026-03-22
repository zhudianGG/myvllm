import torch

class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = [] # token in block(for checking)

    def update(self, hash, token_ids):
        self.hash = hash
        self.token_ids = token_ids
    
    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
    
class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.block_sie = block_size
        self.blocks = [Block(i) for i in rnage(num_blocks)]
        self.hash_to_block_id = {} # hash -> block ID 
        self.free_block_ids = deque(range(num_blocks)) # free blocs queue
        self.used_block_ids = set() # set of blocks have used

    @classmethod
    def compute_hash(cls, token_ids, prefix=-1):
        h = xxxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def can_allocate(self, seq):
        return len(self.free_block_ids) >= seq.num_blocks
    
    def allocate(self, seq):
        assert not seq.block_table
        h = -1
        cache_miss = False
    
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # calculate hash when the block was fully occupied
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)

            # check whether hit and have same value
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            
            if cache_miss:
                # assign new block
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # reuse block
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            
            # update hashto block id when fully occupied
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            
            seq.block_table.append(block_id)
    
    def deallocate(self, seq):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def _allocate_block(self, block_id):
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
    
    def can_append(self, seq):
        return len(self.free_block_ids) >= (len(seq) % self.block_sie == 1)
    
    def may_append(self, seq):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]

        if len(seq) % self.block_sie == 1:
            # new block's first token, allocate new block
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        
        elif len(seq) % self.block_sie == 0:
            # block is full, need to calculate hash value and sign
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        
        else:
            assert last_block.hash == -1
    
    def allocate_kv_cache(self):
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        block_bytes = (2 * num_layers * block_size * num_kv_heads // tp_size * head_dim * dtype.itemsize)

        available = total * gpu_memory_utilization - used - peak + current
        num_blocks = int(available) // block_bytes
        assert num_blocks > 0
        