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