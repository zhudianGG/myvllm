class Scheduler:
    def __init__(self, config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.block_manager = BlockManager(...)
        self.waiting = deque() # WAITING Prompt
        self.running = deque() # RUNNING Prompt

    def schedule(self):
        scheduled = []
        num_seqs = 0
        num_batched_tokens = 0

        while self.waiting:
            seq = self.waiting[0]
            # check limitations
            if num_seqs >= self.max_num_seqs:
                break
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
                break
            if not self.block_manager.can_allocate(seq):
                break
        
        # pass
        num_seqs += 1
        self.block_manager.allocate(seq)
        num_batched_tokens += len(seq) - seq.num_cached_tokens
        seq.status = Sequencestatus.RUNNING
        self.waiting.popleft()
        self.running.attend(seq)
        scheduled.append(seq)

        if scheduled:
            return scheduled, True # True -> Prefill
        
        # Decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()

            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop()) # get the end of queue
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled.append(seq)
            
            assert scheduled
            self.running.extendleft(reversed(scheduled))
            return scheduled, False # False -> Decode
    
    def preempt(self, seq):
        seq.status = SequenceStatus.WATTING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq) # assign to the beginning of the queue, handle next time
    
    def postprocess(self, seqs, token_ids):
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)

            if (not seq.ignore_eos and token_id == self.eos) or \
                seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        