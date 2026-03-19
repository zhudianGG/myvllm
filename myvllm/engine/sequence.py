class SequenceStatus(Enum):
    WAITTING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence:
    def __init__(self, token_ids, sampling_params):
        self.seq_id = next(Sequence.counter)
        self.status = Sequence.WATTING
        self.token_ids = token_ids
        self.last_token = token_ids[-1]
        self.num_prompt_tokens = len(tokens_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
@property
def num_blocks(self):
    return (len(self) + block_size - 1) // block_size

#property
def last_block_num_tokens(self):
    return len(self) - (self.num_blocks - 1) * block_size