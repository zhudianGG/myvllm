import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from myvllm.config import Config
from myvllm.sampling_params import SamplingParams
from myvllm.engine.sequence import Sequence
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.model_runner import ModelRunner

class LLMEngine:
    def __init__(self, model, **kwargs):
        config = Config(model, **kwargs)

        # start multiprocesses(TP > 1)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        
        # rank 0's ModelRunner
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def generate(self, prompts, sampling_params, use_tqdm=True):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        # main loop
        outputs = {}
        while not self.scheduler.is_finished():
            output, num_tokens = self.step()
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
        
        # output ranked by seq_id
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
                   for token_ids in outputs]
        return outputs
