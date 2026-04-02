import os
import time
from random import randint, seed
from myvllm import LLM, SamplingParams

def benchmark():
    num_seqs = 256
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, 1024))]
                        for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, 1024))
                       for _ in range(num_seqs)]
    # Warmup
    llm.generate(["Benchmark: "], SamplingParams())

    # time
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Tim: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
