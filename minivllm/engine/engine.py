# engine/engine.py

from engine.scheduler import Scheduler
from engine.batcher import build_batch
from model.model_runner import ModelRunner

class LLMEngine:
    def __init__(self):
        self.scheduler = Scheduler()
        self.model_runner = ModelRunner()
    
    def add_request(self, request):
        self.scheduler.add_request(request)
    
    def run(self):
        while True:
            active_requests = self.scheduler.step()

            if not active_requests:
                break
                
            inputs = build_batch(active_requests, self.model_runner.tokenizer)
            outputs = self.model_runner.forward(inputs["input_ids"])
            logits = outputs.logits
            next_tokens = logits[:, -1, :].argmax(dim=-1)

            for i, req in enumerate(active_requests):
                token = next_tokens[i].item()
                req.append_token(token)
            
            self.scheduler.running = [
                r for r in active_requests if not r.is_finished
            ]

            print(self.model_runner.decode(req.generated_tokens))