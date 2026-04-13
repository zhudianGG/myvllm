# engine/request.py

class Request:
    def __init__(self, prompt, max_tokens=50):
        self.prompt = prompt
        self.max_tokens = max_tokens

        self.generated_tokens = []
        self.is_finished = False

        # KV cache
        self.past_key_values = None
    
    def append_token(self, token):
        self.generated_tokens.append(token)
        if len(self.generated_tokens) >= self.max_tokens:
            self.is_finished = True
        