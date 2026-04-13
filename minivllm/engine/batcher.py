# engine/batcher.py

def build_batch(requests, tokenizer):
    texts = []
    for req in requests:
        if not req.generated_tokens:
            texts.append(req.prompt)
        else:
            texts.append("")
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    return inputs