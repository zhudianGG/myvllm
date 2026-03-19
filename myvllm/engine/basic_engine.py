def generate(model, input_ids, max_tokens):
    for _ in range(max_tokens):
        logits = model(input_ids)
        next_token = sample(logits[:, -1]) # only for the last position logits
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if next_token == eos_token:
            break
    return input_ids