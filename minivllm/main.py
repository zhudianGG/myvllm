# main.py

from engine.engine import LLMEngine
from engine.request import Request
from model.model_runner import ModelRunner

engine = LLMEngine()

engine.add_request(Request("Hello, how are you?"))
engine.add_request(Request("Explain KV cache in simple terms."))

engine.run()