from exllamav2.generator import ExLlamaV2StreamingGenerator

from .model import cache, model
from .tokenizer import stop_conditions, tokenizer

generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
generator.set_stop_conditions(stop_conditions)
generator.warmup()
