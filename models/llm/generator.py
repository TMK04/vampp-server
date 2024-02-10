from exllamav2.generator import ExLlamaV2StreamingGenerator

from .models import llm_cache, llm_model, sd_cache, sd_model
from .tokenizer import stop_conditions, tokenizer

generator = ExLlamaV2StreamingGenerator(llm_model, llm_cache, tokenizer, sd_model, sd_cache)
generator.set_stop_conditions(stop_conditions)
generator.warmup()
