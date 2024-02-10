from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
)

from .configs import llm_config, sd_config

llm_model = ExLlamaV2(llm_config)
llm_cache = ExLlamaV2Cache(llm_model, lazy=True)
print(f"Loading {llm_config.model_dir}")
llm_model.load_autosplit(llm_cache)

sd_model = ExLlamaV2(sd_config)
sd_cache = ExLlamaV2Cache(sd_model, lazy=True)
print(f"Loading {sd_config.model_dir}")
sd_model.load_autosplit(sd_cache)
