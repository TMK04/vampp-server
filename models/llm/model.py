from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
)

from .config import config

model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, lazy=True)
print(f"Loading {config.model_dir}")
model.load_autosplit(cache)
