from exllamav2 import ExLlamaV2Config
import os

from server.config import (MODELS_DIR, MODEL_LLM_DIR, MODEL_LLM_CONTEXT_LEN, MODEL_LLM_SCALE_POS_EMB)

wd = os.path.join(MODELS_DIR, "llm")

config = ExLlamaV2Config()
config.model_dir = os.path.join(wd, MODEL_LLM_DIR)
config.max_seq_len = MODEL_LLM_CONTEXT_LEN
config.scale_pos_emb = MODEL_LLM_SCALE_POS_EMB
config.prepare()
