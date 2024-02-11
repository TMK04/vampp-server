from exllamav2 import ExLlamaV2Config
from pathlib import Path
import os

from server.config import (MODEL_LLM_DIR, MODEL_LLM_CONTEXT_LEN, MODEL_LLM_SCALE_POS_EMB)

models_dir = Path(__file__).parent / "models"

config = ExLlamaV2Config()
config.model_dir = os.path.join(models_dir, MODEL_LLM_DIR)
config.max_seq_len = MODEL_LLM_CONTEXT_LEN
config.scale_pos_emb = MODEL_LLM_SCALE_POS_EMB
config.prepare()
