from exllamav2 import ExLlamaV2Config
from pathlib import Path
import os

from server.config import (MODEL_LLM_DIR, MODEL_LLM_CONTEXT_LEN, MODEL_LLM_SCALE_POS_EMB,
                           MODEL_SD_DIR)

models_dir = Path(__file__).parent / "models"

llm_config = ExLlamaV2Config()
llm_config.model_dir = os.path.join(models_dir, MODEL_LLM_DIR)
llm_config.max_seq_len = MODEL_LLM_CONTEXT_LEN
llm_config.scale_pos_emb = MODEL_LLM_SCALE_POS_EMB
llm_config.prepare()

sd_config = ExLlamaV2Config()
sd_config.model_dir = os.path.join(models_dir, MODEL_SD_DIR)
sd_config.max_seq_len = MODEL_LLM_CONTEXT_LEN
sd_config.scale_pos_emb = MODEL_LLM_SCALE_POS_EMB
sd_config.prepare()
