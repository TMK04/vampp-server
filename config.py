import numpy as np
import os

OUT_DIR = os.environ.get("OUT_DIR", "/tmp")

AUDIO_BATCH = int(os.environ.get("AUDIO_BATCH", "1"))
AUDIO_SR = int(os.environ.get("AUDIO_SR", "16000"))

FRAME_ATTIRE_MASK = np.array(os.environ.get("FRAME_ATTIRE_MASK", "5").split(","), dtype=int)
FRAME_ATTIRE_MASK = np.concatenate((FRAME_ATTIRE_MASK, -(FRAME_ATTIRE_MASK + 1)))
FRAME_BATCH = int(os.environ.get("FRAME_BATCH", "1"))
FRAME_INTERVAL = int(os.environ.get("FRAME_INTERVAL", "1"))

for key in ["MODEL_LLM_DIR", "MODEL_LLM2_DIR"]:
  env_var = os.environ.get(key)
  if env_var is None:
    raise ValueError(f"env_var {key} is unset")

MODEL_FR_W = float(os.environ.get("MODEL_FR_W", "0.5"))

MODEL_LLM_DIR = os.environ.get("MODEL_LLM_DIR")
MODEL_LLM_CONTEXT_LEN = int(os.environ.get("MODEL_LLM_CONTEXT_LEN", "8192"))
MODEL_LLM_SCALE_POS_EMB = float(os.environ.get("MODEL_LLM_SCALE_POS_EMB", "3.0"))

MODEL_LLM2_DIR = os.environ.get("MODEL_LLM2_DIR")
MODEL_LLM2_CONTEXT_LEN = int(os.environ.get("MODEL_LLM2_CONTEXT_LEN", "4096"))
MODEL_LLM2_SCALE_POS_EMB = float(os.environ.get("MODEL_LLM2_SCALE_POS_EMB", "2.0"))
