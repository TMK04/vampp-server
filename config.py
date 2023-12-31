import numpy as np
import os

OUT_DIR = os.environ.get("OUT_DIR", "/notebooks/out")

HOST = os.environ.get("HOST", "0.0.0.0")
LOGLVL = os.environ.get("LOGLVL", "info")
PORT = int(os.environ.get("PORT", "8000"))

AUDIO_BATCH = int(os.environ.get("AUDIO_BATCH", "1"))
AUDIO_SR = int(os.environ.get("AUDIO_SR", "16000"))

FRAME_ATTIRE_MASK = np.array(os.environ.get("FRAME_ATTIRE_MASK", "5").split(","), dtype=int)
FRAME_ATTIRE_MASK = np.concatenate((FRAME_ATTIRE_MASK, -(FRAME_ATTIRE_MASK + 1)))
FRAME_BATCH = int(os.environ.get("FRAME_BATCH", "1"))
FRAME_INTERVAL = int(os.environ.get("FRAME_INTERVAL", "1"))

for key in ["MODEL_LLM_AUTHOR", "MODEL_LLM_NAME", "MODEL_SD_AUTHOR", "MODEL_SD_NAME"]:
  env_var = os.environ.get(key)
  if env_var is None:
    raise ValueError(f"env_var {key} is unset")

MODEL_FR_W = float(os.environ.get("MODEL_FR_W", "0.5"))

MODEL_LLM_DIR = os.path.join(os.environ.get("MODEL_LLM_AUTHOR"), os.environ.get("MODEL_LLM_NAME"))
MODEL_LLM_CONTEXT_LEN = int(os.environ.get("MODEL_LLM_CONTEXT_LEN", "8192"))
MODEL_LLM_SCALE_POS_EMB = float(os.environ.get("MODEL_LLM_SCALE_POS_EMB", "3.0"))

MODEL_SD_DIR = os.path.join(os.environ.get("MODEL_SD_AUTHOR"), os.environ.get("MODEL_SD_NAME"))
