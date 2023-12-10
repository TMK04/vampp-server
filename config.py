from dotenv import load_dotenv

load_dotenv()

import numpy as np
import os

TMP_DIR = os.environ.get("TMP_DIR", "/tmp")
TMP_FILENAME = os.environ.get("TMP_FILENAME", "temp")

AUDIO_BATCH = int(os.environ.get("AUDIO_BATCH", "1"))
AUDIO_SR = int(os.environ.get("AUDIO_SR", "16000"))

FRAME_ATTIRE_MASK = np.array(os.environ.get("FRAME_ATTIRE_MASK", "5").split(","), dtype=int)
FRAME_ATTIRE_MASK = np.concatenate((FRAME_ATTIRE_MASK, -(FRAME_ATTIRE_MASK + 1)))
FRAME_BATCH = int(os.environ.get("FRAME_BATCH", "1"))
FRAME_INTERVAL = int(os.environ.get("FRAME_INTERVAL", "1"))

for key in ["MODEL_LLM_DIR"]:
  env_var = os.environ.get(key)
  if env_var is None:
    raise ValueError(f"env_var {key} is unset")

MODEL_LLM_DIR = os.environ.get("MODEL_LLM_DIR")
# gpu_split
MODEL_LLM_GS = os.environ.get("MODEL_LLM_GS", "46")
MODEL_LLM_CONTEXT_LEN = int(os.environ.get("MODEL_LLM_CONTEXT_LEN", "4096"))
