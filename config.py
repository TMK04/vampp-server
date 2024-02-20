import numpy as np
import os


def parseBool(key: str, default="") -> bool:
  return os.environ.get(key, default).lower() in ["true", "1"]


def parseInt(key: str, default="") -> int:
  return int(os.environ.get(key, default))


OUT_DIR = os.environ.get("OUT_DIR", "/notebooks/out")

DEBUG = parseBool("DEBUG")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = parseInt("PORT", "8000")
SHARE = parseBool("SHARE")

AUDIO_BATCH = parseInt("AUDIO_BATCH", "1")
AUDIO_SR = parseInt("AUDIO_SR", "16000")

FRAME_ATTIRE_MASK = np.array(os.environ.get("FRAME_ATTIRE_MASK", "5").split(","), dtype=int)
FRAME_ATTIRE_MASK = np.concatenate((FRAME_ATTIRE_MASK, -(FRAME_ATTIRE_MASK + 1)))
FRAME_BATCH = parseInt("FRAME_BATCH", "1")
FRAME_INTERVAL = parseInt("FRAME_INTERVAL", "1")

MODEL_FR_W = float(os.environ.get("MODEL_FR_W", "0.5"))

for key in ["MODELS_DIR", "MODEL_LLM_AUTHOR", "MODEL_LLM_NAME"]:
  env_var = os.environ.get(key)
  if env_var is None:
    raise ValueError(f"env_var {key} is unset")

MODELS_DIR = os.environ.get("MODELS_DIR")
MODEL_LLM_DIR = os.path.join(os.environ.get("MODEL_LLM_AUTHOR"), os.environ.get("MODEL_LLM_NAME"))
MODEL_LLM_CONTEXT_LEN = parseInt("MODEL_LLM_CONTEXT_LEN", "32768")
MODEL_LLM_SCALE_POS_EMB = float(os.environ.get("MODEL_LLM_SCALE_POS_EMB", "1.0"))
