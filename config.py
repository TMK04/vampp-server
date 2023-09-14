from dotenv import load_dotenv

load_dotenv()

import numpy as np
import os

AUDIO_BATCH = int(os.environ.get("AUDIO_BATCH", "1"))
AUDIO_SR = int(os.environ.get("AUDIO_SR", "16000"))

FRAME_ATTIRE_MASK = np.array(os.environ.get("FRAME_ATTIRE_MASK", "5,-6").split(","), dtype=int)
FRAME_BATCH = int(os.environ.get("FRAME_BATCH", "1"))
FRAME_INTERVAL = int(os.environ.get("FRAME_INTERVAL", "1"))
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", "0"))

for key in [
    "MODEL_ATTIRE_PATH", "MODEL_FR_PATH", "MODEL_LLM_PATH", "MODEL_MULTITASK_PATH", "MODEL_PL_PATH",
    "MODEL_RFR_PATH", "MODEL_SS_PATH", "MODEL_SS_PRETRAINED_PATH"
]:
  env_var = os.environ.get(key)
  if env_var is None:
    raise ValueError(f"env_var {key} is unset")

MODEL_ATTIRE_PATH = os.environ.get("MODEL_ATTIRE_PATH")
MODEL_FR_PATH = os.environ.get("MODEL_FR_PATH")
MODEL_MULTITASK_PATH = os.environ.get("MODEL_MULTITASK_PATH")
MODEL_PL_PATH = os.environ.get("MODEL_PL_PATH")
MODEL_RFR_PATH = os.environ.get("MODEL_RFR_PATH")
MODEL_SS_PATH = os.environ.get("MODEL_SS_PATH")
MODEL_SS_PRETRAINED_PATH = os.environ.get("MODEL_SS_PRETRAINED_PATH")

MODEL_LLM_PATH = os.environ.get("MODEL_LLM_PATH")
MODEL_LLM_CONTEXT_LEN = int(os.environ.get("MODEL_LLM_CONTEXT_LEN", "4096"))
MODEL_LLM_DYNAMO_HISTORY_TABLE = os.environ.get("MODEL_LLM_DYNAMO_HISTORY_TABLE")