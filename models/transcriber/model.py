import os
import nemo.collections.asr as nemo_asr
from pathlib import Path

from server.config import MODEL_TRANSCRIBER_DIR

models_dir = Path(__file__).parent / "./models/"
model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
    model_name=os.path.join(models_dir, MODEL_TRANSCRIBER_DIR))
