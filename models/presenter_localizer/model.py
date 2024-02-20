import os
from ultralytics import YOLO

from server.config import MODELS_DIR

wd = os.path.join(MODELS_DIR, "presenter_localizer")
model_path =  os.path.join(wd, "model.pt")
model = YOLO(model_path, task="detect")
