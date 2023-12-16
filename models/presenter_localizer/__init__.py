from cv_helpers import OG_HEIGHT, OG_WIDTH, TO_LOCALIZE_HEIGHT, TO_LOCALIZE_WIDTH, resizeWithPad
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO

model_pl_path = Path(__file__).parent / "model.pt"
model = YOLO(model_pl_path, task="detect")

LOCALIZED_HEIGHT = LOCALIZED_WIDTH = 224


def calculatePresenterXYXYN(to_localize_batch):
  results = model(to_localize_batch)

  for result in results:
    boxes = result.boxes
    conf_ls = boxes.conf
    xyxyn_ls = boxes.xyxyn

    best_conf = 0
    for k, conf in enumerate(conf_ls):
      if conf > best_conf:
        best_conf = conf
        best_k = k
    if best_conf < .5:
      continue

    xyxyn = xyxyn_ls[best_k]
    yield [obj.item() for obj in xyxyn]


def localizePresenter(frame, xyxyn):
  x1 = int(xyxyn[0] * OG_WIDTH)
  x2 = int(xyxyn[2] * OG_WIDTH)
  y1 = int(xyxyn[1] * OG_HEIGHT)
  y2 = int(xyxyn[3] * OG_HEIGHT)
  localized_frame = frame[y1:y2, x1:x2]
  localized_frame = resizeWithPad(localized_frame, LOCALIZED_HEIGHT, LOCALIZED_WIDTH)
  return localized_frame
