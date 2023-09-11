from cv_helpers import OG_HEIGHT, OG_WIDTH, resizeWithPad
import numpy as np
import os
from ultralytics import YOLO

MODEL_PL_PATH = os.environ.get("MODEL_PL_PATH")
if MODEL_PL_PATH is None:
  raise Exception("MODEL_PL_PATH is not set")
model = YOLO(MODEL_PL_PATH, task="detect")

LOCALIZED_HEIGHT = LOCALIZED_WIDTH = 224


def calculatePresenterXYXYN(to_localize_batch):
  # replicate channel dimension x3
  src = np.array(to_localize_batch).repeat(3, -1)
  result_stream = model(src, stream=True)

  for result in result_stream:
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

    xyxyn = xyxyn_ls[best_j]
    yield [obj.item() for obj in xyxyn]


def localizePresenter(frame, xyxyn):
  x1 = np.uint16(xyxyn[0] * 1280)
  x2 = np.uint16(xyxyn[2] * 1280)
  y1 = np.uint16(xyxyn[1] * 720)
  y2 = np.uint16(xyxyn[3] * 720)
  localized_frame = frame[y1:y2, x1:x2]
  localized_frame = resizeWithPad(localized_frame, LOCALIZED_HEIGHT, LOCALIZED_WIDTH)
  return localized_frame