import cv2

from server.utils.cv import OG_HEIGHT, OG_WIDTH, resizeWithPad
from .model import model

TO_LOCALIZE_WIDTH = 426
TO_LOCALIZE_HEIGHT = 240
LOCALIZED_HEIGHT = LOCALIZED_WIDTH = 224


def preprocess(frame):
  preprocessed_frame = frame
  # Resize
  preprocessed_frame = cv2.resize(preprocessed_frame, (TO_LOCALIZE_WIDTH, TO_LOCALIZE_HEIGHT))
  # Grayscale
  preprocessed_frame = cv2.cvtColor(cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2GRAY),
                                    cv2.COLOR_GRAY2BGR)
  return preprocessed_frame


def batchInfer(batch):
  preprocessed_batch = [preprocess(frame) for _, frame in batch]
  return model(preprocessed_batch)


def calculatePresenterXYXY(result):
  boxes = result.boxes
  conf_ls = boxes.conf
  xyxyn_ls = boxes.xyxyn

  best_conf = 0
  for k, conf in enumerate(conf_ls):
    if conf > best_conf:
      best_conf = conf
      best_k = k
  if best_conf < .5:
    return None

  xyxyn = xyxyn_ls[best_k]
  xyxy_dict = dict(
      x1=int(xyxyn[0] * OG_WIDTH),
      y1=int(xyxyn[1] * OG_HEIGHT),
      x2=int(xyxyn[2] * OG_WIDTH),
      y2=int(xyxyn[3] * OG_HEIGHT),
  )
  return xyxy_dict


def localizePresenter(frame, xyxy_dict):
  localized_frame = frame[xyxy_dict["y1"]:xyxy_dict["y2"], xyxy_dict["x1"]:xyxy_dict["x2"]]
  localized_frame = resizeWithPad(localized_frame, LOCALIZED_HEIGHT, LOCALIZED_WIDTH)
  return localized_frame
