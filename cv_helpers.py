import cv2
import numpy as np
import os

OG_WIDTH = 1280
OG_HEIGHT = 720
TO_LOCALIZE_WIDTH = 426
TO_LOCALIZE_HEIGHT = 240


def resizeWithPad(image, target_width: int, target_height: int, print_diff=False):
  height, width, _ = image.shape
  if print_diff:
    if width == target_width and height == target_height:
      return image
    else:
      print("Resizing...")
  ratio = min(target_width / width, target_height / height)
  new_width = int(width * ratio)
  new_height = int(height * ratio)
  resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
  delta_w = target_width - new_width
  delta_h = target_height - new_height
  top, bottom = delta_h // 2, delta_h - (delta_h // 2)
  left, right = delta_w // 2, delta_w - (delta_w // 2)
  color = [0, 0, 0]
  return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def extractFrames(input_file: str):
  cap = cv2.VideoCapture(input_file)
  skip = int(os.environ.get("FRAME_SKIP", "0"))
  interval = int(os.environ.get("FRAME_INTERVAL", "1"))
  batch_size = int(os.environ.get("FRAME_BATCH", "1"))
  current_batch = []
  i = -1
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    i += 1
    if skip:
      skip -= 1
      continue
    if i % interval == 0:
      frame = resizeWithPad(frame, OG_WIDTH, OG_HEIGHT)
      current_batch.append((i, frame))
      if len(current_batch) == batch_size:
        yield current_batch
        current_batch = []
  # Yield the last batch (may be smaller than batch_size)
  if len(current_batch) > 0:
    yield current_batch
  cap.release()


def resizeToLocalize(frame):
  to_localize_frame = cv2.resize(frame, (TO_LOCALIZE_WIDTH, TO_LOCALIZE_HEIGHT))
  to_localize_frame = cv2.cvtColor(cv2.cvtColor(to_localize_frame, cv2.COLOR_BGR2GRAY),
                                   cv2.COLOR_GRAY2BGR)
  return to_localize_frame