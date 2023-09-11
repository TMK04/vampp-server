import cv2
import numpy as np
import os

FRAME_ATTIRE_MASK = os.environ.get("FRAME_ATTIRE_MASK",
                                   "5,10,15,20,25,-26,-21,-16,-11,-6").split(",")
FRAME_BATCH = int(os.environ.get("FRAME_BATCH", "1"))
FRAME_INTERVAL = int(os.environ.get("FRAME_INTERVAL", "1"))
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", "0"))

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
  skip = FRAME_SKIP
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
    if i % FRAME_INTERVAL == 0:
      frame = resizeWithPad(frame, OG_WIDTH, OG_HEIGHT)
      current_batch.append((i, frame))
      if len(current_batch) == FRAME_BATCH:
        yield current_batch
        current_batch = []
  # Yield the last batch (may be smaller than FRAME_BATCH)
  if len(current_batch) > 0:
    yield current_batch
  cap.release()


def grayscale(frame):
  return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def resizeToLocalize(frame):
  to_localize_frame = cv2.resize(frame, (TO_LOCALIZE_WIDTH, TO_LOCALIZE_HEIGHT))
  to_localize_frame = grayscale(to_localize_frame)
  return to_localize_frame


def processRestoredFrames(gen_restored_frames, attire_df_dict):
  n = 0
  multitask_df_dict = {key: [] for key in ["i", "frame"]}
  for j, (i, restored_frame) in enumerate(gen_restored_frames):
    if j in FRAME_ATTIRE_MASK:
      attire_frame = restored_frame[-134:]
      attire_df_dict["i"].append(i)
      attire_df_dict["attire"].append(attire_frame)
    multitask_df_dict["i"].append(i)
    multitask_df_dict["frame"].append(restored_frame)
    n += 1
    if n == FRAME_BATCH:
      yield dict_i_multitask
      multitask_df_dict = {key: [] for key in ["i", "frame"]}
      n = 0
  if n > 0:
    yield multitask_df_dict