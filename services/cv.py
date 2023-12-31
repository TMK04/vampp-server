import cv2
import os
import pandas as pd

from server.config import FRAME_BATCH, FRAME_INTERVAL
from server.models.presenter_localizer import batchInfer, calculatePresenterXYXY, localizePresenter

OG_WIDTH = 1280
OG_HEIGHT = 720


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


def extractFrameBatchLs(input_file: str):
  cap = cv2.VideoCapture(input_file)
  batch_ls = []
  current_batch = []
  i = -1
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    i += 1
    if i % FRAME_INTERVAL == 0:
      frame = resizeWithPad(frame, OG_WIDTH, OG_HEIGHT)
      current_batch.append((i, frame))
      if len(current_batch) == FRAME_BATCH:
        batch_ls.append(current_batch)
        current_batch = []
  if len(current_batch) > 0:
    batch_ls.append(current_batch)
  cap.release()
  return batch_ls


def localizeFrames(temp_mp4_path, temp_localized_dir, temp_xyxyn_path):
  xyxy_df = {key: [] for key in ["i", "x1", "y1", "x2", "y2"]}
  for batch in extractFrameBatchLs(temp_mp4_path):
    for inferred_i, result in enumerate(batchInfer(batch)):
      xyxy_dict = calculatePresenterXYXY(result)
      if xyxy_dict is None:
        continue
      i, frame = batch[inferred_i]
      localized_frame = localizePresenter(frame, xyxy_dict)

      i_jpg = f"{i}.jpg"
      temp_localized_name = os.path.join(temp_localized_dir, i_jpg)
      cv2.imwrite(temp_localized_name, localized_frame)
      xyxy_df["i"].append(i)
      for key in xyxy_dict:
        xyxy_df[key].append(xyxy_dict[key])
  xyxy_df = pd.DataFrame(xyxy_df).set_index("i")
  xyxy_df.to_csv(temp_xyxyn_path)


def batchRestoredFrames(gen_restored_frames):
  i_ls = []
  frame_ls = []
  for i, restored_frame in gen_restored_frames:
    i_ls.append(i)
    frame_ls.append(restored_frame)
    if len(i_ls) == FRAME_BATCH:
      yield i_ls, frame_ls
      i_ls = []
      frame_ls = []
  if len(i_ls) > 0:
    yield i_ls, frame_ls
