import cv2
import os
import numpy as np
import pandas as pd

from server.config import FRAME_ATTIRE_MASK, FRAME_BATCH, FRAME_INTERVAL
from server.models.face_restorer import restoreFaces
from server.models.presenter_localizer import batchInfer, calculatePresenterXYXY, localizePresenter
from server.models.xdensenet import batchInferAttire, batchInferMultitask
from server.utils.common import DictKeyArr, batchGen
from server.utils.cv import OG_HEIGHT, OG_WIDTH, resizeWithPad


def extractFrameBatchLs(input_file: str):
  cap = cv2.VideoCapture(input_file)
  batch_ls = []
  current_batch = []
  i = -1
  l = 0
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    i += 1
    if i % FRAME_INTERVAL == 0:
      frame = resizeWithPad(frame, OG_WIDTH, OG_HEIGHT)
      current_batch.append((i, frame))
      l += 1
      if l == FRAME_BATCH:
        batch_ls.append(current_batch)
        current_batch = []
  if l > 0:
    batch_ls.append(current_batch)
  cap.release()
  return batch_ls


def localizeFrames(temp_mkv_path, temp_localized_dir, temp_xyxyn_path):
  xyxy_df = {key: [] for key in ["i", "x1", "y1", "x2", "y2"]}
  for batch in extractFrameBatchLs(temp_mkv_path):
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


def restoreAndBatchFrames(temp_localized_dir, temp_restored_dir):
  return batchGen(restoreFaces(temp_localized_dir, temp_restored_dir), FRAME_BATCH)


multitask_key_ls = ["moving", "smiling", "upright", "ec"]
attire_key_ls = ["pa"]


def predictFrames(restored_batch_gen, multitask_path: str, attire_path: str):
  multitask_df = DictKeyArr(multitask_key_ls)
  attire_df_dict = DictKeyArr(attire_key_ls)
  for i_batch, frame_batch in restored_batch_gen:
    multitask_pred = batchInferMultitask(frame_batch)
    multitask_df["i"].extend(i_batch)
    for j, key in enumerate(multitask_key_ls):
      multitask_df[key].extend(multitask_pred[:, j])
    for i_i, i in enumerate(i_batch):
      if i_i in FRAME_ATTIRE_MASK:
        attire_df_dict["i"].append(i)
        attire_df_dict["pa"].append(frame_batch[i_i])
  multitask_df = pd.DataFrame(multitask_df).set_index("i")
  multitask_df.to_csv(multitask_path)

  attire_pred = batchInferAttire(attire_df_dict["pa"])
  attire_df_dict["pa"] = attire_pred
  attire_df = pd.DataFrame(attire_df_dict).set_index("i")
  attire_df.to_csv(attire_path)

  for key in multitask_key_ls:
    yield key, multitask_df[key].mean()
  yield "pa", bool(attire_df["pa"].mode()[0])
