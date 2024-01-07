import cv2
import os
import numpy as np
import pandas as pd

from server.config import FRAME_ATTIRE_MASK, FRAME_BATCH, FRAME_INTERVAL
from server.models.face_restorer import restoreFace
from server.models.presenter_localizer import batchInfer, calculatePresenterXYXY, localizePresenter
from server.models.xdensenet import batchInferAttire, batchInferMultitask
from server.utils.common import DictKeyArr, batchGen, toCsv
from server.utils.cv import OG_HEIGHT, OG_WIDTH, resizeWithPad


def readFrames(input_file: str):
  cap = cv2.VideoCapture(input_file)
  try:
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
          yield current_batch
          current_batch = []
    if l > 0:
      yield current_batch
  finally:
    cap.release()


def localizeFrames(readFrames_gen, temp_xyxyn_path: str):
  xyxy_df = pd.DataFrame({key: [] for key in ["x1", "y1", "x2", "y2"]})
  try:
    for batch in readFrames_gen:
      for inferred_i, result in enumerate(batchInfer(batch)):
        xyxy_dict = calculatePresenterXYXY(result)
        if xyxy_dict is None:
          continue
        i, frame = batch[inferred_i]
        localized_frame = localizePresenter(frame, xyxy_dict)
        xyxy_df.loc[i] = xyxy_dict
        yield i, localized_frame
  finally:
    xyxy_df.to_csv(temp_xyxyn_path)


def restoreFrames(localizeFrames_gen: str, temp_restored_dir: str):
  for i, frame in localizeFrames_gen:
    yield i, restoreFace(i, frame, temp_restored_dir)


def restoreAndBatchFrames(localizeFrames_gen, temp_restored_dir: str):
  restored_batch_gen = batchGen(restoreFrames(localizeFrames_gen, temp_restored_dir), FRAME_BATCH)
  return restored_batch_gen


multitask_key_ls = ["moving", "smiling", "upright", "ec"]
attire_key_ls = ["pa"]


def predictFrames(restored_batch_gen, multitask_path: str, attire_path: str):
  multitask_dict = DictKeyArr(multitask_key_ls)
  attire_dict = DictKeyArr(attire_key_ls)
  for i_batch, frame_batch in restored_batch_gen:
    multitask_pred = batchInferMultitask(frame_batch)
    multitask_dict["i"].extend(i_batch)
    for key_i, key in enumerate(multitask_key_ls):
      multitask_dict[key].extend(multitask_pred[:, key_i])
    for i_i, i in enumerate(i_batch):
      if i_i in FRAME_ATTIRE_MASK:
        attire_dict["i"].append(i)
        attire_dict["pa"].append(frame_batch[i_i])
  multitask_df = pd.DataFrame(multitask_dict)
  toCsv(multitask_df, multitask_path)

  attire_dict["pa"] = batchInferAttire(attire_dict["pa"])
  attire_df = pd.DataFrame(attire_dict)
  toCsv(attire_df, attire_path)

  for key in multitask_key_ls:
    yield key, multitask_df[key].mean()
  yield "pa", bool(attire_df["pa"].mode()[0])
