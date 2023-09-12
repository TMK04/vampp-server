import config

from typing import Union

from audio import transcribe, splitAudio, splitAudioBatch
from aws import AWS_DYNAMO_TABLE, AWS_S3_BUCKET, USE_AWS, dynamo_client, s3_client
import cv2
from config import FRAME_ATTIRE_MASK
from cv_helpers import extractFrames, processRestoredFrames, resizeToLocalize
from fastapi import FastAPI, HTTPException, UploadFile, Form
from models.components import device, infer, toTensor
from models.face_restorer import restoreFaces
from models.llm import runBeholderFirst
from models.presenter_localizer import calculatePresenterXYXYN, localizePresenter
from models.speech_stats import preprocess, speech_stats_model
from models.xdensenet import attire_model, multitask_model
import os
import numpy as np
import pandas as pd
from pathlib import Path
import re
from re_patterns import pattern_mp4_suffix
import shortuuid
import shutil
import soundfile as sf
from subprocess import CalledProcessError
import torch
from video_commands import compressVideo, downloadVideo, extractAudio

app = FastAPI()
tmp_dir = Path("tmp/")
tmp_dir.mkdir(parents=True, exist_ok=True)


@app.get("/")
def read_root():
  return {"Hello": "World"}


@app.post("/")
async def receive_video(topic: str = Form(...), file: Union[UploadFile, str] = Form(...)):
  file_is_ytid = isinstance(file, str)
  try:
    file_name = file if file_is_ytid else file.filename
  # No filename
  except AttributeError:
    raise HTTPException(status_code=400, detail="No file name")
  basename = re.sub(pattern_mp4_suffix, "", file_name)

  # Create temp dir
  while True:
    try:
      basename_random = f"{basename}-{shortuuid.ShortUUID().random(length=7)}"
      temp_dir_name = os.path.join(tmp_dir, basename_random)
      Path(temp_dir_name).mkdir()
      break
    except FileExistsError:
      continue

  Item = {"id": {"S": basename_random}, "topic": {"S": topic}}

  if USE_AWS:

    def s3Key(arg_ls):
      return os.path.join(basename_random, *arg_ls)

  def tempName(arg_ls):
    return os.path.join(temp_dir_name, "-".join(arg_ls))

  # Save file to temp
  temp_arg_ls = ["temp.mp4"]
  temp_file_name = tempName(temp_arg_ls)
  if file_is_ytid:
    try:
      downloadVideo(file, temp_file_name)
    except CalledProcessError:
      raise HTTPException(status_code=400, detail="Invalid YouTube ID")
  else:
    with open(temp_file_name, "wb") as f:
      f.write(await file.read())
  print(os.listdir(temp_dir_name))

  mp4_arg_ls = ["og.mp4"]
  temp_mp4_name = tempName(mp4_arg_ls)
  compressVideo(temp_file_name, temp_mp4_name)
  os.remove(temp_file_name)
  if USE_AWS:
    video_key = s3Key(mp4_arg_ls)
    s3_client.upload_file(temp_mp4_name, AWS_S3_BUCKET, video_key)

  wav_arg_ls = ["audio", "og.wav"]
  temp_wav_name = tempName(wav_arg_ls)
  extractAudio(temp_mp4_name, temp_wav_name)
  if USE_AWS:
    audio_key = s3Key(wav_arg_ls)
    s3_client.upload_file(temp_wav_name, AWS_S3_BUCKET, audio_key)

  localized_dir_arg_ls = ["frame", "localized"]
  temp_localized_dir_name = tempName(localized_dir_arg_ls)
  Path(temp_localized_dir_name).mkdir()

  # def localizeFrames():
  #   localized_frame_ls = []
  #   if USE_AWS:
  #     xyxyn_df = {key: [] for key in ["i", "x1", "y1", "x2", "y2"]}
  #   for batch in extractFrames(temp_mp4_name):
  #     to_localize_frame_batch = []
  #     for i, frame in batch:
  #       if USE_AWS:
  #         og_arg_ls = ["frame", "og", f"{i}.jpg"]
  #         temp_og_name = tempName(og_arg_ls)
  #         cv2.imwrite(temp_og_name, frame)
  #         og_key = s3Key(og_arg_ls)
  #         s3_client.upload_file(temp_og_name, AWS_S3_BUCKET, og_key)
  #         os.remove(temp_og_name)

  #       to_localize_frame = resizeToLocalize(frame)
  #       to_localize_frame_batch.append(to_localize_frame)

  #     gen_xyxyn = calculatePresenterXYXYN(to_localize_frame_batch)
  #     for j, xyxyn in enumerate(gen_xyxyn):
  #       i, frame = batch[j]
  #       localized_frame = localizePresenter(frame, xyxyn)

  #       i_jpg = f"{i}.jpg"
  #       temp_localized_name = os.path.join(temp_localized_dir_name, i_jpg)
  #       cv2.imwrite(temp_localized_name, localized_frame)
  #       localized_frame_ls.append(localized_frame)
  #       if USE_AWS:
  #         localized_key = s3Key([*localized_dir_arg_ls, i_jpg])
  #         s3_client.upload_file(temp_localized_name, AWS_S3_BUCKET, localized_key)

  #         xyxyn_df["i"].append(i)
  #         xyxyn_df["x1"].append(xyxyn[0])
  #         xyxyn_df["y1"].append(xyxyn[1])
  #         xyxyn_df["x2"].append(xyxyn[2])
  #         xyxyn_df["y2"].append(xyxyn[3])
  #   if USE_AWS:
  #     xyxyn_df = pd.DataFrame(xyxyn_df).set_index("i")
  #     xyxyn_arg_ls = ["frame", "xyxyn.csv"]
  #     temp_xyxyn_name = tempName(xyxyn_arg_ls)
  #     xyxyn_df.to_csv(temp_xyxyn_name)
  #     xyxyn_key = s3Key(xyxyn_arg_ls)
  #     s3_client.upload_file(temp_xyxyn_name, AWS_S3_BUCKET, xyxyn_key)
  #     os.remove(temp_xyxyn_name)
  #   return localized_frame_ls

  # localized_frame_ls = localizeFrames()
  # os.remove(temp_mp4_name)

  # def restoreFrames():
  #   temp_restored_dir_name = tempName(["frame", "restored"])
  #   Path(temp_restored_dir_name).mkdir()
  #   restoreFaces(temp_localized_dir_name, temp_restored_dir_name)
  #   temp_restored_dir_name = os.path.join(temp_restored_dir_name, "restored_imgs")
  #   for temp_restored_basename in os.listdir(temp_restored_dir_name):
  #     temp_restored_name = os.path.join(temp_restored_dir_name, temp_restored_basename)
  #     restored_frame = cv2.imread(temp_restored_name, cv2.IMREAD_GRAYSCALE)
  #     restored_frame = np.expand_dims(restored_frame, axis=0)
  #     if USE_AWS:
  #       restored_key = s3Key(["frame", "restored", temp_restored_basename])
  #       s3_client.upload_file(temp_restored_name, AWS_S3_BUCKET, restored_key)
  #     os.remove(temp_restored_name)
  #     i = temp_restored_basename.replace(".jpg", "")
  #     yield i, restored_frame

  # multitask_key_ls = ["moving", "smiling", "upright", "ec"]
  # multitask_df_dict = {key: [] for key in ["i", *multitask_key_ls]}
  # frame_ls = []
  # for batch_i_ls, batch_frame_ls in processRestoredFrames(restoreFrames()):
  #   batch_frame_tensor = toTensor(batch_frame_ls).to(device)
  #   multitask_pred = infer(multitask_model, batch_frame_tensor)
  #   multitask_df_dict["i"].extend(batch_i_ls)
  #   frame_ls.extend(batch_frame_ls)
  #   for j, key in enumerate(multitask_key_ls):
  #     multitask_df_dict[key].extend(multitask_pred[:, j])
  # multitask_df = pd.DataFrame(multitask_df_dict).set_index("i")
  # if USE_AWS:
  #   multitask_arg_ls = ["frame", "multitask.csv"]
  #   temp_multitask_name = tempName(multitask_arg_ls)
  #   multitask_df.to_csv(temp_multitask_name)
  #   multitask_key = s3Key(multitask_arg_ls)
  #   s3_client.upload_file(temp_multitask_name, AWS_S3_BUCKET, multitask_key)
  #   os.remove(temp_multitask_name)
  # for key in multitask_key_ls:
  #   Item[key] = {"N": str(multitask_df[key].mean())}

  # attire_df_dict = {
  #     "i": multitask_df.index[FRAME_ATTIRE_MASK],
  #     "attire": np.array(frame_ls)[FRAME_ATTIRE_MASK]
  # }
  # attire_frame_tensor = toTensor(attire_df_dict["attire"]).to(device)
  # attire_pred = infer(attire_model, attire_frame_tensor)
  # attire_df_dict["attire"] = attire_pred[:, 0]
  # attire_df = pd.DataFrame(attire_df_dict).set_index("i")
  # if USE_AWS:
  #   attire_arg_ls = ["frame", "attire.csv"]
  #   temp_attire_name = tempName(attire_arg_ls)
  #   attire_df.to_csv(temp_attire_name)
  #   attire_key = s3Key(attire_arg_ls)
  #   s3_client.upload_file(temp_attire_name, AWS_S3_BUCKET, attire_key)
  #   os.remove(temp_attire_name)
  # attire_mode = bool(attire_df["attire"].mode()[0])
  # Item["professional_attire"] = {"BOOL": attire_mode}

  # speech_stats_key_ls = ["enthusiasm", "clarity"]
  # speech_stats_df_dict = {key: [] for key in ["i", *speech_stats_key_ls]}
  # for batch_i, batch_window in splitAudioBatch(splitAudio(temp_wav_name)):
  #   batch_window_tensor = toTensor(batch_window)
  #   speech_stats = infer(speech_stats_model, batch_window_tensor)
  #   speech_stats_df_dict["i"].extend(batch_i)
  #   for j, key in enumerate(speech_stats_key_ls):
  #     speech_stats_df_dict[key].extend(speech_stats[:, j])
  # speech_stats_df = pd.DataFrame(speech_stats_df_dict).set_index("i")
  # if USE_AWS:
  #   speech_stats_arg_ls = ["audio", "stats.csv"]
  #   temp_speech_stats_name = tempName(speech_stats_arg_ls)
  #   speech_stats_df.to_csv(temp_speech_stats_name)
  #   speech_stats_key = s3Key(speech_stats_arg_ls)
  #   s3_client.upload_file(temp_speech_stats_name, AWS_S3_BUCKET, speech_stats_key)
  #   os.remove(temp_speech_stats_name)
  # for key in speech_stats_key_ls:
  #   Item[key] = {"N": str(speech_stats_df[key].mean())}

  def transcribeAudio():
    pitch_arg_ls = ["pitch.txt"]
    temp_pitch_name = tempName(pitch_arg_ls)
    pitch = transcribe(temp_wav_name)
    return pitch

  pitch = transcribeAudio()
  Item["pitch"] = {"S": pitch}
  beholder_response = runBeholderFirst(topic, pitch)
  for key, value in beholder_response:
    if key.endswith("_justification"):
      Item[key] = {"S": value}
    else:
      Item[key] = {"N": str(value)}

  print(Item)
  if USE_AWS:
    dynamo_client.put_item(TableName=AWS_DYNAMO_TABLE, Item=Item)

  shutil.rmtree(temp_dir_name, ignore_errors=True)
  return Item