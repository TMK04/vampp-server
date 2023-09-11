from dotenv import load_dotenv

load_dotenv()

from aws import AWS_S3_BUCKET, USE_AWS, s3_client
import cv2
from cv_helpers import FRAME_ATTIRE_MASK, FRAME_BATCH, extractFrames, processRestoredFrames, resizeToLocalize
from fastapi import FastAPI, UploadFile, Form
from ffmpeg_commands import compressVideo, extractAudio
from models.components import device, toTensor
from models.face_restorer import restoreFaces
from models.presenter_localizer import calculatePresenterXYXYN, localizePresenter
from models.xdensenet import attire_model, infer, multitask_model
import os
import numpy as np
import pandas as pd
from pathlib import Path
import re
from re_patterns import pattern_mp4_suffix
import shortuuid
import shutil
import soundfile as sf
import torch
from transcription import transcribe, splitAudio

app = FastAPI()
tmp_dir = Path("tmp/")
tmp_dir.mkdir(parents=True, exist_ok=True)


@app.get("/")
def read_root():
  return {"Hello": "World"}


@app.post("/")
async def receive_video(file: UploadFile = Form(...), topic: str = Form(...)):
  basename = re.sub(pattern_mp4_suffix, "", file.filename)

  # Create temp dir
  while True:
    try:
      basename_random = f"{basename}-{shortuuid.ShortUUID().random(length=7)}"
      temp_dir_name = os.path.join(tmp_dir, basename_random)
      Path(temp_dir_name).mkdir()
      break
    except FileExistsError:
      continue

  if USE_AWS:

    def s3Key(arg_ls):
      return os.path.join(basename_random, *arg_ls)

  def tempName(arg_ls):
    return os.path.join(temp_dir_name, "-".join(arg_ls))

  # Save file to temp
  temp_arg_ls = ["temp.mp4"]
  temp_file_name = tempName(temp_arg_ls)
  with open(temp_file_name, "wb") as f:
    f.write(await file.read())

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

  def localizeFrames():
    localized_frame_ls = []
    if USE_AWS:
      xyxyn_df = {key: [] for key in ["i", "x1", "y1", "x2", "y2"]}
    for batch in extractFrames(temp_mp4_name):
      to_localize_frame_batch = []
      for i, frame in batch:
        if USE_AWS:
          og_arg_ls = ["frame", "og", f"{i}.jpg"]
          temp_og_name = tempName(og_arg_ls)
          cv2.imwrite(temp_og_name, frame)
          og_key = s3Key(og_arg_ls)
          s3_client.upload_file(temp_og_name, AWS_S3_BUCKET, og_key)
          os.remove(temp_og_name)

        to_localize_frame = resizeToLocalize(frame)
        to_localize_frame_batch.append(to_localize_frame)

      gen_xyxyn = calculatePresenterXYXYN(to_localize_frame_batch)
      for j, xyxyn in enumerate(gen_xyxyn):
        i, frame = batch[j]
        localized_frame = localizePresenter(frame, xyxyn)

        i_jpg = f"{i}.jpg"
        temp_localized_name = os.path.join(temp_localized_dir_name, i_jpg)
        cv2.imwrite(temp_localized_name, localized_frame)
        localized_frame_ls.append(localized_frame)
        if USE_AWS:
          localized_key = s3Key([*localized_dir_arg_ls, i_jpg])
          s3_client.upload_file(temp_localized_name, AWS_S3_BUCKET, localized_key)

          xyxyn_df["i"].append(i)
          xyxyn_df["x1"].append(xyxyn[0])
          xyxyn_df["y1"].append(xyxyn[1])
          xyxyn_df["x2"].append(xyxyn[2])
          xyxyn_df["y2"].append(xyxyn[3])
    if USE_AWS:
      xyxyn_df = pd.DataFrame(xyxyn_df).set_index("i")
      xyxyn_arg_ls = ["frame", "xyxyn.csv"]
      temp_xyxyn_name = tempName(xyxyn_arg_ls)
      xyxyn_df.to_csv(temp_xyxyn_name)
      xyxyn_key = s3Key(xyxyn_arg_ls)
      s3_client.upload_file(temp_xyxyn_name, AWS_S3_BUCKET, xyxyn_key)
      os.remove(temp_xyxyn_name)
    return localized_frame_ls

  localized_frame_ls = localizeFrames()
  os.remove(temp_mp4_name)

  def restoreFrames():
    temp_restored_dir_name = tempName(["frame", "restored"])
    Path(temp_restored_dir_name).mkdir()
    restoreFaces(temp_localized_dir_name, temp_restored_dir_name)
    temp_restored_dir_name = os.path.join(temp_restored_dir_name, "restored_imgs")
    for temp_restored_basename in os.listdir(temp_restored_dir_name):
      temp_restored_name = os.path.join(temp_restored_dir_name, temp_restored_basename)
      restored_frame = cv2.imread(temp_restored_name, cv2.IMREAD_GRAYSCALE)
      restored_frame = np.expand_dims(restored_frame, axis=0)
      if USE_AWS:
        restored_key = s3Key(["frame", "restored", temp_restored_basename])
        s3_client.upload_file(temp_restored_name, AWS_S3_BUCKET, restored_key)
      os.remove(temp_restored_name)
      i = temp_restored_basename.replace(".jpg", "")
      yield i, restored_frame

  attire_df_dict = {key: [] for key in ["i", "attire"]}
  multitask_key_ls = ["moving", "smiling", "upright", "ec"]
  multitask_df_dict = {key: [] for key in ["i", *multitask_key_ls]}
  for batch_multitask_df_dict in processRestoredFrames(restoreFrames(), attire_df_dict):
    restored_frame_batch_tensor = toTensor(batch_multitask_df_dict["frame"]).to(device)
    multitask_pred = infer(multitask_model, restored_frame_batch_tensor)
    multitask_df_dict["i"].extend(batch_multitask_df_dict["i"])
    for j, key in enumerate(multitask_key_ls):
      multitask_df_dict[key].extend(multitask_pred[:, j].tolist())
  multitask_df = pd.DataFrame(multitask_df_dict).set_index("i")
  print(multitask_df.head())
  attire_frame_tensor = toTensor(attire_df_dict["attire"]).to(device)
  attire_df_dict["attire"] = infer(attire_model, restored_frame_batch_tensor)
  attire_df = pd.DataFrame(attire_df_dict).set_index("i")
  print(attire_df.head())

  for i, window in enumerate(splitAudio(temp_wav_name)):
    i_file = f"{i}.mp3"
    window_arg_ls = ["audio", "window", i_file]
    temp_window_name = tempName(window_arg_ls)
    sf.write(temp_window_name, window, 16000)
    window_key = s3Key(window_arg_ls)
    s3_client.upload_file(temp_window_name, AWS_S3_BUCKET, window_key)
    os.remove(temp_window_name)

  def transcribeAudio():
    text_arg_ls = ["text.txt"]
    temp_text_name = tempName(text_arg_ls)
    text = transcribe(temp_wav_name)
    with open(temp_text_name, "w") as f:
      f.write(text)
    if USE_AWS:
      text_key = s3Key(text_arg_ls)
      s3_client.upload_file(temp_text_name, AWS_S3_BUCKET, text_key)
    os.remove(temp_text_name)

  transcribeAudio()

  shutil.rmtree(temp_dir_name, ignore_errors=True)
  return "ok"