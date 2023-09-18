import config

from typing import Union

from audio import transcribe, splitAudio, splitAudioBatch
from aws import AWS_DYNAMO_TABLE, AWS_S3_BUCKET, USE_AWS, dynamo_client, s3_client
import concurrent.futures
from config import FRAME_ATTIRE_MASK, TMP_DIR, TMP_FILENAME
from cv_helpers import batchRestoredFrames, extractFrames, resizeToLocalize
import cv2
from fastapi import FastAPI, HTTPException, UploadFile, Form
from models.components import device, infer, toTensor
from models.face_restorer import restoreFaces
from models.llm import Chain, runBeholderFirst, summarize
from models.presenter_localizer import calculatePresenterXYXYN, localizePresenter
from models.rfr import rfr_bv, rfr_clarity, rfr_pe, rfrInfer
from models.speech_stats import preprocess, speech_stats_model
from models.xdensenet import attire_model, multitask_model
import os
import numpy as np
import pandas as pd
from pathlib import Path
import re
import shutil
from video_commands import compressVideo, extractAudio

app = FastAPI()
tmp_dir = Path("tmp/")
tmp_dir.mkdir(parents=True, exist_ok=True)


@app.post("/")
async def receive_video(topic: str = Form(""), basename: str = Form(...), random: str = Form(...)):
  basename_random = f"{basename}-{random}"
  temp_dir_name = os.path.join(TMP_DIR, basename_random)

  Item = {"id": {"S": basename_random}, "topic": {"S": topic}}

  def setItem(key, type_code, value):
    Item[key] = {type_code: value}
    print(f"{key}: {value}")

  if USE_AWS:

    def s3Key(arg_ls):
      return os.path.join(basename_random, *arg_ls)

  def tempName(arg_ls):
    return os.path.join(temp_dir_name, "-".join(arg_ls))

  # Save file to temp
  temp_arg_ls = [f"{TMP_FILENAME}.mp4"]
  temp_file_name = tempName(temp_arg_ls)

  mp4_arg_ls = ["og.mp4"]
  temp_mp4_name = tempName(mp4_arg_ls)
  compressVideo(temp_file_name, temp_mp4_name)
  os.remove(temp_file_name)
  # if USE_AWS:
  #   video_key = s3Key(mp4_arg_ls)
  #   s3_client.upload_file(temp_mp4_name, AWS_S3_BUCKET, video_key)

  wav_arg_ls = ["audio", "og.wav"]
  temp_wav_name = tempName(wav_arg_ls)
  extractAudio(temp_mp4_name, temp_wav_name)

  if USE_AWS:
    audio_key = s3Key(wav_arg_ls)
    s3_client.upload_file(temp_wav_name, AWS_S3_BUCKET, audio_key)

  def localizeFrames(temp_localized_dir_name):
    if USE_AWS:
      xyxyn_df = {key: [] for key in ["i", "x1", "y1", "x2", "y2"]}
    for batch in extractFrames(temp_mp4_name):
      to_localize_frame_batch = []
      for i, frame in batch:
        to_localize_frame = resizeToLocalize(frame)
        to_localize_frame_batch.append(to_localize_frame)

      gen_xyxyn = calculatePresenterXYXYN(to_localize_frame_batch)
      for j, xyxyn in enumerate(gen_xyxyn):
        i, frame = batch[j]
        localized_frame = localizePresenter(frame, xyxyn)

        i_jpg = f"{i}.jpg"
        temp_localized_name = os.path.join(temp_localized_dir_name, i_jpg)
        cv2.imwrite(temp_localized_name, localized_frame)
        if USE_AWS:
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

  def restoreFrames(temp_localized_dir_name):
    temp_restored_dir_name = tempName(["frame", "restored"])
    Path(temp_restored_dir_name).mkdir()
    restoreFaces(temp_localized_dir_name, temp_restored_dir_name)
    temp_restored_dir_name = os.path.join(temp_restored_dir_name, "restored_imgs")
    temp_restored_basename_ls = os.listdir(temp_restored_dir_name)
    for temp_restored_basename in temp_restored_basename_ls:
      temp_restored_name = os.path.join(temp_restored_dir_name, temp_restored_basename)
      restored_frame = cv2.imread(temp_restored_name, cv2.IMREAD_GRAYSCALE)
      restored_frame = np.expand_dims(restored_frame, axis=0)
      if USE_AWS:
        restored_key = s3Key(["frame", "restored", temp_restored_basename])
        s3_client.upload_file(temp_restored_name, AWS_S3_BUCKET, restored_key)
      os.remove(temp_restored_name)
      i = temp_restored_basename.replace(".jpg", "")
      yield i, restored_frame

  def predictFrames(gen_restored_frame_batches):
    multitask_key_ls = ["moving", "smiling", "upright", "ec"]
    multitask_df_dict = {key: [] for key in ["i", *multitask_key_ls]}
    frame_ls = []
    for batch_i_ls, batch_frame_ls in gen_restored_frame_batches:
      batch_frame_tensor = toTensor(batch_frame_ls).to(device)
      multitask_pred = infer(multitask_model, batch_frame_tensor)
      multitask_df_dict["i"].extend(batch_i_ls)
      frame_ls.extend(batch_frame_ls)
      for j, key in enumerate(multitask_key_ls):
        multitask_df_dict[key].extend(multitask_pred[:, j])
    multitask_df = pd.DataFrame(multitask_df_dict).set_index("i")

    if USE_AWS:
      multitask_arg_ls = ["frame", "multitask.csv"]
      temp_multitask_name = tempName(multitask_arg_ls)
      multitask_df.to_csv(temp_multitask_name)
      multitask_key = s3Key(multitask_arg_ls)
      s3_client.upload_file(temp_multitask_name, AWS_S3_BUCKET, multitask_key)
      os.remove(temp_multitask_name)
    for key in multitask_key_ls:
      value = str(multitask_df[key].mean())
      setItem(key, "N", value)

    attire_df_dict = {
        "i": multitask_df.index[FRAME_ATTIRE_MASK],
        "attire": np.array(frame_ls)[FRAME_ATTIRE_MASK]
    }
    attire_frame_tensor = toTensor(attire_df_dict["attire"]).to(device)
    attire_pred = infer(attire_model, attire_frame_tensor)
    attire_df_dict["attire"] = attire_pred[:, 0]
    attire_df = pd.DataFrame(attire_df_dict).set_index("i")
    if USE_AWS:
      attire_arg_ls = ["frame", "attire.csv"]
      temp_attire_name = tempName(attire_arg_ls)
      attire_df.to_csv(temp_attire_name)
      attire_key = s3Key(attire_arg_ls)
      s3_client.upload_file(temp_attire_name, AWS_S3_BUCKET, attire_key)
      os.remove(temp_attire_name)
    attire_mode = bool(attire_df["attire"].mode()[0])
    setItem("pa", "BOOL", attire_mode)

  def framesFn():
    localized_dir_arg_ls = ["frame", "localized"]
    temp_localized_dir_name = tempName(localized_dir_arg_ls)
    Path(temp_localized_dir_name).mkdir()

    localizeFrames(temp_localized_dir_name)
    os.remove(temp_mp4_name)
    predictFrames(batchRestoredFrames(restoreFrames(temp_localized_dir_name)))

  def predictSpeechStats():
    speech_stats_key_ls = ["enthusiasm", "clarity"]
    speech_stats_df_dict = {key: [] for key in ["i", *speech_stats_key_ls]}
    for batch_i, batch_window in splitAudioBatch(splitAudio(temp_wav_name)):
      batch_window_tensor = toTensor(batch_window)
      speech_stats = infer(speech_stats_model, batch_window_tensor)
      speech_stats_df_dict["i"].extend(batch_i)
      for j, key in enumerate(speech_stats_key_ls):
        speech_stats_df_dict[key].extend(speech_stats[:, j])
    speech_stats_df = pd.DataFrame(speech_stats_df_dict).set_index("i")
    if USE_AWS:
      speech_stats_arg_ls = ["audio", "stats.csv"]
      temp_speech_stats_name = tempName(speech_stats_arg_ls)
      speech_stats_df.to_csv(temp_speech_stats_name)
      speech_stats_key = s3Key(speech_stats_arg_ls)
      s3_client.upload_file(temp_speech_stats_name, AWS_S3_BUCKET, speech_stats_key)
      os.remove(temp_speech_stats_name)
    for key in speech_stats_key_ls:
      Item_key = f"speech_{key}"
      value = str(speech_stats_df[key].mean())
      setItem(Item_key, "N", value)

  def predictPitch():
    pitch_arg_ls = ["pitch.txt"]
    temp_pitch_name = tempName(pitch_arg_ls)
    pitch = transcribe(temp_wav_name)
    Item["pitch"] = {"S": pitch}

    topic, summary = summarize(pitch, topic)
    Item["topic"] = {"S": topic}
    Item["summary"] = {"S": summary}

    chain = Chain(basename_random)
    try:
      beholder_response = runBeholderFirst(chain, topic, pitch)
    except ValueError as e:
      raise HTTPException(status_code=500, detail=str(e))
    for key, value in beholder_response:
      Item_key = f"beholder_{key}"
      if key.endswith("_justification"):
        setItem(Item_key, "S", value)
      else:
        setItem(Item_key, "N", str(value))

  # Create a ThreadPoolExecutor to run the functions in parallel
  with concurrent.futures.ThreadPoolExecutor() as executor:
    concurrent.futures.wait([
        executor.submit(fn) for fn in [
            # framesFn,
            # predictSpeechStats,
            predictPitch,
        ]
    ])
  print(Item)

  # X_pe = [
  #     *[Item[key]["N"] for key in ["moving", "smiling", "upright", "ec"]],
  #     Item["pa"]["BOOL"],
  #     Item["speech_enthusiasm"]["N"],
  # ]
  # Item["pe"] = {"N": str(rfrInfer(rfr_pe, X_pe))}
  # X_clarity = [Item[key]["N"] for key in ["speech_clarity", "beholder_clarity"]]
  # Item["clarity"] = {"N": str(rfrInfer(rfr_clarity, X_clarity))}
  # X_bv = [
  #     Item[key]["N"] for key in ["beholder_creativity", "beholder_feasibility", "beholder_impact"]
  # ]
  # Item["bv"] = {"N": str(rfrInfer(rfr_bv, X_bv))}

  # print(Item)
  # if USE_AWS:
  #   dynamo_client.put_item(TableName=AWS_DYNAMO_TABLE, Item=Item)

  shutil.rmtree(temp_dir_name, ignore_errors=True)
  return Item
