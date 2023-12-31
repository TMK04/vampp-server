import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
  sys.path.append(module_path)

from dotenv import load_dotenv

load_dotenv(os.path.join(module_path, "server/.env"))

import concurrent.futures
import cv2
from fastapi import FastAPI, HTTPException, Form
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from time import time

from server.audio import transcribe, splitAudio, splitAudioBatch
from server.aws import AWS_DYNAMO_TABLE, USE_AWS, dynamo_client
from server.config import FRAME_ATTIRE_MASK, OUT_DIR
from server.models.components import device, infer, toTensor
from server.models.face_restorer import restoreFaces
from server.models.llm import Chain, runBeholderFirst, summarize
from server.models.rfr import rfr_bv, rfr_clarity, rfr_pe, rfrInfer
from server.models.speech_stats import preprocess, speech_stats_model
from server.models.xdensenet import attire_model, multitask_model
from server.services.cv import localizeFrames, batchRestoredFrames
from server.video_commands import compressVideo, extractAudio

app = FastAPI()
tmp_dir = Path("tmp/")
tmp_dir.mkdir(parents=True, exist_ok=True)


@app.post("/")
async def receive_video(topic: str = Form(""), title: str = Form(""), id: str = Form(...)):
  temp_dir = os.path.join(OUT_DIR, id)

  Item = {"id": {"S": id}, "topic": {"S": topic}}

  def setItem(key, type_code, value):
    Item[key] = {type_code: value}
    print(f"{key}: {value}")

  def tempPath(arg_ls):
    return os.path.join(temp_dir, "-".join(arg_ls))

  # Save file to temp
  temp_file_path = tempPath(["temp.mp4"])

  temp_mp4_path = tempPath(["og.mp4"])
  compressVideo(temp_file_path, temp_mp4_path)
  os.remove(temp_file_path)

  temp_wav_path = tempPath(["audio", "og.wav"])
  extractAudio(temp_mp4_path, temp_wav_path)

  def restoreFrames(temp_localized_dir):
    temp_restored_dir = tempPath(["frame", "restored"])
    Path(temp_restored_dir).mkdir()
    restoreFaces(temp_localized_dir, temp_restored_dir)
    temp_restored_dir = os.path.join(temp_restored_dir, "restored_imgs")
    temp_restored_basename_ls = os.listdir(temp_restored_dir)
    for temp_restored_basename in temp_restored_basename_ls:
      temp_restored_path = os.path.join(temp_restored_dir, temp_restored_basename)
      restored_frame = cv2.imread(temp_restored_path, cv2.IMREAD_GRAYSCALE)
      restored_frame = np.expand_dims(restored_frame, axis=0)
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
      temp_multitask_path = tempPath(multitask_arg_ls)
      multitask_df.to_csv(temp_multitask_path)
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
      temp_attire_path = tempPath(attire_arg_ls)
      attire_df.to_csv(temp_attire_path)
    attire_mode = bool(attire_df["attire"].mode()[0])
    setItem("pa", "BOOL", attire_mode)

  def framesFn():
    localized_dir_arg_ls = ["frame", "localized"]
    temp_localized_dir = tempPath(localized_dir_arg_ls)
    Path(temp_localized_dir).mkdir()

    temp_xyxyn_path = tempPath(["frame", "xyxy.csv"])
    localizeFrames(temp_mp4_path, temp_localized_dir, temp_xyxyn_path)
    os.remove(temp_mp4_path)
    predictFrames(batchRestoredFrames(restoreFrames(temp_localized_dir)))

  def predictSpeechStats():
    speech_stats_key_ls = ["enthusiasm", "clarity"]
    speech_stats_df_dict = {key: [] for key in ["i", *speech_stats_key_ls]}
    for batch_i, batch_window in splitAudioBatch(splitAudio(temp_wav_path)):
      batch_window_tensor = toTensor(batch_window)
      speech_stats = infer(speech_stats_model, batch_window_tensor)
      speech_stats_df_dict["i"].extend(batch_i)
      for j, key in enumerate(speech_stats_key_ls):
        speech_stats_df_dict[key].extend(speech_stats[:, j])
    speech_stats_df = pd.DataFrame(speech_stats_df_dict).set_index("i")
    speech_stats_arg_ls = ["audio", "stats.csv"]
    temp_speech_stats_path = tempPath(speech_stats_arg_ls)
    speech_stats_df.to_csv(temp_speech_stats_path)
    for key in speech_stats_key_ls:
      Item_key = f"speech_{key}"
      value = str(speech_stats_df[key].mean())
      setItem(Item_key, "N", value)

  def predictPitch():
    nonlocal topic, title
    pitch_arg_ls = ["pitch.txt"]
    temp_pitch_path = tempPath(pitch_arg_ls)
    pitch = transcribe(temp_wav_path)
    setItem("pitch", "S", pitch)

    topic, summary = summarize(pitch, topic, title)
    setItem("topic", "S", topic)
    setItem("summary", "S", summary)

    chain = Chain(id)
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
    concurrent.futures.wait(
        [executor.submit(fn) for fn in [
            framesFn,
            predictSpeechStats,
            predictPitch,
        ]])

  X_pe = [
      *[Item[key]["N"] for key in ["moving", "smiling", "upright", "ec"]],
      Item["pa"]["BOOL"],
      Item["speech_enthusiasm"]["N"],
  ]
  setItem("pe", "N", str(rfrInfer(rfr_pe, X_pe)))
  X_clarity = [Item[key]["N"] for key in ["speech_clarity", "beholder_clarity"]]
  setItem("clarity", "N", str(rfrInfer(rfr_clarity, X_clarity)))
  X_bv = [
      Item[key]["N"] for key in ["beholder_creativity", "beholder_feasibility", "beholder_impact"]
  ]
  setItem("bv", "N", str(rfrInfer(rfr_bv, X_bv)))

  setItem("ts", "N", str(int(time())))
  print(Item)
  if USE_AWS:
    dynamo_client.put_item(TableName=AWS_DYNAMO_TABLE, Item=Item)

  shutil.rmtree(temp_dir, ignore_errors=True)
  return Item
