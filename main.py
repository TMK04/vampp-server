import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
  sys.path.append(module_path)

from dotenv import load_dotenv

load_dotenv(os.path.join(module_path, "server/.env"))

import concurrent.futures
from fastapi import FastAPI, HTTPException, Form
from pathlib import Path

from server.config import OUT_DIR
from server.models.ridge import inferPe
from server.services.audio import predictSpeechStats
from server.services.cv import localizeFrames, predictFrames, restoreAndBatchFrames

from typing import Any, Dict

app = FastAPI()
tmp_dir = Path("tmp/")
tmp_dir.mkdir(parents=True, exist_ok=True)


@app.post("/")
async def receive_video(id: str = Form(...), topic: str = Form(""), title: str = Form("")):
  temp_dir = os.path.join(OUT_DIR, id)

  Item: Dict[str, Any] = dict(id=id, )

  def tempPath(arg_ls):
    return os.path.join(temp_dir, "-".join(arg_ls))

  def tempDir(arg_ls):
    temp_path = tempPath(arg_ls)
    Path(temp_path).mkdir()
    return temp_path

  temp_mkv_path = tempPath(["og.mkv"])
  temp_wav_path = tempPath(["og.wav"])

  def framesFn():
    temp_localized_dir = tempDir(["frame", "localized"])
    localizeFrames(temp_mkv_path, temp_localized_dir, tempPath(["frame", "xyxy.csv"]))
    temp_restored_dir = tempDir(["frame", "restored"])
    for k, v in predictFrames(restoreAndBatchFrames(temp_localized_dir, temp_restored_dir),
                              tempPath(["frame", "multitask.csv"]),
                              tempPath(["frame", "attire.csv"])):
      Item[k] = v

  def speechStatsFn():
    for k, v in predictSpeechStats(temp_wav_path, tempPath(["audio", "speech_stats.csv"])):
      Item[k] = v

  # def predictPitch():
  #   nonlocal topic, title
  #   pitch_arg_ls = ["pitch.txt"]
  #   temp_pitch_path = tempPath(pitch_arg_ls)
  #   pitch = transcribe(temp_wav_path)
  #   setItem("pitch", "S", pitch)

  #   topic, summary = summarize(pitch, topic, title)
  #   setItem("topic", "S", topic)
  #   setItem("summary", "S", summary)

  #   chain = Chain(id)
  #   try:
  #     beholder_response = runBeholderFirst(chain, topic, pitch)
  #   except ValueError as e:
  #     raise HTTPException(status_code=500, detail=str(e))
  #   for key, value in beholder_response:
  #     Item_key = f"beholder_{key}"
  #     if key.endswith("_justification"):
  #       setItem(Item_key, "S", value)
  #     else:
  #       setItem(Item_key, "N", str(value))

  # Create a ThreadPoolExecutor to run the functions in parallel
  with concurrent.futures.ThreadPoolExecutor() as executor:
    concurrent.futures.wait([
        executor.submit(fn) for fn in [
            framesFn,
            speechStatsFn,
            # predictPitch,
        ]
    ])

  X_pe = [
      *[Item[key] for key in ["moving", "smiling", "upright", "ec"]],
      Item["pa"],
      Item["speech_enthusiasm"],
  ]
  Item["pe"] = inferPe(X_pe)
  # X_clarity = [Item[key]["N"] for key in ["speech_clarity", "beholder_clarity"]]
  # setItem("clarity", "N", str(rfrInfer(rfr_clarity, X_clarity)))
  # X_bv = [
  #     Item[key]["N"] for key in ["beholder_creativity", "beholder_feasibility", "beholder_impact"]
  # ]
  # setItem("bv", "N", str(rfrInfer(rfr_bv, X_bv)))

  # setItem("ts", "N", str(int(time())))

  print(Item)
  return Item
