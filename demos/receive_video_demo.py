# import concurrent.futures
import gradio as gr
import os

from server.models.ridge import inferPe
from server.services.audio import predictSpeechStats
from server.services.cv import localizeFrames, predictFrames, readFrames, restoreAndBatchFrames
from server.utils.common import tempDir, tempPath

from typing import Any, Dict


async def fn(id: str, _temp_mp4_path: str, _temp_wav_path: str, topic: str):
  if id == "":
    raise gr.Error("id cannot be empty")

  temp_dir = tempDir(id, [])
  temp_mp4_path = tempPath(temp_dir, ["og.mp4"])
  temp_wav_path = tempPath(temp_dir, ["og.wav"])
  os.rename(_temp_mp4_path, temp_mp4_path)
  os.rename(_temp_wav_path, temp_wav_path)

  Item: Dict[str, Any] = {}

  def framesFn():
    nonlocal Item, temp_mp4_path, temp_wav_path
    readFrames_gen = readFrames(temp_mp4_path)
    localizeFrames_gen = localizeFrames(readFrames_gen, tempPath(temp_dir, ["frame", "xyxy.csv"]))
    temp_restored_dir = tempDir(temp_dir, ["frame", "restored"])
    for k, v in predictFrames(restoreAndBatchFrames(localizeFrames_gen, temp_restored_dir),
                              tempPath(temp_dir, ["frame", "multitask.csv"]),
                              tempPath(temp_dir, ["frame", "attire.csv"])):
      Item[k] = v

  try:
    framesFn()
  except Exception as e:
    raise gr.Error(str(e))

  def speechStatsFn():
    nonlocal Item, temp_wav_path
    for k, v in predictSpeechStats(temp_wav_path, tempPath(temp_dir,
                                                           ["audio", "speech_stats.csv"])):
      Item[f"speech_{k}"] = v

  try:
    speechStatsFn()
  except Exception as e:
    raise gr.Error(str(e))

  # def predictPitch():
  #   nonlocal topic, title
  #   pitch_arg_ls = ["pitch.txt"]
  #   temp_pitch_path = tempPath(temp_dir, pitch_arg_ls)
  #   pitch = transcribe(temp_wav_path)
  #   setItem("pitch", "S", pitch)

  #   topic, summary = summarize(pitch, topic, title)
  #   setItem("topic", "S", topic)
  #   setItem("summary", "S", summary)

  #   chain = Chain(temp_dir)
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

  # # Create a ThreadPoolExecutor to run the functions in parallel
  # with concurrent.futures.ThreadPoolExecutor() as executor:
  #   done, not_done = concurrent.futures.wait([
  #       executor.submit(fn) for fn in [
  #           framesFn,
  #           speechStatsFn,
  #           # predictPitch,
  #       ]
  #   ])
  # if len(not_done) > 0:
  #   raise gr.Error("Some tasks failed")

  print(f"Before Ridge:", Item, sep="\n")

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

  print(f"After Ridge:", Item, sep="\n")

  return Item


receive_video_demo = gr.Interface(
    fn=fn,
    inputs=[
        gr.Textbox(label="id"),
        gr.File(label="video", file_types=[".mp4"]),
        gr.File(label="audio", file_types=[".wav"]),
        gr.Textbox(label="topic", value=""),
    ],
    outputs=gr.JSON(label="Item"),
)
