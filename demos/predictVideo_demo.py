import json
import gradio as gr
import os.path
import shutil
from tempfile import _TemporaryFileWrapper

from server.services.cv import localizeFrames, predictFrames, readFrames, restoreAndBatchFrames
from server.utils.common import tempDir, tempPath

from .utils import X_keys, dumpKv

from typing import Any, Dict


async def fn(id: str, _temp_mp4: _TemporaryFileWrapper):
  if id == "":
    raise gr.Error("id cannot be empty")
  print("predictVideo", id, _temp_mp4)

  temp_dir = tempDir(id, [], exist_ok=True)
  temp_mp4_path = tempPath(temp_dir, ["og.mp4"])
  if not os.path.isfile(temp_mp4_path):
    # move _temp_mp4 to temp_mp4_path
    shutil.copy(_temp_mp4.name, temp_mp4_path)
  _temp_mp4.close()

  def framesFn():
    nonlocal temp_mp4_path
    readFrames_gen = readFrames(temp_mp4_path)
    localizeFrames_gen = localizeFrames(readFrames_gen, tempPath(temp_dir, ["frame", "xyxy.csv"]))
    temp_restored_dir = tempDir(temp_dir, ["frame", "restored"])
    return predictFrames(restoreAndBatchFrames(localizeFrames_gen, temp_restored_dir),
                         tempPath(temp_dir, ["frame", "multitask.csv"]),
                         tempPath(temp_dir, ["frame", "attire.csv"]))

  subscores: Dict[str, Any] = {}

  try:
    try:
      for k, v in framesFn():
        if k in X_keys:
          subscores[k] = v
        yield dumpKv(k, v)
    finally:
      with open(tempPath(temp_dir, ["video.json"]), "w") as f:
        json.dump(subscores, f)
  except Exception as e:
    raise gr.Error(str(e))


name = "predictVideo"
demo = gr.Interface(
    api_name=name,
    fn=fn,
    inputs=[
        gr.Textbox(label="id"),
        gr.File(label="video", file_types=[".mp4"]),
    ],
    outputs=gr.Textbox(label="subscores"),
)
