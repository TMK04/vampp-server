import gradio as gr
import json
import shutil
from tempfile import _TemporaryFileWrapper

from server.services.cv import localizeFrames, predictFrames, readFrames, restoreAndBatchFrames
from server.utils.common import tempDir, tempPath

from typing import Any, Dict


async def fn(id: str, _temp_mp4: _TemporaryFileWrapper):
  if id == "":
    raise gr.Error("id cannot be empty")
  print("predictVideo", id, _temp_mp4)

  temp_dir = tempDir(id, [], exist_ok=True)
  temp_mp4_path = tempPath(temp_dir, ["og.mp4"])
  # move _temp_mp4 to temp_mp4_path
  shutil.copy(_temp_mp4.name, temp_mp4_path)
  _temp_mp4.close()

  def framesFn():
    nonlocal temp_mp4_path
    readFrames_gen = readFrames(temp_mp4_path)
    localizeFrames_gen = localizeFrames(readFrames_gen, tempPath(temp_dir, ["frame", "xyxy.csv"]))
    temp_restored_dir = tempDir(temp_dir, ["frame", "restored"])
    for k, v in predictFrames(restoreAndBatchFrames(localizeFrames_gen, temp_restored_dir),
                              tempPath(temp_dir, ["frame", "multitask.csv"]),
                              tempPath(temp_dir, ["frame", "attire.csv"])):
      yield json.dumps({"k": k, "v": v})

  try:
    for subscore in framesFn():
      yield subscore
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
