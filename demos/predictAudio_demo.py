import gradio as gr
import json
import shutil
from tempfile import _TemporaryFileWrapper

from server.services.audio import predictPitch, predictSpeechStats
from server.utils.common import tempDir, tempPath

from .utils import X_keys, dumpKv

from typing import Any, Dict


async def fn(id: str, _temp_wav: _TemporaryFileWrapper, title: str):
  if id == "":
    raise gr.Error("id cannot be empty")

  temp_dir = tempDir(id, [], exist_ok=True)
  temp_wav_path = tempPath(temp_dir, ["og.wav"])
  # move _temp_wav to temp_wav_path
  shutil.copy(_temp_wav.name, temp_wav_path)
  _temp_wav.close()

  def speechStatsFn():
    nonlocal subscores, temp_wav_path
    return predictSpeechStats(temp_wav_path, tempPath(temp_dir, ["audio", "speech_stats.csv"]))

  def pitchFn():
    nonlocal temp_wav_path, title
    return predictPitch(temp_wav_path, title)

  subscores: Dict[str, Any] = {}

  try:
    try:
      for k, v in speechStatsFn():
        if k in X_keys:
          subscores[k] = v
        yield dumpKv(k, v)
      for k, v in pitchFn():
        if k in X_keys:
          subscores[k] = v
        yield dumpKv(k, v)
    finally:
      with open(tempPath(temp_dir, ["audio.json"]), "w") as f:
        json.dump(subscores, f)
  except Exception as e:
    raise gr.Error(str(e))


name = "predictAudio"
demo = gr.Interface(
    api_name=name,
    fn=fn,
    inputs=[
        gr.Textbox(label="id"),
        gr.File(label="audio", file_types=[".wav"]),
        gr.Textbox(label="title", value=""),
    ],
    outputs=gr.Textbox(label="subscores"),
)
