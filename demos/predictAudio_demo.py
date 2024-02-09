# import concurrent.futures
import gradio as gr
import json
import shutil
from tempfile import _TemporaryFileWrapper

from server.services.audio import predictPitch, predictSpeechStats
from server.utils.common import tempDir, tempPath

from typing import Any, Dict


async def fn(id: str, _temp_wav: _TemporaryFileWrapper, topic: str):
  if id == "":
    raise gr.Error("id cannot be empty")

  temp_dir = tempDir(id, [], exist_ok=True)
  temp_wav_path = tempPath(temp_dir, ["og.wav"])
  # move _temp_wav to temp_wav_path
  shutil.copy(_temp_wav.name, temp_wav_path)
  _temp_wav.close()

  subscores: Dict[str, Any] = {}

  def speechStatsFn():
    nonlocal subscores, temp_wav_path
    for k, v in predictSpeechStats(temp_wav_path, tempPath(temp_dir,
                                                           ["audio", "speech_stats.csv"])):
      yield json.dumps({"k": f"speech_{k}", "v": v})

  try:
    for subscore in speechStatsFn():
      yield subscore
  except Exception as e:
    raise gr.Error(str(e))

  def pitchFn():
    nonlocal temp_wav_path
    # nonlocal topic, title
    for k, v in predictPitch(temp_wav_path):
      yield json.dumps({"k": f"pitch_{k}", "v": v})

  try:
    for subscore in pitchFn():
      yield subscore
  except Exception as e:
    raise gr.Error(str(e))

  #   topic, summary = summarize(pitch, topic, title)
  #   setsubscores("topic", "S", topic)
  #   setsubscores("summary", "S", summary)

  #   chain = Chain(temp_dir)
  #   try:
  #     beholder_response = runBeholderFirst(chain, topic, pitch)
  #   except ValueError as e:
  #     raise HTTPException(status_code=500, detail=str(e))
  #   for key, value in beholder_response:
  #     subscores_key = f"beholder_{key}"
  #     if key.endswith("_justification"):
  #       setsubscores(subscores_key, "S", value)
  #     else:
  #       setsubscores(subscores_key, "N", str(value))

  # # Create a ThreadPoolExecutor to run the functions in parallel
  # with concurrent.futures.ThreadPoolExecutor() as executor:
  #   done, not_done = concurrent.futures.wait([
  #       executor.submit(fn) for fn in [
  #           speechStatsFn,
  #           # predictPitch,
  #       ]
  #   ])
  # if len(not_done) > 0:
  #   raise gr.Error("Some tasks failed")


name = "predictAudio"
demo = gr.Interface(
    api_name=name,
    fn=fn,
    inputs=[
        gr.Textbox(label="id"),
        gr.File(label="audio", file_types=[".wav"]),
        gr.Textbox(label="topic", value=""),
    ],
    outputs=gr.Textbox(label="subscores"),
)
