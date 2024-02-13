import json
import gradio as gr
import shutil

from server.models.ridge import inferBv, inferClarity, inferPe
from server.services.final_video import makeFinalVideo
from server.utils.common import tempDir, tempPath

from .utils import X_bv_keys, X_clarity_keys, X_pe_keys

from typing import Any, Dict, Iterable


def loadSubscores(temp_path: str):
  with open(temp_path) as f:
    return json.load(f)


def SubsetArr(_dict: Dict[str, Any], keys: Iterable[str]):
  return [_dict[key] for key in keys]


async def fn(id: str):
  if id == "":
    raise gr.Error("id cannot be empty")

  try:
    temp_dir = tempDir(id, [], exist_ok=True)

    subscores = {
        **loadSubscores(tempPath(temp_dir, ["audio.json"])),
        **loadSubscores(tempPath(temp_dir, ["video.json"]))
    }

    scores = {}

    X_pe = SubsetArr(subscores, X_pe_keys)
    scores["pe"] = inferPe(X_pe)

    X_clarity = SubsetArr(subscores, X_clarity_keys)
    scores["clarity"] = inferClarity(X_clarity)

    X_bv = SubsetArr(subscores, X_bv_keys)
    scores["bv"] = inferBv(X_bv)

    scores["final_video"] = makeFinalVideo(temp_dir)

    return json.dumps(scores)
  finally:
    shutil.rmtree(temp_dir)


name = "predictFinal"
demo = gr.Interface(
    api_name=name,
    fn=fn,
    inputs=gr.Textbox(label="id"),
    outputs=gr.Textbox(label="final"),
)
