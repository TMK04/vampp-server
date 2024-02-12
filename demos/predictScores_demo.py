import json
import gradio as gr

from server.models.ridge import inferBv, inferClarity, inferPe
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

  return scores


name = "predictScores"
demo = gr.Interface(
    api_name=name,
    fn=fn,
    inputs=gr.Textbox(label="id"),
    outputs=gr.JSON(label="scores"),
)
