# import concurrent.futures
import gradio as gr
import os

from server.models.ridge import inferBv, inferClarity, inferPe

from typing import Any, Dict, Iterable


def SubsetArr(_dict: Dict[str, Any], keys: Iterable[str]):
  return [_dict[key] for key in keys]


async def fn(subscores: Dict[str, Any]):
  scores = {}

  X_pe = SubsetArr(subscores, ["moving", "smiling", "upright", "ec", "pa", "speech_enthusiasm"])
  scores["pe"] = inferPe(X_pe)

  X_clarity = SubsetArr(subscores, ["speech_clarity", "beholder_clarity"])
  scores["clarity"] = inferClarity(X_clarity)

  X_bv = SubsetArr(subscores, ["beholder_creativity", "beholder_feasibility", "beholder_impact"])
  scores["bv"] = inferBv(X_bv)

  return scores


name = "predictScores"
demo = gr.Interface(
    api_name=name,
    fn=fn,
    inputs=gr.JSON(label="subscores"),
    outputs=gr.JSON(label="scores"),
)
