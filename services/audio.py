import numpy as np
import pandas as pd
import soundfile as sf

from server.config import AUDIO_BATCH, AUDIO_SR
from server.models.speech_stats import batchInferSpeechStats, preprocess
from server.utils.common import DictKeyArr, batchGen


def splitAudio(wav_path: str):
  for i, window in enumerate(sf.blocks(wav_path, blocksize=AUDIO_SR * 10, overlap=0, fill_value=0)):
    yield str(i), preprocess(window)


def splitAndBatchAudio(temp_wav_path):
  return batchGen(splitAudio(temp_wav_path), AUDIO_BATCH)


speech_stats_key_ls = ["enthusiasm", "clarity"]


def predictSpeechStats(wav_path: str, speech_stats_path: str):
  speech_stats_df = DictKeyArr(speech_stats_key_ls)
  for i_batch, window_batch in splitAndBatchAudio(wav_path):
    speech_stats = batchInferSpeechStats(window_batch)
    speech_stats_df["i"].extend(i_batch)
    for j, key in enumerate(speech_stats_key_ls):
      speech_stats_df[key].extend(speech_stats[:, j])
  speech_stats_df = pd.DataFrame(speech_stats_df).set_index("i")
  speech_stats_df.to_csv(speech_stats_path)

  for key in speech_stats_key_ls:
    yield key, speech_stats_df[key].mean()
