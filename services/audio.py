import pandas as pd
import soundfile as sf

from server.config import AUDIO_BATCH, AUDIO_SR
from server.models.llm import generate
from server.models.speech_stats import batchInferSpeechStats, preprocess
from server.models.transcriber import transcribe
from server.utils.common import DictKeyArr, batchGen, toCsv

WINDOW_LEN = 10


def splitAudio(wav_path: str):
  for i, window in enumerate(
      sf.blocks(wav_path, blocksize=AUDIO_SR * WINDOW_LEN, overlap=0, fill_value=0)):
    yield i * WINDOW_LEN, preprocess(window)


def splitAndBatchAudio(temp_wav_path):
  return batchGen(splitAudio(temp_wav_path), AUDIO_BATCH)


speech_stats_keys = ("enthusiasm", "clarity")


def predictSpeechStats(wav_path: str, speech_stats_path: str):
  speech_stats_dict = DictKeyArr(speech_stats_keys)
  for i_batch, window_batch in splitAndBatchAudio(wav_path):
    speech_stats = batchInferSpeechStats(window_batch)
    speech_stats_dict["i"].extend(i_batch)
    for j, key in enumerate(speech_stats_keys):
      speech_stats_dict[key].extend(speech_stats[:, j])
  speech_stats_df = pd.DataFrame(speech_stats_dict)
  toCsv(speech_stats_df, speech_stats_path)

  for key in speech_stats_keys:
    yield f"speech_{key}", speech_stats_df[key].mean()


def predictPitch(wav_path: str, topic: str, yt_title: str):
  content = transcribe(wav_path)
  yield "pitch_content", content

  for k, v in generate(content, topic, yt_title):
    yield f"pitch_{k}", v
