from server.config import AUDIO_SR
from ..components import infer, toTensor
from .model import processor, speech_stats_model


def preprocess(x):
  y = processor(x, sampling_rate=AUDIO_SR)
  y = y['input_values'][0]
  return y


def batchInferSpeechStats(batch):
  batch_tensor = toTensor(batch)
  return infer(speech_stats_model, batch_tensor)
