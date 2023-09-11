import os
import sys
import torch
import librosa
import whisper


def split_audio(audio_file_path, ):

  audio, sr = librosa.load(audio_file_path, sr=16000)
  # Calculate window size
  window_duration = 10  # in seconds
  window_size = int(window_duration * sr)

  # Split the audio
  for i in range(0, len(audio), window_size):
    window = audio[i:i + window_size]
    if len(window) == window_size:
      yield window


def transcribe_and_correct(audio_file_path):
  model = whisper.load_model("base.en")
  result = model.transcribe(audio_file_path)

  return result['text']
