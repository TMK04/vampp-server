from .model import model


def transcribe(wav_path: str):
  return model.transcribe([wav_path])[0]
