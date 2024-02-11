from .models import punctuation_restorer, transcriber


def transcribe(wav_path: str):
  raw_content = transcriber.transcribe([wav_path])[0][0].strip()
  restored_content = punctuation_restorer.add_punctuation_capitalization([raw_content])[0].strip()
  return restored_content
