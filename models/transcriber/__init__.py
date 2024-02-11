from .models import punctuation_restorer, transcriber


def transcribe(wav_path: str):
  raw_content = transcriber.transcribe([wav_path])[0]
  restored_content = punctuation_restorer.add_punctuation_capitalization(raw_content)
  return restored_content[0]
