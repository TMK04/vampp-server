import librosa
import whisper

from server.config import AUDIO_BATCH, AUDIO_SR
from server.models.speech_stats import preprocess

model = whisper.load_model("base.en")


def splitAudio(audio_file_path):
  audio, _ = librosa.load(audio_file_path, sr=AUDIO_SR)
  # Calculate window size
  window_duration = 10  # in seconds
  window_size = int(window_duration * AUDIO_SR)

  # Split the audio
  for i in range(0, len(audio), window_size):
    window = audio[i:i + window_size]
    if len(window) == window_size:
      yield i, preprocess(window)


def splitAudioBatch(gen_split_audio):
  # Split the audio
  i_batch = []
  window_batch = []
  for i, window in gen_split_audio:
    i_batch.append(i)
    window_batch.append(window)
    if len(window_batch) == AUDIO_BATCH:
      yield i_batch, window_batch
      i_batch = []
      window_batch = []
  if len(window_batch) > 0:
    yield i_batch, window_batch


def transcribe(audio_file_path):
  result = model.transcribe(audio_file_path)
  return result['text'].strip()
