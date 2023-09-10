import boto3
from fastapi import FastAPI, UploadFile, Form
import os
import subprocess
import tempfile

app = FastAPI()
s3_client = boto3.client('s3')


def compressCommand(mp4_name):
  return [
      "ffmpeg", "-i", "-", "-vf", "fps=1", "-c:v", "libx265", "-crf", "28", "-acodec", "pcm_s16le",
      "-ar", "16000", "-ac", "1", "-f", "mp4", "--", mp4_name
  ]


def audioCommand(mp4_name, wav_name):
  return ["ffmpeg", "-i", mp4_name, "-vn", "--", wav_name]


@app.get("/")
def read_root():
  return {"Hello": "World"}


@app.post("/")
async def receive_video(file: UploadFile = Form(...), topic: str = Form(...)):
  # Log
  basename = file.filename.replace(".mp4", "")
  print("File:", basename)
  print("Topic:", topic)
  # Convert to bytes-like object
  video_bytes = await file.read()
  temp_dir = tempfile.TemporaryDirectory()
  # Compress
  mp4_name = os.path.join(temp_dir.name, "compressed.mp4")
  compress_process = subprocess.Popen(compressCommand(mp4_name), stdin=subprocess.PIPE)
  compress_process.communicate(video_bytes)
  video_key = os.path.join("video/og", basename + ".mp4")
  s3_client.upload_file(mp4_name, "vampp", video_key)
  # Extract audio
  wav_name = os.path.join(temp_dir.name, "audio.wav")
  audio_process = subprocess.Popen(audioCommand(mp4_name, wav_name))
  audio_process.communicate()
  audio_key = os.path.join("audio/og", basename + ".wav")
  s3_client.upload_file(wav_name, "vampp", audio_key)

  temp_dir.cleanup()

  return "ok"