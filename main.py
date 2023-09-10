import boto3
from fastapi import FastAPI, UploadFile, Form
import os
import subprocess
import tempfile

app = FastAPI()
s3_client = boto3.client('s3')


def compressCommand(mp4_name):
  return [
      "ffmpeg", "-i", mp4_name, "-vf", "fps=1", "-c:v", "libx265", "-crf", "28", "-acodec",
      "pcm_s16le", "-ar", "16000", "-ac", "1", "-f", "mp4", "-"
  ]


def audioCommand(mp4_name, wav_name):
  return ["ffmpeg", "-i", mp4_name, "-vn", wav_name]


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
  # Create temp file
  with tempfile.NamedTemporaryFile(suffix=".mp4") as mp4_temp:
    mp4_temp.write(video_bytes)
    mp4_temp.flush()
    # Compress
    compress_process = subprocess.Popen(compressCommand(mp4_temp.name), stdout=subprocess.PIPE)
    video_bytes, err = compress_process.communicate()
    if err:
      print(err)
      return "error"
    print(type(video_bytes))
    video_key = os.path.join("og", basename + ".mp4")
    s3_client.upload_fileobj(mp4_temp, "vampp", video_key)
    # Overwrite mp4_temp file
    mp4_temp.seek(0)
    mp4_temp.write(video_bytes)
    mp4_temp.flush()
    # Extract audio
    with tempfile.NamedTemporaryFile(suffix=".wav") as wav_temp:
      audio_process = subprocess.Popen(audioCommand(mp4_temp.name, wav_temp.name))
      audio_bytes, err = audio_process.communicate()
      if err:
        print(err)
        return "error"
      print(type(audio_bytes))
      audio_key = os.path.join("audio/og", basename + ".wav")
      s3_client.upload_fileobj(wav_temp, "vampp", audio_key)

  return "ok"