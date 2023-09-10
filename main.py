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
      "pcm_s16le", "-ar", "16000", "-ac", "1", "-"
  ]


def audioCommand(mp4_name):
  return ["ffmpeg", "-i", mp4_name, "-vn", "-f", "wav", "-"]


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
  with tempfile.NamedTemporaryFile(suffix=".mp4") as temp:
    temp.write(video_bytes)
    temp.flush()
    # Compress
    compress_process = subprocess.Popen(compressCommand(temp.name), stdout=subprocess.PIPE)
    video_bytes, err = compress_process.communicate()
    if err:
      print(err)
      return "error"
    print(video_bytes)
    # Overwrite temp file
    temp.seek(0)
    temp.write(video_bytes)
    temp.flush()
    # Extract audio
    audio_process = subprocess.Popen(audioCommand(temp.name), stdout=subprocess.PIPE)
    audio_bytes, err = audio_process.communicate()
    if err:
      print(err)
      return "error"
    print(audio_bytes)
  # Upload to S3
  video_key = os.path.join(topic, basename + ".mp4")
  audio_key = os.path.join("audio", topic, basename + ".wav")
  s3_client.put_object(Bucket="vampp", Key=video_key, Body=video_bytes)
  s3_client.put_object(Bucket="vampp", Key=audio_key, Body=audio_bytes)

  return "ok"