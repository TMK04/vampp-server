import boto3
from fastapi import FastAPI, UploadFile, Form
import os
import subprocess

app = FastAPI()
s3_client = boto3.client('s3')

compress_command = [
    "ffmpeg", "-hwaccel", "cuda", "-vf", "fps=1", "-c:v", "libx265", "-crf", "28", "-acodec",
    "pcm_s16le", "-ar", "16000", "-ac", "1"
]
audio_command = [
    "ffmpeg",
    "-vn",
    "--audio-format",
    "wav",
]


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
  # Compress
  compress_process = subprocess.Popen(compress_command, stdin=subprocess.PIPE)
  audio_process = subprocess.Popen(audio_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
  video_bytes, err = compress_process.communicate(video_bytes)
  if err:
    print(err)
    return "error"
  # Extract audio
  audio_bytes, err = audio_process.communicate(video_bytes)
  if err:
    print(err)
    return "error"
  # Upload to S3
  s3_client.put_object(Bucket="vampp", Key=os.path.join("og", basename + ".mp4"), Body=video_bytes)
  s3_client.put_object(Bucket="vampp",
                       Key=os.path.join("audio/og", basename + ".wav"),
                       Body=audio_bytes)

  return "ok"