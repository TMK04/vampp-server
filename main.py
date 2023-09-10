import boto3
from fastapi import FastAPI, UploadFile, Form
import ffmpeg
import os
from uuid import uuid4

app = FastAPI()
s3_client = boto3.client('s3')


@app.get("/")
def read_root():
  return {"Hello": "World"}


compress_process = (ffmpeg.input("pipe:").output("pipe:",
                                                 hwaccel="cuda",
                                                 vf="fps=1",
                                                 vcodec="libx265",
                                                 crf=28,
                                                 acodec="pcm_s16le",
                                                 ar=16000,
                                                 ac=1).run_async(pipe_stdin=True, pipe_stdout=True))
audio_process = (ffmpeg.input("pipe:").output("pipe:", format="wav",
                                              vn=1).run_async(pipe_stdin=True, pipe_stdout=True))


@app.post("/")
async def receive_video(file: UploadFile = Form(...), topic: str = Form(...)):
  # Log
  print("File:", file.filename)
  print("Topic:", topic)
  basename = uuid4().hex
  # Convert to bytes-like object
  video_bytes = await file.read()
  # Compress
  video_bytes = compress_process.communicate(input=video_bytes)[0]
  # Extract audio
  audio_bytes = audio_process.communicate(input=video_bytes)[0]
  # Upload to S3
  s3_client.put_object(Bucket="vampp", Key=os.path.join("og", basename + ".mp4"), Body=video_bytes)
  s3_client.put_object(Bucket="vampp",
                       Key=os.path.join("audio/og", basename + ".wav"),
                       Body=audio_bytes)

  return "ok"