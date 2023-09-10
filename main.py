import boto3
from fastapi import FastAPI, UploadFile, Form
import ffmpeg
import os

app = FastAPI()
s3_client = boto3.client('s3')


@app.get("/")
def read_root():
  return {"Hello": "World"}


compress_process = (ffmpeg.input("pipe:", format="mp4", pix_fmt='rgb24').output(
    "pipe:",
    acodec="pcm_s16le",
    ar=16000,
    ac=1,
    format='rawvideo',
    vcodec="libx265",
    vf="fps=1",
    crf=28,
).run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True))
video_process = (ffmpeg.input("pipe:").output(
    "pipe:",
    format="mp4",
).run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True))
audio_process = (ffmpeg.input("pipe:").output("pipe:", format="wav",
                                              vn=1).run_async(pipe_stdin=True,
                                                              pipe_stdout=True,
                                                              pipe_stderr=True))


@app.post("/")
async def receive_video(file: UploadFile = Form(...), topic: str = Form(...)):
  # Log
  basename = file.filename.replace(".mp4", "")
  print("File:", basename)
  print("Topic:", topic)
  # Convert to bytes-like object
  video_bytes = await file.read()
  # Compress
  video_bytes, err = compress_process.communicate(input=video_bytes)
  if err:
    print(err)
    return "error"
  # Extract audio
  audio_bytes, err = audio_process.communicate(input=video_bytes)
  if err:
    print(err)
    return "error"
  # Upload to S3
  s3_client.put_object(Bucket="vampp", Key=os.path.join("og", basename + ".mp4"), Body=video_bytes)
  s3_client.put_object(Bucket="vampp",
                       Key=os.path.join("audio/og", basename + ".wav"),
                       Body=audio_bytes)

  return "ok"