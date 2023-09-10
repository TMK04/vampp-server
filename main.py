import boto3
from fastapi import FastAPI, UploadFile, Form
import ffmpeg
import os

app = FastAPI()
s3_client = boto3.client('s3')


@app.get("/")
def read_root():
  return {"Hello": "World"}


process = (ffmpeg.input("pipe:").output("out.mp4", s='{}x{}'.format(
    1280, 720)).overwrite_output().run_async(pipe_stdin=True))


@app.post("/")
async def receive_video(file: UploadFile = Form(...), topic: str = Form(...)):
  # Log
  print("File:", file.filename)
  print("Topic:", topic)
  # Convert to bytes-like object
  file_bytes = await file.read()
  file_bytes = process.communicate(input=file_bytes)[0]
  # Upload to S3
  Key = os.path.join("og", file.filename)
  s3_client.put_object(Bucket="vampp", Key=Key, Body=file_bytes)
  return "ok"