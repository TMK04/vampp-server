import boto3
from fastapi import FastAPI, UploadFile, Form
from ffmpeg_commands import compressVideo, extractAudio
import os
import tempfile

app = FastAPI()
s3_client = boto3.client('s3')


@app.get("/")
def read_root():
  return {"Hello": "World"}


@app.post("/")
async def receive_video(file: UploadFile = Form(...), topic: str = Form(...)):
  mp4_name = file.filename
  wav_name = file.filename.replace(".mp4", ".wav")

  with tempfile.TemporaryDirectory() as tempdir_name:
    tempfile_name = os.path.join(tempdir_name, "temp.mp4")
    # Save file to temp
    with open(tempfile_name, "wb") as f:
      f.write(await file.read())

    # Compress
    tempmp4_name = os.path.join(tempdir_name, mp4_name)
    compressVideo(tempfile_name, tempmp4_name)
    video_key = os.path.join("og", mp4_name)
    s3_client.upload_file(tempmp4_name, "vampp", video_key)
    # Extract audio
    input_stream = ffmpeg.input(tempmp4_name)
    tempwav_name = os.path.join(tempdir_name, wav_name)
    extractAudio(input_stream, tempwav_name)
    audio_key = os.path.join("audio/og", wav_name)
    s3_client.upload_file(tempwav_name, "vampp", audio_key)

  return "ok"