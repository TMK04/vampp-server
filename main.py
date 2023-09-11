from dotenv import load_dotenv

load_dotenv()

import boto3
import cv2
from cv_helpers import extractFrames
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
  basename = mp4_name.replace(".mp4", "")
  wav_name = f"{basename}.wav"

  with tempfile.TemporaryDirectory() as tempdir_name:
    tempfile_name = os.path.join(tempdir_name, "temp.mp4")
    # Save file to temp
    with open(tempfile_name, "wb") as f:
      f.write(await file.read())

    tempmp4_name = os.path.join(tempdir_name, mp4_name)
    compressVideo(tempfile_name, tempmp4_name)
    os.remove(tempfile_name)
    video_key = os.path.join("og", mp4_name)
    s3_client.upload_file(tempmp4_name, "vampp", video_key)

    tempwav_name = os.path.join(tempdir_name, wav_name)
    extractAudio(tempmp4_name, tempwav_name)
    audio_key = os.path.join("audio/og", wav_name)
    s3_client.upload_file(tempwav_name, "vampp", audio_key)

    def handleFrames(i, frame, to_localize_frame):
      jpg_name = f"{basename}-{i}.jpg"
      og_file = os.path.join(tempdir_name, "og", jpg_name)
      cv2.imwrite(og_file, frame)
      og_key = os.path.join("frames/og", jpg_name)
      s3_client.upload_file(og_file, "vampp", og_key)

      to_localize_file = os.path.join(tempdir_name, "to_localize", jpg_name)
      cv2.imwrite(to_localize_file, to_localize_frame)
      to_localize_key = os.path.join("frames/to_localize", jpg_name)
      s3_client.upload_file(to_localize_file, "vampp", to_localize_key)

    extractFrames(tempmp4_name, handleFrames)
    os.remove(tempmp4_name)

  return "ok"