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


def tempName(temp_dir_name, *args):
  return os.path.join(temp_dir_name, "-".join(args))


@app.post("/")
async def receive_video(file: UploadFile = Form(...), topic: str = Form(...)):
  mp4_name = file.filename
  basename = mp4_name.replace(".mp4", "")
  wav_name = f"{basename}.wav"

  with tempfile.TemporaryDirectory() as temp_dir_name:
    temp_file_name = tempName(temp_dir_name, "temp.mp4")
    # Save file to temp
    with open(temp_file_name, "wb") as f:
      f.write(await file.read())

    temp_mp4_name = tempName(temp_dir_name, mp4_name)
    compressVideo(temp_file_name, temp_mp4_name)
    os.remove(temp_file_name)
    video_key = os.path.join("og", mp4_name)
    s3_client.upload_file(temp_mp4_name, "vampp", video_key)

    temp_wav_name = tempName(temp_dir_name, wav_name)
    extractAudio(temp_mp4_name, temp_wav_name)
    audio_key = os.path.join("audio/og", wav_name)
    s3_client.upload_file(temp_wav_name, "vampp", audio_key)

    def handleFrames(i, frame, to_localize_frame):
      i_file = f"{i}.jpg"
      basename_i_file = f"{basename}-{i_file}"

      og_prefix_ls = ["frames", "og"]
      temp_og_name = tempName(temp_dir_name, *og_prefix_ls, basename_i_file)
      cv2.imwrite(temp_og_name, frame)
      og_key = os.path.join(*og_prefix_ls, basename, i_file)
      s3_client.upload_file(temp_og_name, "vampp", og_key)

      to_localize_prefix_ls = ["frames", "to_localize"]
      temp_to_localize_name = tempName(temp_dir_name, *to_localize_prefix_ls, basename_i_file)
      cv2.imwrite(temp_to_localize_name, to_localize_frame)
      to_localize_key = os.path.join(*to_localize_prefix_ls, basename, i_file)
      s3_client.upload_file(temp_to_localize_name, "vampp", to_localize_key)

    extractFrames(temp_mp4_name, handleFrames)
    os.remove(temp_mp4_name)

  return "ok"