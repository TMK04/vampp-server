from dotenv import load_dotenv

load_dotenv()

import boto3
import cv2
from cv_helpers import extractFrames
from fastapi import FastAPI, UploadFile, Form
from ffmpeg_commands import compressVideo, extractAudio
import os
from re_patterns import pattern_mp4_suffix
import tempfile

app = FastAPI()
s3_client = boto3.client('s3')


@app.get("/")
def read_root():
  return {"Hello": "World"}


@app.post("/")
async def receive_video(file: UploadFile = Form(...), topic: str = Form(...)):
  basename = re.sub(pattern_mp4_suffix, "", file.filename)

  def s3Key(arg_ls):
    return os.path.join(basename, *arg_ls)

  temp_arg_ls = ["temp.mp4"]
  with tempfile.TemporaryDirectory() as temp_dir_name:

    def tempName(arg_ls):
      return os.path.join(temp_dir_name, "-".join(arg_ls))

    temp_file_name = tempName(temp_arg_ls)
    # Save file to temp
    with open(temp_file_name, "wb") as f:
      f.write(await file.read())

    mp4_arg_ls = ["og.mp4"]
    temp_mp4_name = tempName(mp4_arg_ls)
    compressVideo(temp_file_name, temp_mp4_name)
    os.remove(temp_file_name)
    video_key = s3Key(mp4_arg_ls)
    s3_client.upload_file(temp_mp4_name, "vampp", video_key)

    wav_arg_ls = ["audio", "og.wav"]
    temp_wav_name = tempName(wav_arg_ls)
    extractAudio(temp_mp4_name, temp_wav_name)
    audio_key = s3Key(wav_arg_ls)
    s3_client.upload_file(temp_wav_name, "vampp", audio_key)

    def handleFrames(i, frame, to_localize_frame):
      i_file = f"{i}.jpg"

      og_arg_ls = ["frame", "og", i_file]
      temp_og_name = tempName(og_arg_ls)
      cv2.imwrite(temp_og_name, frame)
      og_key = s3Key(og_arg_ls)
      s3_client.upload_file(temp_og_name, "vampp", og_key)

      to_localize_arg_ls = ["frame", "to_localize", i_file]
      temp_to_localize_name = tempName(to_localize_arg_ls)
      cv2.imwrite(temp_to_localize_name, to_localize_frame)
      to_localize_key = s3Key(to_localize_arg_ls)
      s3_client.upload_file(temp_to_localize_name, "vampp", to_localize_key)

    extractFrames(temp_mp4_name, handleFrames)
    os.remove(temp_mp4_name)

  return "ok"