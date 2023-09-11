from dotenv import load_dotenv

load_dotenv()

from aws import AWS_S3_BUCKET, USE_AWS, s3_client
import cv2
from cv_helpers import extractFrames
from fastapi import FastAPI, UploadFile, Form
from ffmpeg_commands import compressVideo, extractAudio
import os
from pathlib import Path
import re
from re_patterns import pattern_mp4_suffix
import shortuuid

app = FastAPI()
tmp_dir = Path("tmp/")
tmp_dir.mkdir(parents=True, exist_ok=True)


@app.get("/")
def read_root():
  return {"Hello": "World"}


@app.post("/")
async def receive_video(file: UploadFile = Form(...), topic: str = Form(...)):
  basename = re.sub(pattern_mp4_suffix, "", file.filename)

  # Create temp dir
  while True:
    try:
      basename_random = f"{basename}-{shortuuid.ShortUUID().random(length=7)}"
      temp_dir_name = os.path.join(tmp_dir, basename_random)
      Path(temp_dir_name).mkdir()
      break
    except FileExistsError:
      continue

  if USE_AWS:

    def s3Key(arg_ls):
      return os.path.join(basename_random, *arg_ls)

  def tempName(arg_ls):
    return os.path.join(temp_dir_name, "-".join(arg_ls))

  # Save file to temp
  temp_arg_ls = ["temp.mp4"]
  temp_file_name = tempName(temp_arg_ls)
  with open(temp_file_name, "wb") as f:
    f.write(await file.read())

  mp4_arg_ls = ["og.mp4"]
  temp_mp4_name = tempName(mp4_arg_ls)
  compressVideo(temp_file_name, temp_mp4_name)
  os.remove(temp_file_name)
  if USE_AWS:
    video_key = s3Key(mp4_arg_ls)
    s3_client.upload_file(temp_mp4_name, AWS_S3_BUCKET, video_key)

  wav_arg_ls = ["audio", "og.wav"]
  temp_wav_name = tempName(wav_arg_ls)
  extractAudio(temp_mp4_name, temp_wav_name)
  if USE_AWS:
    audio_key = s3Key(wav_arg_ls)
    s3_client.upload_file(temp_wav_name, AWS_S3_BUCKET, audio_key)

  def saveFrames():
    for batch in extractFrames(temp_mp4_name):
      current_batch = []
      for i, frame, to_localize_frame in batch:
        i_file = f"{i}.jpg"

        og_arg_ls = ["frame", "og", i_file]
        temp_og_name = tempName(og_arg_ls)
        cv2.imwrite(temp_og_name, frame)
        if USE_AWS:
          og_key = s3Key(og_arg_ls)
          s3_client.upload_file(temp_og_name, AWS_S3_BUCKET, og_key)

        to_localize_arg_ls = ["frame", "to_localize", i_file]
        temp_to_localize_name = tempName(to_localize_arg_ls)
        cv2.imwrite(temp_to_localize_name, to_localize_frame)
        if USE_AWS:
          to_localize_key = s3Key(to_localize_arg_ls)
          s3_client.upload_file(temp_to_localize_name, AWS_S3_BUCKET, to_localize_key)

        current_batch.append((temp_og_name, temp_to_localize_name))
      yield current_batch
      current_batch = []

  for _ in saveFrames():
    pass
  os.remove(temp_mp4_name)

  os.rmdir(temp_dir_name)

  return "ok"