from dotenv import load_dotenv

load_dotenv()

from aws import AWS_S3_BUCKET, USE_AWS, s3_client
import cv2
from cv_helpers import extractFrames
from fastapi import FastAPI, UploadFile, Form
from ffmpeg_commands import compressVideo, extractAudio
from models.person_localizer import calculatePresenterXYXYN, localizePresenter
import os
from pathlib import Path
import re
from re_patterns import pattern_mp4_suffix
import shortuuid
import shutil
from transcription import transcribe_and_correct, split_audio
import soundfile as sf

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

    
  for i, window in enumerate(split_audio(temp_wav_name)):
    i_file = f"{i}.mp3"
    window_arg_ls = ["audio", "window", i_file]
    temp_window_name = tempName(window_arg_ls)
    sf.write(temp_window_name, window, 16000)
    window_key = s3Key(window_arg_ls)
    s3_client.upload_file(temp_window_name, AWS_S3_BUCKET, window_key)

  transcribe_and_correct(temp_wav_name)


  def saveFrames():
    for batch in extractFrames(temp_mp4_name):
      i_batch = []
      frame_batch = []
      to_localize_frame_batch = []
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

        i_batch.append(i_file)
        frame_batch.append(frame)
        to_localize_frame_batch.append(to_localize_frame)
      yield i_batch, frame_batch, to_localize_frame_batch

  def localizeFrames():
    for i_batch, frame_batch, to_localize_frame_batch in saveFrames():
      localized_frame_batch = []
      for j, xyxyn in enumerate(calculatePresenterXYXYN(to_localize_frame_batch)):
        print(j, xyxyn)
        i = i_batch[j]
        frame = frame_batch[j]
        localized_frame = localizePresenter(frame, xyxyn)
        localized_frame_batch.append(localized_frame)
      yield i_batch, localized_frame_batch

  for _ in localizeFrames():
    pass



  os.remove(temp_mp4_name)
  shutil.rmtree(temp_dir_name, ignore_errors=True)
  return "ok"