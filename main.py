import boto3
from fastapi import FastAPI, UploadFile, Form
import os
import subprocess
import tempfile

app = FastAPI()
s3_client = boto3.client('s3')


def compressVideo(input_file, output_file):
  command = [
      "ffmpeg", "-i", input_file, "-vf", "fps=1", "-c:v", "libx265", "-crf", "28", "-acodec",
      "pcm_s16le", "-ar", "16000", "-ac", "1", "--", output_file
  ]
  subprocess.run(command, check=True)


def extractAudio(input_file, output_file):
  command = ["ffmpeg", "-i", input_file, "-vn", "--", output_file]
  subprocess.run(command, check=True)


@app.get("/")
def read_root():
  return {"Hello": "World"}


@app.post("/")
async def receive_video(file: UploadFile = Form(...), topic: str = Form(...)):
  mp4_name = file.filename
  wav_name = file.filename.replace(".mp4", ".wav")

  temp_dir = tempfile.mkdtemp()
  tempfile_name = os.path.join(temp_dir.name, "temp.mp4")
  # Save file to temp
  with open(tempfile_name, "wb") as f:
    f.write(await file.read())

  # Compress
  tempmp4_name = os.path.join(temp_dir.name, mp4_name)
  compressVideo(tempfile_name, tempmp4_name)
  video_key = os.path.join("og", mp4_name)
  print(video_key)
  s3_client.upload_file(tempmp4_name, "vampp", video_key)
  # Extract audio
  tempwav_name = os.path.join(temp_dir.name, wav_name)
  extractAudio(tempmp4_name, tempwav_name)
  audio_key = os.path.join("audio/og", wav_name)
  print(audio_key)
  s3_client.upload_file(tempwav_name, "vampp", audio_key)

  temp_dir.cleanup()

  return "ok"