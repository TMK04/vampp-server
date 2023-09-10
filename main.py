import boto3
from fastapi import FastAPI, UploadFile, Form
import os

app = FastAPI()
s3_client = boto3.client('s3')


@app.get("/")
def read_root():
  return {"Hello": "World"}


@app.post("/")
async def receive_video(file: UploadFile = Form(...), topic: str = Form(...)):
  # Log
  print("Video:", file)
  print("Topic:", topic)
  # Convert to bytes-like object
  file_bytes = await file.read()
  print(file_bytes)
  # Upload to S3
  Key = os.path.join("og", file.filename)
  s3_client.put_object(Bucket="vampp", Key=Key, Body=file_bytes)
  return "ok"