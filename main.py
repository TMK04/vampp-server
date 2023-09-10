from extract import process
from fastapi import FastAPI, UploadFile, Form

app = FastAPI()


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
  process.communicate(input=file_bytes)
  return "ok"