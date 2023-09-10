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
  read_file = file.read()
  print(read_file)
  process.communicate(input=read_file)
  return "ok"