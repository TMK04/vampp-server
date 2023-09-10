from fastapi import FastAPI, UploadFile, Form

app = FastAPI()


@app.get("/")
def read_root():
  return {"Hello": "World"}


@app.post("/")
async def receive_video(file: UploadFile = Form(...), topic: str = Form(...)):
  # Log
  print("Video size:", len(file))
  print("Topic:", topic)
  return "ok"