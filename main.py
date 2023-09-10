from fastapi import FastAPI, File, Form

app = FastAPI()


@app.get("/")
def read_root():
  return {"Hello": "World"}


@app.post("/")
async def receive_video(video: bytes = File(...), topic: str = Form(...)):
  # Log
  print("Video size:", len(video))
  print("String data:", string_data)
  return {"video_size": len(video), "string_data": string_data}