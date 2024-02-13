from base64 import b64encode
import cv2
import ffmpy
import lzma
import pandas as pd

from server.utils.common import tempPath
from server.utils.cv import OG_HEIGHT, OG_WIDTH


def generateFinalVideo(temp_dir: str):
  temp_mp4_path = tempPath(temp_dir, ["og.mp4"])
  temp_audio_path = tempPath(temp_dir, ["og.wav"])
  temp_temp_final_path = tempPath(temp_dir, ["temp_final.mp4"])
  temp_final_path = tempPath(temp_dir, ["final.mp4"])
  temp_speech_stats_path = tempPath(temp_dir, ["audio-speech_stats.csv"])
  temp_xdensenet_path = tempPath(temp_dir, ["frame-multitask.csv"])
  temp_xyxy_path = tempPath(temp_dir, ["frame-xyxy.csv"])

  speech_stats_df = pd.read_csv(temp_speech_stats_path, index_col=0)
  xdensenet_df = pd.read_csv(temp_xdensenet_path, index_col=0)
  xyxy_df = pd.read_csv(temp_xyxy_path, index_col=0)

  cap = cv2.VideoCapture(temp_mp4_path)
  temp_final_video = cv2.VideoWriter(temp_temp_final_path, cv2.VideoWriter_fourcc(*'mp4v'), 1,
                                     (OG_WIDTH, OG_HEIGHT))
  i = 0
  stats = {}
  xyxy = None
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    if i in xdensenet_df.index:
      xdensenet = xdensenet_df.loc[i]
      stats["Moving"] = xdensenet["moving"]
      stats["Smiling"] = xdensenet["smiling"]
      stats["Upright"] = xdensenet["upright"]
      stats["Eye Contact"] = xdensenet["ec"]
    if i in speech_stats_df.index:
      speech_stats = speech_stats_df.loc[i]
      stats["Speech Clarity"] = speech_stats["clarity"]
      stats["Speech Enthusiasm"] = speech_stats["enthusiasm"]
    for i, (k, v) in enumerate(stats.items()):
      if type(v) == float:
        v = f"{v:.2f}"
      cv2.putText(frame, f"{k}: {v}", (10, 20 * (i + 1)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                  (255, 255, 255), 1)

    if i in xyxy_df.index:
      xyxy = xyxy_df.loc[i]
    if xyxy is not None:
      cv2.rectangle(frame, (xyxy["x1"], xyxy["y1"]), (xyxy["x2"], xyxy["y2"]), (0, 0, 255), 1)

    temp_final_video.write(cv2.resize(frame, (OG_WIDTH, OG_HEIGHT)))
    i += 1
  cap.release()
  temp_final_video.release()
  # join temp_temp_final_video & temp_audio
  ffmpy.FFmpeg(inputs={
      temp_temp_final_path: None,
      temp_audio_path: None
  },
               outputs={
                   temp_final_path: "-c:v copy -c:a aac -y -crf 28"
               }).run()
  # get data url (base64) of final video
  with open(temp_final_path, "rb") as f:
    data = f.read()
  data = f"data:video/mp4;base64,{b64encode(data).decode()}"
  data = b64encode(lzma.compress(data.encode())).decode()
  return data
