import cv2
import os


def resizeWithPad(image, target_width: int, target_height: int, print_diff=False):
  height, width, _ = image.shape
  if print_diff:
    if width == target_width and height == target_height:
      return image
    else:
      print("Resizing...")
  ratio = min(target_width / width, target_height / height)
  new_width = int(width * ratio)
  new_height = int(height * ratio)
  resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
  delta_w = target_width - new_width
  delta_h = target_height - new_height
  top, bottom = delta_h // 2, delta_h - (delta_h // 2)
  left, right = delta_w // 2, delta_w - (delta_w // 2)
  color = [0, 0, 0]
  return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def extractFrames(input_file: str, handleFrames: callable):
  cap = cv2.VideoCapture(input_file)
  i = 0
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    frame = resizeWithPad(frame, 1280, 720)
    to_localize_frame = cv2.cvtColor(cv2.resize(frame, (426, 240)), cv2.COLORBGR2GRAY)
    handleFrames(i, frame, to_localize_frame)
    i += 1
  cap.release()