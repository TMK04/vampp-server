import cv2

OG_WIDTH = 1280
OG_HEIGHT = 720


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
