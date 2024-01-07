from basicsr.utils import imwrite, img2tensor, tensor2img
import cv2
import os
import numpy as np
from server.models.presenter_localizer import LOCALIZED_HEIGHT, LOCALIZED_WIDTH
import torch
from torchvision.transforms.functional import normalize

from server.config import MODEL_FR_W
from ..components import device
from .models import face_helper, net


# Adapted from https://github.com/sczhou/CodeFormer/blob/v0.1.0/inference_codeformer.py
def restoreFace(basename, img, output_dir):
  # clean all the intermediate results to process the next image
  face_helper.clean_all()

  print(f'Processing: {basename}')

  face_helper.read_image(img)
  # get face landmarks for each face
  num_det_faces = face_helper.get_face_landmarks_5(only_center_face=True,
                                                   resize=640,
                                                   eye_dist_threshold=5)
  print(f'\tdetect {num_det_faces} faces')
  # align and warp each face
  face_helper.align_warp_face()

  # face restoration for each cropped face
  for idx, cropped_face in enumerate(face_helper.cropped_faces):
    # prepare data
    cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

    try:
      with torch.no_grad():
        output = net(cropped_face_t, w=MODEL_FR_W, adain=True)[0]
        restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
      del output
      torch.cuda.empty_cache()
    except Exception as error:
      print(f'\tFailed inference for CodeFormer: {error}')
      restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

    restored_face = restored_face.astype('uint8')
    face_helper.add_restored_face(restored_face)

  face_helper.get_inverse_affine(None)
  # paste each restored face to the input image
  restored_img = face_helper.paste_faces_to_input_image()

  # save restored img
  if restored_img is not None:
    save_restore_path = os.path.join(output_dir, f'{basename}.jpg')
    restored_img = cv2.resize(restored_img, (LOCALIZED_HEIGHT, LOCALIZED_WIDTH),
                              interpolation=cv2.INTER_CUBIC)
    imwrite(restored_img, save_restore_path)
    restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2GRAY)
    restored_img = np.expand_dims(restored_img, axis=0)
    return restored_img
