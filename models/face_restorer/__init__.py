from basicsr.utils import imwrite, img2tensor, tensor2img
from config import MODEL_FR_PATH
import cv2
import glob
import os
import subprocess
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from ..components import device
from .models import face_helper, net


# Adapted from https://github.com/sczhou/CodeFormer/blob/v0.1.0/inference_codeformer.py
def restoreFaces(input_dir, output_dir):
  for img_path in sorted(glob.glob(os.path.join(input_dir, '*.[jp][pn]g'))):
    # clean all the intermediate results to process the next image
    face_helper.clean_all()

    img_name = os.path.basename(img_path)
    # if not '04' in img_name:
    #     continue
    print(f'Processing: {img_name}')
    basename, ext = os.path.splitext(img_name)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

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
          output = net(cropped_face_t, w=w, adain=True)[0]
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
      save_restore_path = os.path.join(output_dir, f'{basename}.png')
      imwrite(restored_img, save_restore_path)
