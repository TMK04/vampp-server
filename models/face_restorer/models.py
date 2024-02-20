from basicsr.utils.registry import ARCH_REGISTRY
import os
import torch

from facelib.utils.face_restoration_helper import FaceRestoreHelper
from server.config import MODELS_DIR
from ..components import device

net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512,
                                      codebook_size=1024,
                                      n_head=8,
                                      n_layers=9,
                                      connect_list=['32', '64', '128', '256']).to(device)

wd = os.path.join(MODELS_DIR, 'face_restorer')
checkpoint = torch.load(os.path.join(wd, "codeformer.pth"))['params_ema']
net.load_state_dict(checkpoint)
net.eval()

face_helper = FaceRestoreHelper(1,
                                face_size=512,
                                crop_ratio=(1, 1),
                                det_model='YOLOv5n',
                                save_ext='jpg',
                                use_parse=True,
                                device=device)
