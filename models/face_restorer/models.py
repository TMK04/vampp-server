from basicsr.utils.registry import ARCH_REGISTRY
from pathlib import Path
import torch
from facelib.utils.face_restoration_helper import FaceRestoreHelper

from ..components import device

net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512,
                                      codebook_size=1024,
                                      n_head=8,
                                      n_layers=9,
                                      connect_list=['32', '64', '128', '256']).to(device)

ckpt_path = Path(__file__).parent / './CodeFormer/weights/CodeFormer/codeformer.pth'
checkpoint = torch.load(ckpt_path)['params_ema']
net.load_state_dict(checkpoint)
net.eval()

face_helper = FaceRestoreHelper(1,
                                face_size=512,
                                crop_ratio=(1, 1),
                                det_model='YOLOv5n',
                                save_ext='jpg',
                                use_parse=True,
                                device=device)
