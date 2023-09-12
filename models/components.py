import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

if torch.cuda.is_available():
  device = "cuda"
  torch.cuda.empty_cache()
else:
  device = "cpu"
print(device)


def autocast():
  return torch.autocast(device_type=device, dtype=torch.float16)


def infer(model, test_tensor):
  with torch.inference_mode():
    with autocast():
      y_pred = model(test_tensor.to(device))
      y_pred = y_pred > 0
  return y_pred.cpu().numpy()


def toTensor(ls):
  return torch.tensor(np.array(ls), dtype=torch.float32)


def N(i):
  """Convert Index to human-readable Number"""
  return i + 1


def init_weights(modules):
  # Official init from torch repo.
  for m in modules():
    if isinstance(m, nn.Conv2d):
      nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight, 1)
      nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
      nn.init.constant_(m.bias, 0)


def _Activation():
  return nn.SiLU(inplace=True)


class _NormAct(nn.Sequential):
  def __init__(self, channels_in: int):
    super().__init__()
    self.norm = nn.BatchNorm2d(channels_in)
    self.act = _Activation()


class _ConvNormAct(nn.Sequential):
  def __init__(self, channels_in: int, channels_out: int, **kwargs):
    super().__init__()
    self.conv = nn.Conv2d(channels_in, channels_out, **kwargs)
    self.norm_act = _NormAct(channels_out)


class _DepthwiseSeparableConv2d(nn.Sequential):
  def __init__(self, channels_in: int, channels_out: int, k: int = 1):
    super().__init__()
    channels_k = channels_in * k
    self.depthwise = nn.Conv2d(channels_in,
                               channels_k,
                               kernel_size=3,
                               padding=1,
                               groups=channels_in)
    self.norm = nn.BatchNorm2d(channels_k)
    self.pointwise = nn.Conv2d(channels_k, channels_out, kernel_size=1)


class _GAP(nn.Sequential):
  def __init__(self, num_features: int, size: Tuple[int, int] = (1, 1)):
    super().__init__()
    self.norm_act = _NormAct(num_features)
    self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(size)
    self.flatten = nn.Flatten()


def calculate_n_h(n_i: int, n_o: int):
  return int(np.sqrt(n_i * n_o))


class _LinearAct(nn.Sequential):
  def __init__(self, n_i: int, n_o: int, drop_rate: float = 0.1):
    super().__init__()
    self.dropout = nn.Dropout(drop_rate)
    self.linear = nn.Linear(n_i, n_o)
    self.act = _Activation()


class _Linear(nn.Sequential):
  def __init__(self, n_i: int, num_labels: int, drop_rate: float = 0.1):
    super().__init__()
    self.dropout = nn.Dropout(drop_rate)
    self.linear = nn.Linear(n_i, num_labels)


class _Classifier(nn.Sequential):
  def __init__(self,
               num_features: int,
               num_labels: int,
               num_hidden_layers: int,
               drop_rate: float = 0.1):
    super().__init__()
    if num_hidden_layers:
      n_h = calculate_n_h(num_features, num_labels)
      self.fc = nn.Sequential()
      self.fc.add_module("1", _LinearAct(num_features, n_h, drop_rate))
      for n in range(2, N(num_hidden_layers)):
        self.fc.add_module(str(n), _LinearAct(n_h, n_h, drop_rate))
      self.classifier = _Linear(n_h, num_labels, drop_rate)
    else:
      self.classifier = _Linear(num_features, num_labels, drop_rate)
