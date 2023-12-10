from ..components import _Classifier, _ConvNormAct, _DepthwiseSeparableConv2d, _GAP, _NormAct, autocast, device, init_weights
import numpy as np
import os
from pathlib import Path
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class _Stem(nn.Sequential):
  def __init__(self, channels_in: int, channels_out: int):
    super().__init__()
    self.conv = _ConvNormAct(channels_in,
                             channels_out,
                             kernel_size=7,
                             stride=2,
                             padding=3,
                             bias=False)
    self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


class _DenseLayer(nn.Module):
  def __init__(self, channels_in: int, growth_rate: int, bn_size: int, k: int,
               drop_rate: float) -> None:
    super().__init__()
    channels_hidden = bn_size * growth_rate
    self.norm_act = _NormAct(channels_in)
    self.bn = _ConvNormAct(channels_in, channels_hidden, kernel_size=1, stride=1, bias=False)
    self.dsconv = _DepthwiseSeparableConv2d(channels_in=channels_hidden,
                                            channels_out=growth_rate,
                                            k=k)
    self.drop_rate = float(drop_rate)

  def forward(self, features: Tensor) -> Tensor:
    bottleneck_output = self.bn(self.norm_act(features))
    new_features = self.dsconv(bottleneck_output)

    if self.drop_rate > 0:
      new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
    return new_features


class _DenseBlock(nn.ModuleDict):
  def __init__(self, num_layers: int, channels_in: int, growth_rate: int, bn_size: int, k: int,
               drop_rate: int) -> None:
    super().__init__()
    for n in range(1, num_layers + 1):
      layer = _DenseLayer(channels_in=channels_in,
                          growth_rate=growth_rate,
                          bn_size=bn_size,
                          k=k,
                          drop_rate=drop_rate)
      channels_in += growth_rate
      self.add_module(f"denselayer{n}", layer)

  def forward(self, init_features: Tensor) -> Tensor:
    features = [init_features]
    for name, layer in self.items():
      new_features = layer(torch.cat(features, 1))
      features.append(new_features)
    return torch.cat(features, 1)


class _Transition(nn.Sequential):
  def __init__(self, channels_in: int, channels_out: int) -> None:
    super().__init__()
    self.norm_act = _NormAct(channels_in)
    self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=1, bias=False)
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class XDenseNet(nn.Module):
  def __init__(
      self,
      block_config: List[int] = [6, 12, 24, 16],
      num_features: int = 64,
      growth_rate: int = 32,
      bn_size: int = 4,
      k: int = 1,
      drop_rate: float = 0.1,
  ):
    super().__init__()
    self.features = nn.Sequential()
    # Grayscale input
    self.features.add_module("stem", _Stem(1, num_features))

    for n, num_layers in enumerate(block_config[:-1], 1):
      block = _DenseBlock(num_layers, num_features, growth_rate, bn_size, k, drop_rate)
      self.features.add_module(f"denseblock{n}", block)

      prev_num_features = num_features + num_layers * growth_rate
      num_features = prev_num_features // 2

      trans = _Transition(prev_num_features, num_features)
      self.features.add_module(f"transition{n}", trans)

    # Final dense block
    block = _DenseBlock(block_config[-1], num_features, growth_rate, bn_size, k, drop_rate)
    self.features.add_module(f"denseblock{len(block_config)}", block)

    init_weights(self.modules)

  def forward(self, x):
    return self.features(x)


class Head(nn.Sequential):
  def __init__(self,
               num_features: int,
               num_labels: int,
               num_hidden_layers: int = 0,
               gap_size: Tuple[int, int] = (1, 1),
               drop_rate: float = 0.2):
    super().__init__()
    self.gap = _GAP(num_features, gap_size)
    num_features *= gap_size[0] * gap_size[1]
    self.classifier = _Classifier(num_features, num_labels, num_hidden_layers, drop_rate)
    init_weights(self.modules)


wd = Path(__file__).parent

multitask_model = nn.Sequential(XDenseNet(block_config=[6, 12, 32, 32]), Head(1664, 4, 3, (1, 1)))
multitask_state_dict = torch.load(wd / "./multitask.pth")["model"]
for key in list(multitask_state_dict.keys()):
  if key.startswith('1.fc'):
    multitask_state_dict[key.replace('1.fc', '1.classifier.fc')] = multitask_state_dict.pop(key)
  elif key.startswith('1.classifier'):
    multitask_state_dict[key.replace('1.classifier',
                                     '1.classifier.classifier')] = multitask_state_dict.pop(key)
multitask_model.load_state_dict(multitask_state_dict)
multitask_model = multitask_model.to(device)
multitask_model.eval()

attire_model = nn.Sequential(XDenseNet(block_config=[3, 6, 12, 8]), Head(516, 1, 1, (1, 1)))
attire_model.load_state_dict(torch.load(wd / "./attire.pth")["model"])
attire_model = attire_model.to(device)
attire_model.eval()
