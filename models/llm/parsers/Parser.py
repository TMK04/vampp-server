import torch

from server.models.llm.tokenizer import tokenizer

from typing import Dict


class Parser:

  def __init__(self):
    self.current_v = []
    self.output = None

  def appendCurrentV(self, token):
    self.current_v.append(token)

  def setCurrentV(self):
    """
    :return: (whether all keys have been set, next tokens)
    """
    raise NotImplementedError()

  def popCurrentV(self):
    v = tokenizer.decode(torch.tensor(self.current_v, dtype=torch.long))
    self.current_v = []
    return v


def PretokenizeAndWrap(prepend: torch.Tensor, input: str, append: torch.Tensor):
  pretokenized = torch.cat((prepend, PretokenizeInput(input), append), dim=1)
  return pretokenized


def PretokenizePrepend(prepend: str):
  pretokenized = tokenizer.encode(prepend, add_bos=True, encode_special_tokens=True)
  return pretokenized


def PretokenizeInput(input: str):
  pretokenized = tokenizer.encode(input)
  return pretokenized


def PretokenizeAppend(append: str):
  pretokenized = tokenizer.encode(f"\n{append}", encode_special_tokens=True)[:, 2:]
  return pretokenized


def PretokenizeDict(d: Dict):
  """
  Pretokenizes leaf values of a dict
  :param d: Dict of unknown depth
  """
  pretokenized = {}
  for k in d:
    if isinstance(d[k], dict):
      pretokenized[k] = PretokenizeDict(d[k])
      continue
    pretokenized[k] = PretokenizeAppend(d[k])
  return pretokenized
