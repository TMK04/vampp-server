from exllamav2 import ExLlamaV2Tokenizer as _ExLlamaV2Tokenizer
import re
import torch

from .config import config

from typing import Union


class ExLlamaV2Tokenizer(_ExLlamaV2Tokenizer):
  comment_token: str = "#"
  comment_token_id: Union[int, None]

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    try:
      self.comment_token_id = self.tokenizer.encode(self.comment_token)[-1]
    except:
      self.comment_token_id = None


tokenizer = ExLlamaV2Tokenizer(config)

response_sep = f""" {tokenizer.eos_token}

{tokenizer.bos_token} """

stop_conditions = {tokenizer.bos_token_id, tokenizer.eos_token_id, "INPUT:", "RESPONSE:", "===="}


def StopStringsRe():
  stop_strings = []
  for stop_condition in stop_conditions:
    if isinstance(stop_condition, int):
      stop_string = tokenizer.decode(torch.tensor([stop_condition], dtype=torch.long),
                                     decode_special_tokens=True)
    else:
      stop_string = stop_condition
    stop_strings.append(re.escape(stop_string))
  print(stop_strings)
  stop_strings_re = re.compile("|".join(stop_strings))
  return stop_strings_re


stop_strings_re = StopStringsRe()
whitespace_re = re.compile(r"\s{2,}")


def cleanStopStrings(text):
  return whitespace_re.sub(" ", stop_strings_re.sub(" ", text).strip())
