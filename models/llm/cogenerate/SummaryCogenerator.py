import torch

from server.models.llm.cogenerate.fns import cogenerateMulti
from server.models.llm.tokenizer import response_sep

from .Cogenerator import Cogenerator, PretokenizeAppend, PretokenizePrepend
from . import TopicCogenerator


class SummaryCogenerator(Cogenerator):

  def __init__(self):
    super().__init__()
    self.output = None

  def setCurrentV(self):
    v = self.popCurrentV()
    self.output = v
    return True, None


append_pretokenized = """
========
Summary
* brief (max 1000 characters)
====
"""
prepend_pretokenized = """Summarize the following project pitches.""" + response_sep + f"""INPUT:
Example Pitch{TopicCogenerator.append_short}Example Topic{append_pretokenized}Example Summary""" + response_sep + """INPUT:
"""

prepend_pretokenized = PretokenizePrepend(prepend_pretokenized)
append_pretokenized = PretokenizeAppend(append_pretokenized)


def Wrap(content_pretokenized: torch.Tensor, topic_pretokenized: torch.Tensor):
  return torch.cat(
      (prepend_pretokenized, content_pretokenized, TopicCogenerator.append_short_pretokenized,
       topic_pretokenized, append_pretokenized),
      dim=1)


def cogenerate(content_pretokenized: torch.Tensor, topic_pretokenized: torch.Tensor):
  input = Wrap(content_pretokenized, topic_pretokenized)
  cogenerator = SummaryCogenerator()
  return cogenerateMulti(cogenerator, input, 512)
